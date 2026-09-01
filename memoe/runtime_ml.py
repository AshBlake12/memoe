"""
MEMoE-RT, second path: architectures that keep experts as a ModuleList.

OLMoE in transformers 5.x stores experts as stacked tensors, so the first path
(memoe/runtime.py) replaces the whole MoE block. Most other checkpoints, and
anything shipping its own modelling code, keep a ModuleList of expert modules
instead. DeepSeek-V2-Lite is the case we target here: 26 MoE layers of 64
experts, 17.3 MB each, a 28.8 GB pool that does not fit on the GPU we have.

Rather than reimplement each architecture's routing, this path leaves the
model's own forward untouched and does two things:

  1. Points every offloaded expert's Linear weights at a view into the GPU
     staging buffer. Layer L always uses ring slot L mod (d+1) and position i
     within it, so those views are correct for the lifetime of the process and
     are set once at load time.

  2. Registers a pre-hook and a post-hook on each MoE block, which wait for the
     layer's weights to land and then issue the transfer for the layer d ahead.

The model computes exactly as it always did. It simply finds its weights in a
buffer that was filled a moment earlier.

Shared experts, dense layers and the router stay resident: shared experts fire
on every token, so offloading them would be pointless, and the dense first
layer has no MoE block to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from memoe.runtime import ExpertTier, RunStats


@dataclass
class MLSpec:
    block: nn.Module
    experts: nn.ModuleList
    n_experts: int
    d_model: int
    d_ff: int
    proj: tuple                     # (gate, up, down) attribute names
    layer_index: int                # index within the model's layer list


_TRIPLES = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))


def _triple(expert: nn.Module):
    for t in _TRIPLES:
        if all(hasattr(expert, n) for n in t):
            return t
    have = [n for n, _ in expert.named_children()]
    raise RuntimeError(f"Unrecognised expert layout, children: {have}")


def discover_modulelist_blocks(model: nn.Module) -> list[MLSpec]:
    """Every MoE block that keeps its experts as a ModuleList, in forward order."""
    specs = []
    for _, mod in model.named_modules():
        experts = getattr(mod, "experts", None)
        if not isinstance(experts, nn.ModuleList) or len(experts) < 2:
            continue
        e0 = experts[0]
        if not hasattr(e0, "named_children"):
            continue
        try:
            proj = _triple(e0)
        except RuntimeError:
            continue
        d_ff, d_model = getattr(e0, proj[0]).weight.shape
        specs.append(MLSpec(block=mod, experts=experts, n_experts=len(experts),
                            d_model=int(d_model), d_ff=int(d_ff), proj=proj,
                            layer_index=len(specs)))
    if not specs:
        raise RuntimeError(
            "No ModuleList MoE blocks found. If the model stores experts as "
            "stacked tensors, use memoe.runtime.tier_model instead.")
    return specs


class ModuleListTier(ExpertTier):
    """
    Same ring, streams and event bookkeeping as the fused path. Only the way
    weights are packed and handed back differs.
    """

    def build_modulelist(self, specs: list[MLSpec], residency: float,
                         device: torch.device) -> None:
        s0 = specs[0]
        self.d_model, self.d_ff = s0.d_model, s0.d_ff
        n = s0.d_model * s0.d_ff
        self.expert_elems = 3 * n
        self.gu_elems = 2 * n                       # gate and up together
        self.dtype = getattr(s0.experts[0], s0.proj[0]).weight.dtype
        esize = torch.tensor([], dtype=self.dtype).element_size()

        n_res = int(round(residency * s0.n_experts))
        self.n_resident = n_res
        self.stats = RunStats()
        self.stats.layers = len(specs)

        # --- pack into pinned host memory, one contiguous block per layer ---
        for spec in specs:
            g, u, d = spec.proj
            off_ids = list(range(n_res, spec.n_experts))

            if off_ids:
                host = torch.empty(len(off_ids) * self.expert_elems,
                                   dtype=self.dtype, pin_memory=True)
                pos = {}
                for i, e in enumerate(off_ids):
                    ex = spec.experts[e]
                    b = i * self.expert_elems
                    host[b:b + n].copy_(getattr(ex, g).weight.data.reshape(-1))
                    host[b + n:b + 2 * n].copy_(getattr(ex, u).weight.data.reshape(-1))
                    host[b + 2 * n:b + 3 * n].copy_(getattr(ex, d).weight.data.reshape(-1))
                    pos[e] = i
                    # Release the CPU copy as we go, so peak host memory stays
                    # near the size of the pool rather than twice it.
                    for pn in (g, u, d):
                        getattr(ex, pn).weight.data = torch.empty(0, dtype=self.dtype)
            else:
                host, pos = None, {}

            self.host.append(host)
            self.offloaded_ids.append(off_ids)
            self.pos_of.append(pos)
            self.res_gu.append(None)
            self.res_dn.append(None)
            self._layer_bytes.append(len(off_ids) * self.expert_elems * esize)

        self.stats.resident_experts = n_res
        self.stats.offloaded_experts = s0.n_experts - n_res
        self.stats.bytes_per_layer = max(self._layer_bytes) if self._layer_bytes else 0

        widest = max([len(o) for o in self.offloaded_ids] or [0])
        if widest:
            for _ in range(self.depth + 1):
                self.ring.append(torch.empty(widest * self.expert_elems,
                                             dtype=self.dtype, device=device))
                self.ready.append(torch.cuda.Event())
                ev = torch.cuda.Event()
                ev.record()
                self.consumed.append(ev)

    def bind_views(self, specs: list[MLSpec]) -> None:
        """
        Point each offloaded expert's Linear weights at its slice of the ring.

        Call this after the model has been moved to the GPU. Moving it
        afterwards would copy the tensors and silently break the aliasing.
        """
        n = self.d_model * self.d_ff
        for li, spec in enumerate(specs):
            if self.host[li] is None:
                continue
            g, u, d = spec.proj
            buf = self.ring[li % len(self.ring)]
            for e, i in self.pos_of[li].items():
                b = i * self.expert_elems
                ex = spec.experts[e]
                getattr(ex, g).weight.data = buf[b:b + n].view(self.d_ff, self.d_model)
                getattr(ex, u).weight.data = buf[b + n:b + 2 * n].view(self.d_ff, self.d_model)
                getattr(ex, d).weight.data = buf[b + 2 * n:b + 3 * n].view(self.d_model, self.d_ff)

    def attach_hooks(self, specs: list[MLSpec]) -> list:
        handles = []
        for li, spec in enumerate(specs):
            def pre(mod, inp, _li=li):
                self.begin_layer(_li)
            def post(mod, inp, out, _li=li):
                self.end_layer(_li)
            handles.append(spec.block.register_forward_pre_hook(pre))
            handles.append(spec.block.register_forward_hook(post))
        return handles


def tier_model_modulelist(model: nn.Module, residency: float, depth: int = 2,
                          device="cuda") -> ModuleListTier:
    """
    Partition a ModuleList-style MoE model across the two tiers.

    Unlike the fused path, this one moves the model to the GPU itself, because
    the weight views must be bound after the move rather than before.
    """
    device = torch.device(device)
    specs = discover_modulelist_blocks(model)
    tier = ModuleListTier(device=device, depth=depth)
    tier.build_modulelist(specs, residency, device)
    model.to(device)
    tier.bind_views(specs)
    tier.attach_hooks(specs)
    tier.specs = specs
    return tier


def tier_model_auto(model: nn.Module, residency: float, depth: int = 2,
                    device="cuda"):
    """
    Pick the right path for whatever architecture this is.

    Returns (tier, moved), where moved says whether the model is already on the
    GPU. The fused path expects the caller to move it afterwards; this one has
    already done so.
    """
    from memoe.runtime import tier_model

    for _, mod in model.named_modules():
        ex = getattr(mod, "experts", None)
        if ex is not None and getattr(ex, "gate_up_proj", None) is not None:
            return tier_model(model, residency, depth, device), False
    return tier_model_modulelist(model, residency, depth, device), True
