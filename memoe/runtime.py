"""
MEMoE-RT: a real tiered-memory runtime for Mixture-of-Experts inference.

Not a simulator. It runs an actual checkpoint with most of its expert weights
held outside GPU memory, streams them in over PCIe while attention computes,
and reports wall-clock throughput.

The design follows from what we measured:

  Coverage saturates at serving batch sizes, so every expert in a layer will
  be needed. Prediction is therefore trivial and the whole problem is
  bandwidth window scheduling. We do not fetch experts individually on demand.
  We fetch every non-resident expert of layer L+d as one contiguous transfer,
  issued d layers early, on a dedicated copy stream.

  One large transfer also happens to be the fastest thing the hardware can do.
  Measured on the target box: 4 MB reaches 33.05 GB/s, 12 MB reaches 36.67,
  and the plateau is 40.81. A whole-layer transfer sits on the plateau.

  Which experts are resident does not matter. Our skew sweep showed fetch hit
  rate equals the resident fraction at every skew including uniform routing,
  so the resident set here is just the low indices. There is deliberately no
  popularity ranking anywhere in this file.

Host DRAM over PCIe stands in for a CXL tier. It is not CXL. It is the same
shape of problem: a large, slow, byte-addressable tier behind a link, and at
PCIe 5.0 x16 it lands between our measured single-channel figure of 16.88 GB/s
and our scaled four-channel figure of 67.5 GB/s.

Written against the fused expert layout used in transformers 5.x, where a
block holds stacked `gate_up_proj` (E, 2*d_ff, d_model) and `down_proj`
(E, d_model, d_ff) tensors rather than a ModuleList of expert modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

@dataclass
class MoESpec:
    block: nn.Module          # the sparse MoE block
    experts: nn.Module        # holds the fused weight tensors
    router: nn.Module         # returns (logits, weights, indices)
    n_experts: int
    d_model: int
    d_ff: int
    gu_shape: tuple[int, int]
    dn_shape: tuple[int, int]


def discover_moe_blocks(model: nn.Module) -> list[MoESpec]:
    """Return one MoESpec per MoE block, in forward order."""
    specs: list[MoESpec] = []
    for _, mod in model.named_modules():
        experts = getattr(mod, "experts", None)
        router = getattr(mod, "gate", None)
        if router is None:
            router = getattr(mod, "router", None)
        if experts is None or router is None:
            continue
        gu = getattr(experts, "gate_up_proj", None)
        dn = getattr(experts, "down_proj", None)
        if gu is None or dn is None:
            continue
        if gu.dim() != 3 or dn.dim() != 3:
            continue

        specs.append(MoESpec(
            block=mod, experts=experts, router=router,
            n_experts=int(gu.shape[0]), d_model=int(gu.shape[2]),
            d_ff=int(dn.shape[2]),
            gu_shape=(int(gu.shape[1]), int(gu.shape[2])),
            dn_shape=(int(dn.shape[1]), int(dn.shape[2])),
        ))

    if not specs:
        raise RuntimeError(
            "Found no MoE blocks with fused gate_up_proj/down_proj tensors. "
            "Print [n for n, _ in model.named_modules()] and check the layout."
        )
    return specs


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

@dataclass
class RunStats:
    layers: int = 0
    resident_experts: int = 0
    offloaded_experts: int = 0
    bytes_per_layer: int = 0
    bytes_streamed: int = 0
    stall_seconds: float = 0.0
    stalls_by_layer: list = field(default_factory=list)

    def summary(self, wall_seconds: float, tokens: int) -> dict:
        return {
            "tokens": tokens,
            "wall_seconds": round(wall_seconds, 4),
            "tokens_per_second": round(tokens / wall_seconds, 1) if wall_seconds else 0.0,
            "gb_streamed": round(self.bytes_streamed / 1e9, 3),
            "achieved_gbs": (round(self.bytes_streamed / 1e9 / wall_seconds, 2)
                             if wall_seconds else 0.0),
            "stall_seconds": round(self.stall_seconds, 4),
            "stall_fraction": (round(self.stall_seconds / wall_seconds, 4)
                               if wall_seconds else 0.0),
        }


# --------------------------------------------------------------------------
# the tier
# --------------------------------------------------------------------------

class ExpertTier:
    """
    Offloaded experts live in pinned host memory, one contiguous block per
    layer. A ring of GPU staging buffers receives them.

    Slot arithmetic: with a ring of depth+1 buffers, layer L uses slot
    L % (depth+1). The prefetch issued at layer L targets L+depth, whose slot
    is (L-1) % (depth+1), belonging to the previous layer, which has already
    computed. An event on the compute stream makes that reuse safe.

    All waiting is GPU-side with stream events, never host synchronisation, so
    the copy really does overlap compute. Stall time is measured by recording a
    CUDA event either side of the wait and reading elapsed time after the pass.
    """

    def __init__(self, device: torch.device, depth: int = 2):
        self.device = device
        self.depth = depth
        self.copy_stream = torch.cuda.Stream(device=device)

        self.host = []            # per layer: pinned flat tensor or None
        self.offloaded_ids = []   # per layer: list of expert indices
        self.pos_of = []          # per layer: expert index -> position
        self.res_gu = []          # per layer: resident gate_up or None
        self.res_dn = []          # per layer: resident down or None
        self.n_resident = 0

        self.ring = []
        self.ready = []
        self.consumed = []
        self._wait_events = []

        self.gu_shape = (0, 0)
        self.dn_shape = (0, 0)
        self.gu_elems = 0
        self.expert_elems = 0
        self.dtype = torch.bfloat16
        self.stats = RunStats()
        self._layer_bytes = []

    # -- setup, done once at load time -------------------------------------

    def build(self, specs, residency: float) -> None:
        s0 = specs[0]
        self.gu_shape, self.dn_shape = s0.gu_shape, s0.dn_shape
        self.gu_elems = s0.gu_shape[0] * s0.gu_shape[1]
        dn_elems = s0.dn_shape[0] * s0.dn_shape[1]
        self.expert_elems = self.gu_elems + dn_elems
        self.dtype = s0.experts.gate_up_proj.dtype
        esize = torch.tensor([], dtype=self.dtype).element_size()

        n_res = int(round(residency * s0.n_experts))
        self.n_resident = n_res
        self.stats.layers = len(specs)

        for spec in specs:
            gu = spec.experts.gate_up_proj.data
            dn = spec.experts.down_proj.data
            off_ids = list(range(n_res, spec.n_experts))

            if n_res:
                self.res_gu.append(gu[:n_res].contiguous().to(self.device))
                self.res_dn.append(dn[:n_res].contiguous().to(self.device))
            else:
                self.res_gu.append(None)
                self.res_dn.append(None)

            if off_ids:
                host = torch.empty(len(off_ids) * self.expert_elems,
                                   dtype=self.dtype, pin_memory=True)
                pos = {}
                for i, e in enumerate(off_ids):
                    base = i * self.expert_elems
                    host[base:base + self.gu_elems].copy_(gu[e].reshape(-1))
                    host[base + self.gu_elems:base + self.expert_elems].copy_(
                        dn[e].reshape(-1))
                    pos[e] = i
            else:
                host, pos = None, {}

            # drop the fused weights but leave the modules in place. the
            # rest of .to(device) then walks the tree normally and finds
            # empty shells here instead of 12 GB of experts.
            spec.experts.gate_up_proj.data = torch.empty(0, dtype=self.dtype)
            spec.experts.down_proj.data = torch.empty(0, dtype=self.dtype)

            self.host.append(host)
            self.offloaded_ids.append(off_ids)
            self.pos_of.append(pos)
            self._layer_bytes.append(len(off_ids) * self.expert_elems * esize)

        self.stats.resident_experts = n_res
        self.stats.offloaded_experts = s0.n_experts - n_res
        self.stats.bytes_per_layer = max(self._layer_bytes) if self._layer_bytes else 0

        widest = max([len(o) for o in self.offloaded_ids] or [0])
        if widest:
            for _ in range(self.depth + 1):
                self.ring.append(torch.empty(widest * self.expert_elems,
                                             dtype=self.dtype, device=self.device))
                self.ready.append(torch.cuda.Event())
                ev = torch.cuda.Event()
                ev.record()
                self.consumed.append(ev)

    # -- called on every forward --------------------------------------------

    def reset_stats(self) -> None:
        self.stats.bytes_streamed = 0
        self.stats.stall_seconds = 0.0
        self.stats.stalls_by_layer = [0.0] * self.stats.layers
        self._wait_events = []

    def _slot(self, layer: int) -> int:
        return layer % len(self.ring)

    def prefetch(self, layer: int) -> None:
        if not self.ring or layer >= len(self.host):
            return
        host = self.host[layer]
        if host is None:
            return
        s = self._slot(layer)
        self.copy_stream.wait_event(self.consumed[s])
        with torch.cuda.stream(self.copy_stream):
            self.ring[s][:host.numel()].copy_(host, non_blocking=True)
            self.ready[s].record(self.copy_stream)
        self.stats.bytes_streamed += self._layer_bytes[layer]

    def warmup(self) -> None:
        for l in range(min(self.depth, len(self.host))):
            self.prefetch(l)

    def begin_layer(self, layer: int) -> None:
        """Make the compute stream wait for this layer's weights, then issue
        the transfer for the layer d ahead."""
        if self.ring and self.host[layer] is not None:
            cur = torch.cuda.current_stream(self.device)
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(cur)
            cur.wait_event(self.ready[self._slot(layer)])
            b.record(cur)
            self._wait_events.append((a, b))
        self.prefetch(layer + self.depth)

    def end_layer(self, layer: int) -> None:
        if self.ring and self.host[layer] is not None:
            self.consumed[self._slot(layer)].record()

    def collect_stalls(self) -> None:
        """Read the recorded event pairs. Call after torch.cuda.synchronize()."""
        n = self.stats.layers or 1
        if not self.stats.stalls_by_layer:
            self.stats.stalls_by_layer = [0.0] * n
        for i, pair in enumerate(self._wait_events):
            a, b = pair
            ms = a.elapsed_time(b)
            self.stats.stall_seconds += ms / 1000.0
            layer = i % n
            if layer < len(self.stats.stalls_by_layer):
                self.stats.stalls_by_layer[layer] += ms / 1000.0
        self._wait_events = []

    def weights(self, layer: int, expert: int):
        if expert < self.n_resident:
            return self.res_gu[layer][expert], self.res_dn[layer][expert]
        i = self.pos_of[layer][expert]
        buf = self.ring[self._slot(layer)]
        base = i * self.expert_elems
        gu = buf[base:base + self.gu_elems].view(self.gu_shape[0], self.gu_shape[1])
        dn = buf[base + self.gu_elems:base + self.expert_elems].view(
            self.dn_shape[0], self.dn_shape[1])
        return gu, dn


# --------------------------------------------------------------------------
# the replacement block
# --------------------------------------------------------------------------

class TieredMoE(nn.Module):
    """
    Drop-in replacement for the sparse MoE block. The maths is copied from the
    reference implementation so output is comparable; only the source of the
    weights differs.
    """

    def __init__(self, spec: MoESpec, tier: ExpertTier, layer: int):
        super().__init__()
        self.gate = spec.router
        self.n_experts = spec.n_experts
        act = getattr(spec.experts, "act_fn", None)
        self.act_fn = act if act is not None else F.silu
        self.tier = tier
        self.layer = layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.gate(x)

        self.tier.begin_layer(self.layer)

        out = torch.zeros_like(x)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.n_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().cpu().view(-1).tolist()

        for e in hit:
            if e == self.n_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            cur = x[token_idx]
            w_gu, w_dn = self.tier.weights(self.layer, e)
            gate, up = F.linear(cur, w_gu).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, w_dn) * top_k_weights[token_idx, top_k_pos, None]
            out.index_add_(0, token_idx, h.to(out.dtype))

        self.tier.end_layer(self.layer)
        return out.reshape(batch_size, seq_len, hidden_dim)


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def tier_model(model: nn.Module, residency: float, depth: int = 2,
               device="cuda") -> ExpertTier:
    """
    Partition the experts and swap in tiered blocks. Call this while the model
    is still on CPU, then move the remainder to the GPU.
    """
    device = torch.device(device)
    specs = discover_moe_blocks(model)
    tier = ExpertTier(device=device, depth=depth)
    tier.build(specs, residency)
    for i, spec in enumerate(specs):
        parent, attr = _locate(model, spec.block)
        setattr(parent, attr, TieredMoE(spec, tier, i))
    return tier


def _locate(root: nn.Module, target: nn.Module):
    for _, mod in root.named_modules():
        for cname, child in mod.named_children():
            if child is target:
                return mod, cname
    raise RuntimeError("Could not locate the MoE block inside the model tree.")
