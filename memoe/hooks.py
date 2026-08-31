"""Architecture-agnostic capture of real routing decisions.

Attaches forward hooks to every router/gate Linear in an MoE checkpoint and
records the top-k expert indices per token per layer. Works without knowing
the model's class, by matching module names against common router names and
verifying the output width equals the expert count.

Usage:
    from memoe.hooks import capture
    with capture(model, n_experts=64, top_k=8) as cap:
        model.generate(...)
    trace = cap.trace(model_name="OLMoE-1B-7B")
    trace.save("results/olmoe_trace.npz")
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np

ROUTER_HINTS = ("gate", "router", "wg", "switch", "expert_gate", "gating")


def find_routers(model, n_experts=None):
    """Return [(name, module)] for modules that look like MoE routers."""
    out = []
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1].lower()
        if not any(h in leaf for h in ROUTER_HINTS):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.ndim != 2:
            continue
        if n_experts is not None and w.shape[0] != n_experts:
            continue
        if any(name.startswith(o + ".") for o, _ in out):
            continue
        out.append((name, mod))
    return out


class Capture:
    def __init__(self, model, n_experts, top_k, max_tokens=200_000):
        import torch  # local import: torch is not required for analysis
        self.torch = torch
        self.model = model
        self.n_experts = n_experts
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.routers = find_routers(model, n_experts)
        if not self.routers:
            raise RuntimeError(
                "no router modules found; pass n_experts=None or extend ROUTER_HINTS")
        self.buf = {i: [] for i in range(len(self.routers))}
        self._handles = []

    def _hook(self, idx):
        torch = self.torch
        def fn(mod, inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(logits):
                return
            logits = logits.reshape(-1, logits.shape[-1])
            if logits.shape[-1] != self.n_experts:
                return
            top = torch.topk(logits.float(), self.top_k, dim=-1).indices
            self.buf[idx].append(top.to("cpu", torch.int16).numpy())
        return fn

    def __enter__(self):
        for i, (_, mod) in enumerate(self.routers):
            self._handles.append(mod.register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def trace(self, model_name="captured"):
        from .trace import RoutingTrace
        layers = []
        for i in sorted(self.buf):
            if self.buf[i]:
                layers.append(np.concatenate(self.buf[i], axis=0))
        if not layers:
            raise RuntimeError("no routing captured; did a forward pass actually run?")
        n = min(a.shape[0] for a in layers)
        ids = np.stack([a[:n] for a in layers], axis=1).astype(np.int16)
        return RoutingTrace(ids=ids, model=model_name, n_experts=self.n_experts,
                            source="captured",
                            meta={"routers": [n_ for n_, _ in self.routers]})


@contextmanager
def capture(model, n_experts, top_k, max_tokens=200_000):
    c = Capture(model, n_experts, top_k, max_tokens)
    with c:
        yield c
