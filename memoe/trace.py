"""Routing traces: the record of which expert each token chose at each layer.

A trace is an int16 array `ids` of shape (n_tokens, n_moe_layers, top_k).
Produced either synthetically (`synth_trace`) or captured from a real
checkpoint via `memoe.hooks`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class RoutingTrace:
    ids: np.ndarray            # (n_tokens, n_moe_layers, top_k) int16
    model: str = "unknown"
    n_experts: int = 0
    source: str = "synthetic"  # "synthetic" | "captured"
    meta: dict = field(default_factory=dict)

    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.ids.ndim != 3:
            raise ValueError(f"ids must be 3-D (tokens, layers, k), got {self.ids.shape}")
        if not self.n_experts:
            self.n_experts = int(self.ids.max()) + 1

    @property
    def n_tokens(self): return self.ids.shape[0]
    @property
    def n_layers(self): return self.ids.shape[1]
    @property
    def top_k(self): return self.ids.shape[2]
    @property
    def is_real(self): return self.source == "captured"

    def counts(self):
        """(n_layers, n_experts) token-assignment counts."""
        out = np.zeros((self.n_layers, self.n_experts), dtype=np.int64)
        for l in range(self.n_layers):
            out[l] = np.bincount(self.ids[:, l, :].ravel(), minlength=self.n_experts)
        return out

    def _slab(self, b0, b1, batch_size):
        """Counts for batches [b0, b1): (n, n_layers, n_experts) int32."""
        n = b1 - b0
        L, E = self.n_layers, self.n_experts
        chunk = self.ids[b0 * batch_size:b1 * batch_size].reshape(
            n, batch_size, L, self.top_k)
        out = np.empty((n, L, E), dtype=np.int32)
        off = (np.arange(n, dtype=np.int64) * E)[:, None]
        for l in range(L):
            flat = chunk[:, :, l, :].reshape(n, -1).astype(np.int64) + off
            out[:, l, :] = np.bincount(flat.ravel(), minlength=n * E).reshape(n, E)
        return out

    def n_batches(self, batch_size):
        return self.n_tokens // batch_size

    def iter_batch_counts(self, batch_size, budget_bytes=256 << 20):
        """Stream (first_batch_index, counts_slab) so that a small batch size
        on a 94-layer model does not materialise gigabytes of counters."""
        nb = self.n_batches(batch_size)
        if nb == 0:
            raise ValueError(f"batch_size {batch_size} > trace length {self.n_tokens}")
        if batch_size in self._cache:
            yield 0, self._cache[batch_size]
            return
        per = self.n_layers * self.n_experts * 4
        step = max(1, int(budget_bytes // max(1, per)))
        if step >= nb:                      # small enough to keep around
            arr = self._slab(0, nb, batch_size)
            self._cache[batch_size] = arr
            yield 0, arr
            return
        for b0 in range(0, nb, step):
            yield b0, self._slab(b0, min(b0 + step, nb), batch_size)

    def batch_counts(self, batch_size):
        """(n_batches, n_layers, n_experts) int32 counts, memoised.

        Raises if the array would be unreasonably large; use
        iter_batch_counts for those cases."""
        nb = self.n_batches(batch_size)
        size = nb * self.n_layers * self.n_experts * 4
        if size > (512 << 20):
            raise MemoryError(
                f"batch_counts would need {size/2**30:.1f} GiB; "
                "use iter_batch_counts(batch_size) instead")
        if batch_size not in self._cache:
            self._cache[batch_size] = self._slab(0, nb, batch_size)
        return self._cache[batch_size]

    def batches(self, batch_size):
        """Yield (batch_index, counts[n_layers, n_experts]) for each batch."""
        for b0, slab in self.iter_batch_counts(batch_size):
            for i, c in enumerate(slab):
                yield b0 + i, c

    def distinct_per_batch(self, batch_size):
        """Mean number of distinct experts touched per layer per batch."""
        tot, n = 0.0, 0
        for _, slab in self.iter_batch_counts(batch_size):
            tot += float((slab > 0).sum(axis=2).sum())
            n += slab.shape[0] * slab.shape[1]
        return tot / n if n else 0.0

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=self.ids, model=self.model,
                            n_experts=self.n_experts, source=self.source)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return RoutingTrace(ids=z["ids"], model=str(z["model"]),
                            n_experts=int(z["n_experts"]), source=str(z["source"]))


def _zipf_probs(n, s, rng=None):
    p = 1.0 / np.arange(1, n + 1) ** s
    p /= p.sum()
    if rng is not None:
        rng.shuffle(p)      # popularity rank is arbitrary w.r.t. expert index
    return p


def synth_trace(n_tokens, n_layers, n_experts, top_k, zipf_s=1.0,
                seed=0, drift=0.0, model="synthetic", block=4096):
    """Synthetic routing trace with controllable popularity skew.

    zipf_s = 0.0     -> uniform routing (worst case for any caching scheme)
    zipf_s ~ 0.8-1.2 -> skew of the kind trained MoE routers exhibit
    drift            -> rotates each layer's popularity ranking every `block`
                        tokens, modelling non-stationary routing over a long
                        sequence. drift=0 means a perfectly stationary router.

    Sampling uses the Gumbel-top-k trick, which draws exactly top_k distinct
    experts per token from the categorical distribution p, vectorised over
    the whole block.
    """
    rng = np.random.default_rng(seed)
    if top_k > n_experts:
        raise ValueError("top_k > n_experts")
    ids = np.empty((n_tokens, n_layers, top_k), dtype=np.int16)
    base = [_zipf_probs(n_experts, zipf_s, rng) for _ in range(n_layers)]
    for l in range(n_layers):
        p = base[l]
        for s0 in range(0, n_tokens, block):
            e0 = min(s0 + block, n_tokens)
            pp = np.roll(p, int(rng.normal(0, drift * n_experts))) if drift else p
            logp = np.log(pp / pp.sum() + 1e-300)
            g = rng.gumbel(size=(e0 - s0, n_experts))
            idx = np.argpartition(-(logp + g), top_k - 1, axis=1)[:, :top_k]
            ids[s0:e0, l] = idx.astype(np.int16)
    return RoutingTrace(ids=ids, model=model, n_experts=n_experts,
                        source="synthetic",
                        meta={"zipf_s": zipf_s, "seed": seed, "drift": drift})
