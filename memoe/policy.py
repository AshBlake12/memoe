"""Expert placement / caching policies over the HBM expert slots.

Address space: an expert instance is the pair (layer, expert_id), flattened
to layer * n_experts + expert_id. A policy decides which of these live in
HBM at any moment; everything else lives in CXL-attached memory.
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np


class Cache:
    """base class for every placement policy. capacity is counted in
    expert slots, not bytes, because every expert in a checkpoint is the
    same size."""
    name = "base"
    static = False          # True => residency never changes at runtime

    def __init__(self, capacity, n_layers, n_experts, profile=None):
        self.capacity = int(max(0, capacity))
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.profile = profile          # (n_layers, n_experts) profiling counts
        self.resident = set()
        self.evictions = 0
        self._init_residency()

    def key(self, layer, e):
        return layer * self.n_experts + int(e)

    def _init_residency(self):
        pass

    def contains(self, layer, e):
        return self.key(layer, e) in self.resident

    def admit(self, layer, e):
        """Bring (layer,e) into HBM, evicting if needed."""
        raise NotImplementedError

    def touch(self, layer, e, n=1):
        pass

    def mask(self):
        """(n_layers, n_experts) bool residency map. Static policies only."""
        m = np.zeros(self.n_layers * self.n_experts, dtype=bool)
        r = list(self.resident)
        if r:
            m[np.asarray(r, dtype=np.int64)] = True
        return m.reshape(self.n_layers, self.n_experts)


class StaticPopularity(Cache):
    """Profile once, pin the globally hottest experts. No runtime evictions.

    Cheapest possible policy: zero bookkeeping on the critical path, and it
    is the natural fit for CXL because the placement decision is made at
    load time, not per token.
    """
    name = "static_popularity"
    static = True

    def _init_residency(self):
        if self.profile is None:
            raise ValueError("StaticPopularity needs a profiling trace")
        p = np.asarray(self.profile, dtype=float)
        flat = p.ravel()
        order = np.argsort(flat)[::-1][:self.capacity]
        self.resident = set(int(i) for i in order)

    def admit(self, layer, e):
        return  # nothing is ever promoted at runtime


class BalancedStatic(StaticPopularity):
    """Pin the hottest experts, but with a per-layer quota.

    Prevents a single hot layer from eating the whole HBM budget, which
    matters because a layer with zero resident experts stalls every batch.
    """
    name = "balanced_static"
    static = True

    def _init_residency(self):
        p = np.asarray(self.profile, dtype=float)
        per_layer = self.capacity // self.n_layers
        rem = self.capacity - per_layer * self.n_layers
        res = set()
        for l in range(self.n_layers):
            q = per_layer + (1 if l < rem else 0)
            order = np.argsort(p[l])[::-1][:q]
            res.update(self.key(l, e) for e in order)
        self.resident = res


class LRU(Cache):
    name = "lru"

    def _init_residency(self):
        self.od = OrderedDict()

    def contains(self, layer, e):
        k = self.key(layer, e)
        if k in self.od:
            self.od.move_to_end(k)
            return True
        return False

    def admit(self, layer, e):
        k = self.key(layer, e)
        self.od[k] = 1
        self.od.move_to_end(k)
        while len(self.od) > self.capacity:
            self.od.popitem(last=False)
            self.evictions += 1

    @property
    def resident(self):
        return set(self.od)

    @resident.setter
    def resident(self, v):
        pass


class LFU(Cache):
    """Sampled LFU (Redis-style): on eviction, sample S resident entries and
    drop the least frequently used of them. An exact LFU needs a scan of the
    whole resident set per eviction, which no real critical path could afford
    either."""
    name = "lfu"
    SAMPLE = 16

    def _init_residency(self):
        self.freq = {}
        self.res = set()
        self.order = []
        self._rng = np.random.default_rng(0)

    def contains(self, layer, e):
        return self.key(layer, e) in self.res

    def touch(self, layer, e, n=1):
        k = self.key(layer, e)
        self.freq[k] = self.freq.get(k, 0) + n

    def admit(self, layer, e):
        k = self.key(layer, e)
        if k in self.res:
            return
        self.res.add(k); self.order.append(k)
        self.freq.setdefault(k, 1)
        while len(self.res) > self.capacity:
            if len(self.order) > 4 * len(self.res):
                self.order = [x for x in self.order if x in self.res]
            n = len(self.order)
            idx = self._rng.integers(0, n, size=min(self.SAMPLE, n))
            cand = [self.order[i] for i in idx if self.order[i] in self.res]
            if not cand:
                cand = [next(iter(self.res))]
            victim = min(cand, key=lambda x: self.freq.get(x, 0))
            self.res.discard(victim)
            self.evictions += 1

    @property
    def resident(self):
        return self.res

    @resident.setter
    def resident(self, v):
        self.res = set(v) if v else set()


class AllResident(Cache):
    """Idealised HBM-only baseline: every expert is in HBM. Only feasible
    if the HBM budget actually fits the whole checkpoint."""
    name = "all_resident"
    static = True

    def contains(self, layer, e):
        return True

    def mask(self):
        return np.ones((self.n_layers, self.n_experts), dtype=bool)

    def admit(self, layer, e):
        return


POLICIES = {
    "static_popularity": StaticPopularity,
    "balanced_static": BalancedStatic,
    "lru": LRU,
    "lfu": LFU,
    "all_resident": AllResident,
}
