"""Trace-driven simulator for tiered (HBM + CXL) expert storage.

Timing model, per MoE layer per batch:

  1. The batch touches D distinct experts at this layer.
  2. Experts already in HBM are hits; the rest must be pulled over CXL.
  3. A miss may be covered by a speculative prefetch issued `depth`
     layers earlier, if the predictor named it AND the fetch had time
     to land within the prefetch window.
  4. Uncovered ("demand") misses are on the critical path, but they
     overlap with the expert GEMMs of the tokens that DID hit. Only the
     part of the transfer that outlives that compute is a stall.

  t_layer  = B * top_k * 2 * 3 * d_model * d_ff / eff_flops
  t_fetch  = bytes / BW_cxl + n_transfers * latency
  stall    = max(0, t_fetch_demand - t_layer * hit_token_fraction)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

from .policy import POLICIES

GB = 1024 ** 3


@dataclass
class SimResult:
    config: dict
    feasible: bool
    hbm_slots: int
    total_slots: int
    resident_frac: float
    hit_rate: float                # fraction of token-routings served from HBM
    expert_hit_rate: float         # fraction of distinct expert fetches avoided
    prefetch_coverage: float       # fraction of misses hidden by prefetch
    bytes_per_token: float
    bytes_per_batch: float
    compute_s: float
    stall_s: float
    overhead: float                # stall / compute
    throughput_tok_s: float
    baseline_tok_s: float
    slowdown: float
    n_batches: int
    notes: list = field(default_factory=list)

    def as_row(self):
        d = asdict(self)
        d.pop("notes")
        cfg = d.pop("config")
        return {**cfg, **d}


class Simulator:
    def __init__(self, model, mem, gpu, n_gpus=1, kv_reserve_gb=8.0,
                 activation_reserve_gb=4.0, hbm_slots_override=None):
        self.m = model
        self.mem = mem
        self.gpu = gpu
        self.n_gpus = n_gpus
        self.kv_reserve = kv_reserve_gb * GB
        self.act_reserve = activation_reserve_gb * GB
        # lets a sweep vary residency continuously instead of in GPU quanta
        self.hbm_slots_override = hbm_slots_override

    # ---------- capacity accounting ----------
    def hbm_bytes(self):
        return self.n_gpus * self.gpu.hbm_gb * GB

    def expert_budget_bytes(self):
        """HBM left for routed experts after pinned weights + KV + activations."""
        return (self.hbm_bytes() - self.m.resident_bytes
                - self.kv_reserve - self.act_reserve)

    def hbm_slots(self):
        if self.hbm_slots_override is not None:
            return int(self.hbm_slots_override)
        b = self.expert_budget_bytes()
        return int(max(0, b // self.m.expert_bytes))

    def cxl_bytes_per_s(self):
        """Each GPU/node carries its own CXL attachment, so expansion
        bandwidth scales with the deployment, exactly as HBM does."""
        return self.mem.cxl.bytes_per_s * self.n_gpus if self.mem.has_cxl else 0.0

    def cxl_capacity_bytes(self):
        return self.mem.cxl.capacity_bytes * self.n_gpus if self.mem.has_cxl else 0.0

    def cxl_slots(self):
        if not self.mem.has_cxl:
            return 0
        return int(self.cxl_capacity_bytes() // self.m.expert_bytes)

    def fits(self):
        """Can the checkpoint be held at all in this configuration?"""
        need = self.m.n_expert_instances
        return (self.hbm_slots() + self.cxl_slots()) >= need

    # ---------- timing ----------
    def t_layer_compute(self, batch_size):
        f = (2 * self.m.mats_per_expert * self.m.d_model * self.m.d_ff_expert
             * (self.m.top_k + self.m.n_shared_experts) * batch_size)
        return f / (self.gpu.eff_flops * self.n_gpus)

    def t_fetch(self, n_experts_fetched):
        if n_experts_fetched <= 0 or not self.mem.has_cxl:
            return 0.0
        return (n_experts_fetched * self.m.expert_bytes / self.cxl_bytes_per_s()
                + n_experts_fetched * self.mem.cxl.latency_ns * 1e-9)

    def critical_batch(self, expert_hit_rate=0.0, overhead=0.10):
        """Closed form: the batch size above which CXL expert fetch is hidden.

        Per MoE layer per batch, with E experts, top-k routing, miss rate
        (1-h), expert size S bytes and per-GPU compute F:

            stall   = (1-h) * E * S / BW
            compute = 6 * d * f * (k+shared) * B / F        [= k/E * B tokens
                                                              of work per expert]
        Setting stall <= eps * compute and solving for B:

            B* = (1-h) * E * F / (BW * (k + shared) * eps)

        Note B* is independent of expert size and of the GPU count, because
        both compute and CXL bandwidth scale with the deployment. What it
        does depend on is the hit rate: a hot tier that keeps 90% of routings
        in HBM cuts the required batch by 10x.
        """
        if not self.mem.has_cxl:
            return 0.0
        m = self.m
        f_per_expert_token = 2 * m.mats_per_expert * m.d_model * m.d_ff_expert
        num = (1 - expert_hit_rate) * m.n_experts * m.expert_bytes
        den = (self.cxl_bytes_per_s() * overhead
               * f_per_expert_token * (m.top_k + m.n_shared_experts)
               / (self.gpu.eff_flops * self.n_gpus))
        return num / den if den else float("inf")

    # ---------- main loop ----------
    def run(self, trace, batch_size=64, policy="static_popularity",
            profile=None, prefetch_depth=0, prefetch_width=0.0,
            label=None):
        """
        prefetch_depth  : how many layers ahead the prefetcher runs.
        prefetch_width  : fraction of each layer's experts speculatively
                          pulled in (from the popularity profile). 0 = off.
        """
        m, nL, nE = self.m, trace.n_layers, trace.n_experts
        prof = trace.counts() if profile is None else profile
        slots = self.hbm_slots()
        total_needed = nL * nE

        notes = []
        if not self.mem.has_cxl and slots < total_needed:
            notes.append("INFEASIBLE: HBM-only cannot hold the expert pool")
        if self.mem.is_modeled:
            notes.append("CXL tier is MODELED (pre-production pooling), not measured")

        cache = POLICIES[policy](slots if policy != "all_resident" else total_needed,
                                 nL, nE, profile=prof)

        # Speculative prefetch set. Prefetching the hottest experts is
        # pointless: those are exactly the ones the hot tier already holds.
        # The prefetcher must target the hottest experts that are NOT
        # resident, i.e. the head of the cold tail.
        pf_sets = []
        n_pf = int(round(prefetch_width * nE))
        res_mask = cache.mask()
        for l in range(nL):
            if not n_pf:
                pf_sets.append(set())
                continue
            order = np.argsort(prof[l])[::-1]
            cold = [int(e) for e in order if not res_mask[l, e]]
            pf_sets.append(set(cold[:n_pf]))

        t_layer = self.t_layer_compute(batch_size)
        pf_budget_bytes = prefetch_depth * t_layer * self.cxl_bytes_per_s()

        if cache.static:
            return self._run_static(trace, batch_size, cache, pf_sets, t_layer,
                                    pf_budget_bytes, policy, prefetch_depth,
                                    prefetch_width, label, notes, slots,
                                    total_needed)

        tot_route = tot_hit_route = 0
        tot_dist = tot_miss = tot_pf_cov = 0
        compute_s = stall_s = 0.0
        fetched_experts = 0
        nb = 0

        for _, counts in trace.batches(batch_size):
            nb += 1
            for l in range(nL):
                row = counts[l]
                need = np.nonzero(row)[0]
                if need.size == 0:
                    continue
                total_tok = int(row.sum())
                hit_tok = 0
                misses = []
                for e in need:
                    if cache.contains(l, e):
                        hit_tok += int(row[e])
                        cache.touch(l, e, int(row[e]))
                    else:
                        misses.append(int(e))

                # prefetch coverage, limited by the transfer window
                covered = [e for e in misses if e in pf_sets[l]]
                max_pf = int(pf_budget_bytes // m.expert_bytes) if prefetch_depth else 0
                covered = covered[:max_pf]
                demand = [e for e in misses if e not in set(covered)]

                fetched_experts += len(misses)
                compute_s += t_layer
                hit_frac = hit_tok / total_tok if total_tok else 0.0
                stall_s += max(0.0, self.t_fetch(len(demand)) - t_layer * hit_frac)

                tot_route += total_tok
                tot_hit_route += hit_tok
                tot_dist += int(need.size)
                tot_miss += len(misses)
                tot_pf_cov += len(covered)

                for e in misses:                # a fetched expert is used now
                    cache.touch(l, e, int(row[e]))
                    cache.admit(l, e)

        bytes_total = fetched_experts * m.expert_bytes
        tokens = nb * batch_size
        base_tok_s = tokens / compute_s if compute_s else 0.0
        real_tok_s = tokens / (compute_s + stall_s) if compute_s else 0.0

        cfg = {
            "model": m.name, "memory": self.mem.name, "gpu": self.gpu.name,
            "n_gpus": self.n_gpus, "batch_size": batch_size, "policy": policy,
            "prefetch_depth": prefetch_depth, "prefetch_width": prefetch_width,
            "trace_source": trace.source, "label": label or policy,
        }
        return SimResult(
            config=cfg,
            feasible=self.fits(),
            hbm_slots=slots,
            total_slots=slots + self.cxl_slots(),
            resident_frac=min(1.0, slots / total_needed),
            hit_rate=tot_hit_route / tot_route if tot_route else 0.0,
            expert_hit_rate=1 - tot_miss / tot_dist if tot_dist else 0.0,
            prefetch_coverage=tot_pf_cov / tot_miss if tot_miss else 0.0,
            bytes_per_token=bytes_total / tokens if tokens else 0.0,
            bytes_per_batch=bytes_total / nb if nb else 0.0,
            compute_s=compute_s, stall_s=stall_s,
            overhead=stall_s / compute_s if compute_s else 0.0,
            throughput_tok_s=real_tok_s, baseline_tok_s=base_tok_s,
            slowdown=base_tok_s / real_tok_s if real_tok_s else float("inf"),
            n_batches=nb, notes=notes,
        )

    # ---------- vectorised path (static residency) ----------
    def _run_static(self, trace, batch_size, cache, pf_sets, t_layer,
                    pf_budget_bytes, policy, prefetch_depth, prefetch_width,
                    label, notes, slots, total_needed):
        m = self.m
        nL, nE = trace.n_layers, trace.n_experts
        res = cache.mask()                               # (L, E)
        pf = np.zeros((nL, nE), dtype=bool)
        for l, st in enumerate(pf_sets):
            if st:
                pf[l, list(st)] = True
        max_pf = int(pf_budget_bytes // m.expert_bytes) if prefetch_depth else 0
        bw = self.cxl_bytes_per_s()
        lat = self.mem.cxl.latency_ns * 1e-9 if self.mem.has_cxl else 0.0

        nb = 0
        tok_s = hit_s = dist_s = miss_s = cov_s = 0
        stall_s = 0.0
        for _, counts in trace.iter_batch_counts(batch_size):
            nb += counts.shape[0]
            used = counts > 0
            tok = counts.sum(axis=2)
            hit_tok = (counts * res[None]).sum(axis=2)
            missm = used & ~res[None]
            miss = missm.sum(axis=2)
            cov = np.minimum((missm & pf[None]).sum(axis=2), max_pf)
            demand = miss - cov
            hit_frac = np.where(tok > 0, hit_tok / np.maximum(tok, 1), 0.0)
            if self.mem.has_cxl:
                t_f = demand * m.expert_bytes / bw + demand * lat
            else:
                t_f = np.zeros(demand.shape, dtype=float)
            stall_s += float(np.maximum(0.0, t_f - t_layer * hit_frac).sum())
            tok_s += int(tok.sum()); hit_s += int(hit_tok.sum())
            dist_s += int(used.sum()); miss_s += int(miss.sum())
            cov_s += int(cov.sum())
        compute_s = float(t_layer * nb * nL)

        tokens = nb * batch_size
        bytes_total = float(miss_s) * m.expert_bytes
        tok, hit_tok, dist, miss, cov = tok_s, hit_s, dist_s, miss_s, cov_s
        base = tokens / compute_s if compute_s else 0.0
        real = tokens / (compute_s + stall_s) if compute_s else 0.0
        cfg = {"model": m.name, "memory": self.mem.name, "gpu": self.gpu.name,
               "n_gpus": self.n_gpus, "batch_size": batch_size, "policy": policy,
               "prefetch_depth": prefetch_depth, "prefetch_width": prefetch_width,
               "trace_source": trace.source, "label": label or policy}
        return SimResult(
            config=cfg, feasible=self.fits(), hbm_slots=slots,
            total_slots=slots + self.cxl_slots(),
            resident_frac=min(1.0, slots / total_needed),
            hit_rate=float(hit_tok / max(1, tok)),
            expert_hit_rate=float(1 - miss / max(1, dist)),
            prefetch_coverage=float(cov / max(1, miss)),
            bytes_per_token=bytes_total / tokens if tokens else 0.0,
            bytes_per_batch=bytes_total / nb if nb else 0.0,
            compute_s=compute_s, stall_s=stall_s,
            overhead=stall_s / compute_s if compute_s else 0.0,
            throughput_tok_s=real, baseline_tok_s=base,
            slowdown=base / real if real else float("inf"),
            n_batches=nb, notes=notes)
