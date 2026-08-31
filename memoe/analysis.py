"""Closed-form analyses that motivate the simulation."""
from __future__ import annotations
import numpy as np
import pandas as pd

GB = 1024 ** 3
MB = 1024 ** 2


def capacity_table(models, gpu, hbm_gb=None, cxl_gb=512.0):
    """How many GPUs does each model need, with and without CXL expansion?"""
    hbm_gb = hbm_gb or gpu.hbm_gb
    rows = []
    for m in models:
        tot = m.total_bytes / GB
        gpus_hbm = int(np.ceil(tot / hbm_gb))
        # with CXL, one node holds hbm + cxl bytes
        per_node = hbm_gb + cxl_gb
        gpus_cxl = int(np.ceil(tot / per_node))
        rows.append({
            "model": m.name,
            "params_B": round(m.total_params / 1e9, 1),
            "active_B": round(m.active_params_per_token / 1e9, 2),
            "experts/layer": m.n_experts,
            "top_k": m.top_k,
            "expert_MB": round(m.expert_bytes / MB, 1),
            "expert_pool_GB": round(m.routed_expert_bytes / GB, 1),
            "pinned_GB": round(m.resident_bytes / GB, 1),
            "total_GB": round(tot, 1),
            "expert_frac": round(m.expert_fraction, 3),
            "gpus_hbm_only": gpus_hbm,
            "nodes_with_cxl": gpus_cxl,
            "capacity_gain": round(per_node / hbm_gb, 2),
            "gpu_reduction": round(gpus_hbm / max(1, gpus_cxl), 2),
        })
    return pd.DataFrame(rows)


def batch_regime(model, trace, batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                 hit_rate=0.0):
    """Per-token CXL traffic as a function of batch size.

    This is the amortisation argument: the cost of pulling an expert is paid
    once per batch, not once per token, so per-token traffic collapses as the
    batch grows.
    """
    rows = []
    for B in batch_sizes:
        if B > trace.n_tokens:
            continue
        D = trace.distinct_per_batch(B)
        miss = D * (1 - hit_rate)
        per_batch = miss * model.expert_bytes * trace.n_layers
        rows.append({
            "batch_size": B,
            "distinct_experts_per_layer": round(D, 2),
            "coverage_frac": round(D / trace.n_experts, 3),
            "bytes_per_batch_MB": round(per_batch / MB, 2),
            "bytes_per_token_MB": round(per_batch / B / MB, 3),
            "amortisation_vs_b1": None,
        })
    if rows:
        b1 = rows[0]["bytes_per_token_MB"]
        for r in rows:
            r["amortisation_vs_b1"] = round(b1 / r["bytes_per_token_MB"], 1) if r["bytes_per_token_MB"] else None
    return pd.DataFrame(rows)


def hitrate_requirement(model, mem, gpu, trace, batch_sizes=(8, 32, 128, 512),
                        overheads=(0.05, 0.10, 0.25, 0.50), n_gpus=1):
    """Minimum HBM hit rate needed to keep the CXL stall under a target.

    Solving  (1-h) * D * expert_bytes / BW  <=  eps * t_layer_compute
    """
    if not mem.has_cxl:
        raise ValueError("hit-rate requirement is only meaningful with a CXL tier")
    bw = mem.cxl.bytes_per_s
    rows = []
    for B in batch_sizes:
        if B > trace.n_tokens:
            continue
        D = trace.distinct_per_batch(B)
        t_c = (2 * model.mats_per_expert * model.d_model * model.d_ff_expert
               * (model.top_k + model.n_shared_experts) * B) / (gpu.eff_flops * n_gpus)
        full = D * model.expert_bytes / bw
        for eps in overheads:
            h = 1 - (eps * t_c) / full if full else 1.0
            rows.append({
                "batch_size": B,
                "target_overhead": eps,
                "t_compute_us": round(t_c * 1e6, 1),
                "t_fetch_all_us": round(full * 1e6, 1),
                "required_hit_rate": round(float(np.clip(h, 0.0, 1.0)), 4),
                "achievable": bool(h <= 1.0),
            })
    return pd.DataFrame(rows)


def prefetch_depth(model, mem, gpu, trace, batch_sizes=(8, 32, 128, 512),
                   hit_rates=(0.5, 0.8, 0.9, 0.95), n_gpus=1):
    """How many layers ahead must a prefetch be issued to fully hide the fetch?"""
    if not mem.has_cxl:
        raise ValueError("prefetch depth is only meaningful with a CXL tier")
    bw, lat = mem.cxl.bytes_per_s, mem.cxl.latency_ns * 1e-9
    rows = []
    for B in batch_sizes:
        if B > trace.n_tokens:
            continue
        D = trace.distinct_per_batch(B)
        t_c = (2 * model.mats_per_expert * model.d_model * model.d_ff_expert
               * (model.top_k + model.n_shared_experts) * B) / (gpu.eff_flops * n_gpus)
        for h in hit_rates:
            miss = D * (1 - h)
            t_f = miss * model.expert_bytes / bw + miss * lat
            rows.append({
                "batch_size": B,
                "hit_rate": h,
                "missing_experts": round(miss, 2),
                "t_fetch_us": round(t_f * 1e6, 1),
                "t_layer_us": round(t_c * 1e6, 1),
                "layers_of_lookahead": int(np.ceil(t_f / t_c)) if t_c else -1,
                "feasible_depth<=4": bool(t_c and np.ceil(t_f / t_c) <= 4),
            })
    return pd.DataFrame(rows)
