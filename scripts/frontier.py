#!/usr/bin/env python3
"""The feasibility frontier.

CXL sustained bandwidth is ~50 GB/s against HBM's ~3350 GB/s: a 64x gap. So
expert offload cannot be a streaming strategy. The question that decides
whether MEMoE is useful is narrower and sharper:

    how much of the expert pool can be pushed to CXL before the stall
    exceeds a given fraction of compute?

Answer that, and the capacity multiplier follows directly:
    achievable_multiplier = 1 / resident_fraction
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memoe import *
from memoe.analysis import GB

RES = ROOT / "results"; TAB = RES / "tables"; FIG = RES / "figures"
for d in (TAB, FIG): d.mkdir(parents=True, exist_ok=True)

MODELS = ["olmoe", "mixtral_8x7b", "qwen3_235b", "deepseek_v3"]
TOK = {"olmoe": 32768, "mixtral_8x7b": 32768, "qwen3_235b": 8192, "deepseek_v3": 8192}
TARGETS = [0.05, 0.10, 0.25, 0.50, 1.00]


def max_offload(m, mem, gpu, tr, batch, eps, policy="balanced_static",
                depth=0, width=0.0, tol=1e-3):
    """Largest fraction of the expert pool that can sit in CXL with
    stall/compute <= eps. Binary search on resident fraction."""
    total = m.n_moe_layers * m.n_experts

    def overhead(frac):
        sim = Simulator(m, mem, gpu, n_gpus=1,
                        hbm_slots_override=int(round(frac * total)))
        return sim.run(tr, batch_size=batch, policy=policy,
                       prefetch_depth=depth, prefetch_width=width).overhead

    if overhead(1.0) > eps:
        return None                       # unreachable even fully resident
    lo, hi = 0.0, 1.0                     # lo infeasible, hi feasible
    if overhead(0.0) <= eps:
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if overhead(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    gpu = load_gpu("h100")
    mem = load_memory("hbm_cxl_local")
    rows = []
    for name in MODELS:
        m = load_model(name)
        tr = synth_trace(TOK[name], m.n_moe_layers, m.n_experts, m.top_k,
                         zipf_s=1.0, seed=7, drift=0.02, model=m.name)
        hbm_only_gpus = int(np.ceil(
            (m.total_bytes + 12 * GB) / (gpu.hbm_gb * GB)))
        for batch in (128, 512):
            for eps in TARGETS:
                for tag, d, w in (("no prefetch", 0, 0.0),
                                  ("prefetch d=4 w=25%", 4, 0.25)):
                    r = max_offload(m, mem, gpu, tr, batch, eps, depth=d, width=w)
                    if r is None:
                        rows.append({"model": m.name, "batch_size": batch,
                                     "target_overhead": eps, "variant": tag,
                                     "min_resident_frac": None,
                                     "max_offload_frac": None,
                                     "capacity_multiplier": None,
                                     "gpus_hbm_only": hbm_only_gpus,
                                     "gpus_needed": None, "gpu_saving": None})
                        continue
                    hbm_bytes_needed = (r * m.routed_expert_bytes
                                        + m.resident_bytes + 12 * GB)
                    g = max(1, int(np.ceil(hbm_bytes_needed / (gpu.hbm_gb * GB))))
                    rows.append({
                        "model": m.name, "batch_size": batch,
                        "target_overhead": eps, "variant": tag,
                        "min_resident_frac": round(r, 4),
                        "max_offload_frac": round(1 - r, 4),
                        "capacity_multiplier": round(1 / r, 2) if r > 0 else None,
                        "gpus_hbm_only": hbm_only_gpus, "gpus_needed": g,
                        "gpu_saving": round(hbm_only_gpus / g, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "14_frontier.csv", index=False)
    print(df.to_string(index=False))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .3, "axes.spines.top": False,
                         "axes.spines.right": False})
    sub = df[(df.batch_size == 512) & df.max_offload_frac.notna()]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for (mdl, var), g in sub.groupby(["model", "variant"]):
        ax.plot(g.target_overhead * 100, g.max_offload_frac * 100,
                marker="o", ms=3, ls="-" if var == "no prefetch" else "--",
                label=f"{mdl} ({var})")
    ax.set_xscale("log")
    ax.set_xlabel("tolerated stall overhead (% of compute)")
    ax.set_ylabel("% of expert pool that can live in CXL")
    ax.set_title("Feasibility frontier, batch 512, measured-bandwidth CXL")
    ax.legend(fontsize=5.5, ncol=2); fig.tight_layout()
    fig.savefig(FIG / "f9_frontier.png"); plt.close(fig)
    print("\n-> tables/14_frontier.csv, figures/f9_frontier.png")


if __name__ == "__main__":
    main()
