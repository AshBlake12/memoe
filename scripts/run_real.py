#!/usr/bin/env python3
"""MEMoE analysis against REAL captured routing traces.

Consumes results/traces/olmoe_{prose,math,code,chat}.npz produced by
capture_traces.py, and the DRAMSim3-calibrated memory config. Everything here
is measured: routing from OLMoE-1B-7B, bandwidth and latency from DRAMSim3.

Run from the repo root:   python scripts/run_real.py
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memoe import *
from memoe.trace import RoutingTrace
from memoe.analysis import GB

RES = ROOT / "results"; TAB = RES / "tables_real"; FIG = RES / "figures_real"
for d in (TAB, FIG): d.mkdir(parents=True, exist_ok=True)

DOMAINS = ["prose", "math", "code", "chat"]
BATCHES = [16, 64, 256, 1024, 4096, 16384]
RESID = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
MEMCFG = "cxl_measured"


def save(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"  -> tables_real/{name}.csv ({len(df)} rows)", flush=True)
    return df


def load_traces():
    tr = {}
    for d in DOMAINS:
        p = RES / "traces" / f"olmoe_{d}.npz"
        if not p.exists():
            print(f"  MISSING {p} -- run capture_traces.py first"); sys.exit(1)
        tr[d] = RoutingTrace.load(p)
        print(f"  {d:6s} {tr[d].n_tokens:>7,} tokens  source={tr[d].source}")
    return tr


def topk_set(counts, layer, frac):
    k = max(1, int(frac * counts.shape[1]))
    return set(np.argsort(counts[layer])[::-1][:k].tolist())


def overlap(cA, cB, frac):
    k = max(1, int(frac * cA.shape[1]))
    return float(np.mean([len(topk_set(cA, l, frac) & topk_set(cB, l, frac)) / k
                          for l in range(cA.shape[0])]))


def main():
    print("\n[0] traces")
    tr = load_traces()
    n_min = min(t.n_tokens for t in tr.values())
    print(f"  equalising to {n_min:,} tokens for cross-domain comparisons")
    eq = {d: RoutingTrace(ids=t.ids[:n_min], model=t.model,
                          n_experts=t.n_experts, source=t.source)
          for d, t in tr.items()}
    cnt = {d: t.counts() for d, t in eq.items()}
    L, E = cnt["prose"].shape

    m = load_model("olmoe"); gpu = load_gpu("h100")
    mem = load_memory(MEMCFG)
    print(f"  memory: {mem.name}, CXL {mem.cxl.bandwidth_gbs} GB/s @ "
          f"{mem.cxl.latency_ns} ns, measured={mem.cxl.measured}")
    total = m.n_moe_layers * m.n_experts

    print("\n[1] per-domain skew")
    rows = []
    for d in DOMAINS:
        st = skew_stats(cnt[d])["aggregate"]
        f, cov = coverage_curve(cnt[d])
        rows.append({"domain": d, "tokens_full": tr[d].n_tokens,
                     "zipf_s": round(st["zipf_s"], 3),
                     "norm_entropy": round(st["norm_entropy"], 3),
                     "gini": round(st["gini"], 3),
                     "max_share": round(st["max_share"], 4),
                     "dead_experts": st["dead_experts"],
                     "top10": round(cov[10], 3), "top25": round(cov[25], 3),
                     "top50": round(cov[50], 3)})
    skew_df = save(pd.DataFrame(rows), "r01_skew_by_domain")

    print("\n[2] hot-set overlap, cross-domain and within-domain control")
    rows = []
    for frac in (0.25, 0.50):
        # within-domain control: split each trace in half
        for d in DOMAINS:
            h = n_min // 2
            a = RoutingTrace(ids=eq[d].ids[:h], model=d, n_experts=E).counts()
            b = RoutingTrace(ids=eq[d].ids[h:], model=d, n_experts=E).counts()
            rows.append({"frac": frac, "a": d + "[1st half]", "b": d + "[2nd half]",
                         "overlap": round(overlap(a, b, frac), 4),
                         "chance": frac, "kind": "within-domain control"})
        for a, b in itertools.combinations(DOMAINS, 2):
            rows.append({"frac": frac, "a": a, "b": b,
                         "overlap": round(overlap(cnt[a], cnt[b], frac), 4),
                         "chance": frac, "kind": "cross-domain"})
    ov_df = save(pd.DataFrame(rows), "r02_hotset_overlap")
    ctl = ov_df[(ov_df.kind == "within-domain control") & (ov_df.frac == 0.25)]
    print(f"  within-domain control mean = {ctl.overlap.mean():.1%} "
          f"(chance 25%) -- must be high for the cross-domain result to mean anything")

    print("\n[3] cross-workload placement penalty")
    rows = []
    for train in DOMAINS:
        for test in DOMAINS:
            for r in (0.25, 0.50, 0.75):
                sim = Simulator(m, mem, gpu, n_gpus=1,
                                hbm_slots_override=int(r * total))
                res = sim.run(eq[test], batch_size=1024,
                              policy="balanced_static", profile=cnt[train])
                rows.append({"profiled_on": train, "deployed_on": test,
                             "resident_frac": r,
                             "routing_hit": round(res.hit_rate, 4),
                             "fetch_hit": round(res.expert_hit_rate, 4),
                             "overhead": round(res.overhead, 3)})
    pen_df = save(pd.DataFrame(rows), "r03_placement_penalty")
    p = pen_df[pen_df.resident_frac == 0.5]
    diag = p[p.profiled_on == p.deployed_on].routing_hit.mean()
    off = p[p.profiled_on != p.deployed_on].routing_hit.mean()
    print(f"  matched {diag:.3f} vs mismatched {off:.3f} "
          f"(random placement would give ~0.500)")

    print("\n[4] crossover on real traces")
    rows = []
    for d in DOMAINS:
        for r in RESID:
            sim = Simulator(m, mem, gpu, n_gpus=1,
                            hbm_slots_override=int(r * total))
            for B in BATCHES:
                if B > eq[d].n_tokens: continue
                res = sim.run(eq[d], batch_size=B, policy="balanced_static")
                rows.append({"domain": d, "resident_frac": r, "batch_size": B,
                             "routing_hit": round(res.hit_rate, 4),
                             "fetch_hit": round(res.expert_hit_rate, 4),
                             "coverage": round(eq[d].distinct_per_batch(B) / E, 3),
                             "overhead": round(res.overhead, 4)})
    cross = save(pd.DataFrame(rows), "r04_crossover")

    print("\n[5] offloadable fraction at 10% overhead")
    rows = []
    for d in DOMAINS:
        for B in BATCHES:
            if B > eq[d].n_tokens: continue
            lo, hi = 0.0, 1.0
            for _ in range(12):
                mid = (lo + hi) / 2
                sim = Simulator(m, mem, gpu, n_gpus=1,
                                hbm_slots_override=int(mid * total))
                if sim.run(eq[d], batch_size=B,
                           policy="balanced_static").overhead <= 0.10:
                    hi = mid
                else:
                    lo = mid
            rows.append({"domain": d, "batch_size": B,
                         "min_resident_frac": round(hi, 4),
                         "offloadable_frac": round(1 - hi, 4)})
    off_df = save(pd.DataFrame(rows), "r05_offloadable")

    figures(cnt, skew_df, ov_df, pen_df, cross, off_df)
    print("\nDONE. tables in results/tables_real/, figures in results/figures_real/")


def figures(cnt, skew_df, ov_df, pen_df, cross, off_df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .3, "axes.spines.top": False,
                         "axes.spines.right": False})
    print("\n[figures]")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for d in DOMAINS:
        f, cov = coverage_curve(cnt[d])
        ax.plot(f * 100, cov * 100, lw=1.8,
                label=f"{d} (s={skew_df[skew_df.domain==d].zipf_s.iloc[0]})")
    ax.plot([0, 100], [0, 100], "k:", lw=1, label="uniform routing")
    ax.set_xlabel("% of experts kept in HBM (hottest first)")
    ax.set_ylabel("% of token-routings served from HBM")
    ax.set_title("Measured expert popularity, OLMoE-1B-7B")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "r1_coverage.png"); plt.close(fig)

    sub = ov_df[(ov_df.frac == 0.25) & (ov_df.kind == "cross-domain")]
    M = np.full((4, 4), np.nan)
    for _, r in sub.iterrows():
        i, j = DOMAINS.index(r.a), DOMAINS.index(r.b)
        M[i, j] = M[j, i] = r.overlap * 100
    np.fill_diagonal(M, 100)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(M, cmap="RdBu", vmin=0, vmax=50)
    ax.set_xticks(range(4)); ax.set_xticklabels(DOMAINS)
    ax.set_yticks(range(4)); ax.set_yticklabels(DOMAINS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title("Hot-set overlap, top 25% (chance = 25)")
    fig.colorbar(im, label="%"); fig.tight_layout()
    fig.savefig(FIG / "r2_overlap.png"); plt.close(fig)

    p = pen_df[pen_df.resident_frac == 0.5]
    M2 = np.zeros((4, 4))
    for _, r in p.iterrows():
        M2[DOMAINS.index(r.profiled_on), DOMAINS.index(r.deployed_on)] = r.routing_hit * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(M2, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(4)); ax.set_xticklabels(DOMAINS)
    ax.set_yticks(range(4)); ax.set_yticklabels(DOMAINS)
    ax.set_xlabel("deployed on"); ax.set_ylabel("profiled on")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M2[i,j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title("Routing hit rate, 50% resident (random = 50)")
    fig.colorbar(im, label="%"); fig.tight_layout()
    fig.savefig(FIG / "r3_penalty.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for d in DOMAINS:
        g = off_df[off_df.domain == d].sort_values("batch_size")
        ax.plot(g.batch_size, g.offloadable_frac * 100, marker="o", ms=4, label=d)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size (tokens)")
    ax.set_ylabel("% of expert pool that can live on CXL")
    ax.set_title("Offloadable share at 10% overhead, measured CXL @ 16.88 GB/s")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "r4_offloadable.png"); plt.close(fig)
    print("  -> 4 figures in results/figures_real/")


if __name__ == "__main__":
    main()
