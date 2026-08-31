#!/usr/bin/env python3
"""MEMoE end-to-end experiment driver.

Produces every table and figure in results/.  Run:  python scripts/run_all.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memoe import *                                       # noqa: E402
from memoe.analysis import GB, MB                         # noqa: E402

RES = ROOT / "results"; TAB = RES / "tables"; FIG = RES / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):          # never leave stale artefacts behind
        f.unlink()

MODELS = ["olmoe", "mixtral_8x7b", "qwen3_235b", "deepseek_v3"]
TRACE_TOKENS = {"olmoe": 65536, "mixtral_8x7b": 65536,
                "qwen3_235b": 32768, "deepseek_v3": 32768}
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
SWEEP_B = [32, 128, 512, 2048, 8192]
RESIDENCY = [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]


def save(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"  -> tables/{name}.csv  ({len(df)} rows)", flush=True)
    return df


def min_gpus(model, gpu, cxl_gb_per_gpu=0.0, reserve_gb=12.0):
    """Fewest GPUs whose HBM (plus per-GPU CXL, if any) holds the checkpoint."""
    for n in range(1, 8193):
        budget = (n * (gpu.hbm_gb + cxl_gb_per_gpu) * GB
                  - model.resident_bytes - reserve_gb * GB)
        if budget >= model.routed_expert_bytes:
            return n
    return -1


def main():
    gpu = load_gpu("h100")
    models = {n: load_model(n) for n in MODELS}
    mem_cxl = load_memory("hbm_cxl_local")
    mem_slow = load_memory("hbm_cxl_local_pessimistic")
    mem_pool = load_memory("hbm_cxl_pooled")

    print("\n[1] model footprints", flush=True)
    save(pd.DataFrame([m.summary() for m in models.values()]), "01_model_footprints")

    print("\n[2] capacity: GPUs needed with and without expansion", flush=True)
    rows = []
    for m in models.values():
        g_hbm = min_gpus(m, gpu)
        for mem in (mem_cxl, mem_pool):
            g_cxl = min_gpus(m, gpu, cxl_gb_per_gpu=mem.cxl.capacity_gb)
            rows.append({
                "model": m.name, "total_GB": round(m.total_bytes / GB, 1),
                "expansion": mem.name, "cxl_GB_per_gpu": mem.cxl.capacity_gb,
                "gpus_hbm_only": g_hbm, "gpus_with_cxl": g_cxl,
                "gpu_reduction": round(g_hbm / g_cxl, 2),
                "capacity_multiplier": round(mem.capacity_multiplier, 2),
                "measured": mem.cxl.measured})
    cap = save(pd.DataFrame(rows), "02_capacity_gain")

    print("\n[3] traces", flush=True)
    traces = {}
    for name, m in models.items():
        traces[name] = synth_trace(TRACE_TOKENS[name], m.n_moe_layers,
                                   m.n_experts, m.top_k, zipf_s=1.0, seed=7,
                                   drift=0.02, model=m.name)
        print(f"  {m.name}: {traces[name].ids.shape}", flush=True)

    print("\n[4] skew", flush=True)
    rows, curves = [], {}
    for name, tr in traces.items():
        st = skew_stats(tr.counts())["aggregate"]
        st["model"] = models[name].name
        rows.append(st)
        curves[models[name].name] = coverage_curve(tr.counts())
    save(pd.DataFrame(rows)[["model", "norm_entropy", "gini", "zipf_s",
                             "top10pct_share", "top25pct_share",
                             "top50pct_share", "max_share"]], "04_skew")

    print("\n[5] batch-size amortisation and expert coverage", flush=True)
    allb = []
    for name, tr in traces.items():
        d = batch_regime(models[name], tr, BATCHES)
        d.insert(0, "model", models[name].name)
        allb.append(d)
    br = save(pd.concat(allb, ignore_index=True), "05_batch_regime")

    print("\n[6] required HBM hit rate", flush=True)
    allh = []
    for name, tr in traces.items():
        d = hitrate_requirement(models[name], mem_cxl, gpu, tr,
                                batch_sizes=(128, 512, 2048, 8192), n_gpus=1)
        d.insert(0, "model", models[name].name)
        allh.append(d)
    save(pd.concat(allh, ignore_index=True), "06_hitrate_requirement")

    print("\n[7] prefetch depth", flush=True)
    allp = []
    for name, tr in traces.items():
        d = prefetch_depth(models[name], mem_cxl, gpu, tr,
                           batch_sizes=(128, 512, 2048, 8192), n_gpus=1)
        d.insert(0, "model", models[name].name)
        allp.append(d)
    save(pd.concat(allp, ignore_index=True), "07_prefetch_depth")

    print("\n[8] MAIN SWEEP: policy x batch size x memory config", flush=True)
    rows = []
    for name, m in models.items():
        tr = traces[name]
        for mem in (mem_cxl, mem_slow, mem_pool):
            n = min_gpus(m, gpu, cxl_gb_per_gpu=mem.cxl.capacity_gb)
            sim = Simulator(m, mem, gpu, n_gpus=n)
            pols = ["static_popularity", "balanced_static"]
            if mem.name == "hbm_cxl_local":
                pols += ["lru", "lfu"]
            for pol in pols:
                for B in SWEEP_B:
                    if B > tr.n_tokens: continue
                    if pol in ("lru", "lfu") and B < 512: continue
                    r = sim.run(tr, batch_size=B, policy=pol)
                    row = r.as_row(); row["gpus"] = n
                    rows.append(row)
            print(f"    {m.name} / {mem.name} done", flush=True)
    sweep = save(pd.DataFrame(rows), "08_main_sweep")

    print("\n[9] THE CROSSOVER: overhead vs batch size at fixed residency", flush=True)
    rows = []
    for name, m in models.items():
        tr = traces[name]
        total = m.n_moe_layers * m.n_experts
        for r_frac in RESIDENCY:
            sim = Simulator(m, mem_cxl, gpu, n_gpus=1,
                            hbm_slots_override=int(round(r_frac * total)))
            for B in BATCHES:
                if B > tr.n_tokens or B < 16: continue
                res = sim.run(tr, batch_size=B, policy="balanced_static")
                row = res.as_row()
                row["target_resident_frac"] = r_frac
                row["critical_batch"] = sim.critical_batch(
                    res.expert_hit_rate, overhead=0.10)
                rows.append(row)
        print(f"    {m.name} done", flush=True)
    cross = save(pd.DataFrame(rows), "09_crossover")

    print("\n[10] closed-form critical batch size", flush=True)
    rows = []
    for m in models.values():
        sim = Simulator(m, mem_cxl, gpu, n_gpus=1)
        for h in (0.0, 0.50, 0.90, 0.95, 0.99, 0.999):
            for eps in (0.10, 0.25):
                rows.append({"model": m.name, "expert_hit_rate": h,
                             "target_overhead": eps,
                             "critical_batch_tokens": round(sim.critical_batch(h, eps))})
    crit = save(pd.DataFrame(rows), "10_critical_batch")

    print("\n[11] prefetch effectiveness", flush=True)
    rows = []
    for name in ("olmoe", "qwen3_235b"):
        m, tr = models[name], traces[name]
        total = m.n_moe_layers * m.n_experts
        sim = Simulator(m, mem_cxl, gpu, n_gpus=1,
                        hbm_slots_override=int(0.90 * total))
        for depth in (0, 1, 2, 4, 8, 16):
            for width in (0.0, 0.05, 0.10, 0.25):
                if (depth == 0) != (width == 0.0): continue
                r = sim.run(tr, batch_size=2048, policy="balanced_static",
                            prefetch_depth=depth, prefetch_width=width)
                rows.append(r.as_row())
        for depth in (1, 2, 4, 8, 16):
            for width in (0.05, 0.10, 0.25):
                r = sim.run(tr, batch_size=2048, policy="balanced_static",
                            prefetch_depth=depth, prefetch_width=width)
                rows.append(r.as_row())
    pf = save(pd.DataFrame(rows).drop_duplicates(
        subset=["model", "prefetch_depth", "prefetch_width"]), "11_prefetch_sweep")

    print("\n[12] sensitivity: routing skew", flush=True)
    rows = []
    m = models["qwen3_235b"]
    total = m.n_moe_layers * m.n_experts
    sim = Simulator(m, mem_cxl, gpu, n_gpus=1,
                    hbm_slots_override=int(0.50 * total))
    for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        tr = synth_trace(16384, m.n_moe_layers, m.n_experts, m.top_k,
                         zipf_s=s, seed=7, drift=0.02, model=m.name)
        for B in (512, 4096):
            r = sim.run(tr, batch_size=B, policy="balanced_static")
            row = r.as_row(); row["zipf_s"] = s
            rows.append(row)
    skewsens = save(pd.DataFrame(rows), "12_skew_sensitivity")

    print("\n[13] sensitivity: bandwidth", flush=True)
    rows = []
    from memoe.memory import Tier, MemorySystem
    for name in ("olmoe", "qwen3_235b"):
        m, tr = models[name], traces[name]
        total = m.n_moe_layers * m.n_experts
        for bw in (18, 26, 36, 52, 80, 128):
            mem = MemorySystem(
                name=f"cxl_{bw}",
                hbm=mem_cxl.hbm,
                cxl=Tier(name=f"cxl@{bw}", capacity_gb=512.0, bandwidth_gbs=bw,
                         latency_ns=300.0, measured=bw <= 52))
            sim = Simulator(m, mem, gpu, n_gpus=1,
                            hbm_slots_override=int(0.95 * total))
            r = sim.run(tr, batch_size=2048, policy="balanced_static")
            row = r.as_row(); row["cxl_bw_gbs"] = bw
            row["extrapolated"] = bw > 52
            rows.append(row)
    bwsens = save(pd.DataFrame(rows), "13_bandwidth_sensitivity")

    make_figures(models, curves, br, sweep, cross, pf, skewsens, bwsens, cap)
    write_report(models, cap, br, sweep, cross, crit, pf, skewsens, bwsens)
    print("\nDONE. See results/REPORT.md", flush=True)


# --------------------------------------------------------------------------
def make_figures(models, curves, br, sweep, cross, pf, skewsens, bwsens, cap):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .3, "axes.spines.top": False,
                         "axes.spines.right": False})
    print("\n[figures]", flush=True)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, (f, c) in curves.items():
        ax.plot(f * 100, c * 100, label=name, lw=1.8)
    ax.axhline(99, ls=":", c="k", lw=1)
    ax.text(2, 99.6, "99% of routings", fontsize=6)
    ax.set_xlabel("% of experts kept in HBM (hottest first)")
    ax.set_ylabel("% of token-routings served from HBM")
    ax.set_title("Expert popularity coverage (ceiling on hit rate)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "f1_coverage.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for name, g in br.groupby("model"):
        axes[0].plot(g.batch_size, g.bytes_per_token_MB, marker="o", ms=3, label=name)
        axes[1].plot(g.batch_size, g.coverage_frac * 100, marker="o", ms=3, label=name)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size (tokens)")
    axes[0].set_ylabel("CXL bytes per token (MB)")
    axes[0].set_title("Batching amortises the fetch...")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("batch size (tokens)")
    axes[1].set_ylabel("% of experts touched per layer per batch")
    axes[1].set_title("...but coverage saturates at 100%")
    axes[1].legend(fontsize=6); fig.tight_layout()
    fig.savefig(FIG / "f2_batch_amortisation.png"); plt.close(fig)

    ms = list(models.values())
    fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3.1), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        sub = cross[cross.model == m.name]
        for r, g in sub.groupby("target_resident_frac"):
            g = g.sort_values("batch_size")
            ax.plot(g.batch_size, np.maximum(g.overhead, 1e-4) * 100,
                    marker="o", ms=2.5, label=f"{r:.0%} in HBM")
        ax.axhline(10, ls="--", c="k", lw=1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(m.name, fontsize=8); ax.set_xlabel("batch size")
    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel("stall overhead (% of compute)"); a0.legend(fontsize=5.5)
    fig.suptitle("The crossover: offload only pays above a critical batch size "
                 "(dashed = 10% overhead)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f3_crossover.png"); plt.close(fig)

    g = sweep[sweep.memory == "hbm_cxl_local"]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        sub = g[g.model == m.name]
        for pol, gg in sub.groupby("policy"):
            gg = gg.sort_values("batch_size")
            ax.plot(gg.batch_size, gg.hit_rate * 100, marker="o", ms=3, label=pol)
        ax.set_xscale("log", base=2); ax.set_title(m.name, fontsize=8)
        ax.set_xlabel("batch size")
    np.atleast_1d(axes)[0].set_ylabel("HBM hit rate (%)")
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.suptitle("Tiering policy: hit rate at the minimum feasible GPU count", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f4_policy.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for (mdl, w), gg in pf.groupby(["model", "prefetch_width"]):
        if w == 0: continue
        gg = gg.sort_values("prefetch_depth")
        ax.plot(gg.prefetch_depth, gg.overhead * 100, marker="o", ms=3,
                label=f"{mdl}, width={w:.0%}")
    ax.set_xlabel("prefetch lookahead (layers)")
    ax.set_ylabel("stall overhead (% of compute)"); ax.set_yscale("log")
    ax.set_title("Speculative prefetch from the popularity prior (90% resident, B=2048)")
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(FIG / "f5_prefetch.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for B, gg in skewsens.groupby("batch_size"):
        ax.plot(gg.zipf_s, gg.hit_rate * 100, marker="o", ms=3, label=f"batch={B}")
    ax.set_xlabel("routing skew (Zipf exponent s)")
    ax.set_ylabel("HBM hit rate (%)")
    ax.set_title("Hit rate vs router skew (Qwen3-235B, 50% resident)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "f6_skew_sensitivity.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, gg in bwsens.groupby("model"):
        meas = gg[~gg.extrapolated]; extr = gg[gg.extrapolated]
        p = ax.plot(meas.cxl_bw_gbs, meas.overhead * 100, marker="o", ms=3, label=name)
        ax.plot(extr.cxl_bw_gbs, extr.overhead * 100, marker="o", ms=3,
                ls=":", c=p[0].get_color())
    ax.axvspan(18, 52, alpha=.12, color="grey")
    ax.text(20, ax.get_ylim()[1] * .5, "measured range", fontsize=6)
    ax.set_xlabel("CXL sustained bandwidth (GB/s)")
    ax.set_ylabel("stall overhead (% of compute)"); ax.set_yscale("log")
    ax.set_title("Bandwidth sensitivity (95% resident, B=2048); dotted = extrapolated")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "f7_bandwidth.png"); plt.close(fig)

    sub = cap[cap.expansion == "hbm_cxl_local"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(sub)); w = 0.38
    ax.bar(x - w/2, sub.gpus_hbm_only, w, label="HBM only")
    ax.bar(x + w/2, sub.gpus_with_cxl, w, label="HBM + 512 GB CXL per GPU")
    for i, (a, b) in enumerate(zip(sub.gpus_hbm_only, sub.gpus_with_cxl)):
        ax.text(i, max(a, b) * 1.03, f"{a/b:.1f}x", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sub.model, rotation=15, fontsize=7)
    ax.set_ylabel("GPUs needed to hold the checkpoint")
    ax.set_title("Capacity ceiling: GPUs required to fit the weights")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "f8_gpu_reduction.png"); plt.close(fig)
    print("  -> 8 figures in results/figures/", flush=True)


def write_report(models, cap, br, sweep, cross, crit, pf, skewsens, bwsens):
    A = [].append
    L = []; A = L.append
    A("# MEMoE results\n")
    A("_Generated by `scripts/run_all.py`. CXL numbers use **measured** "
      "sustained bandwidth (18-52 GB/s), never link peak. Rows marked "
      "modeled/extrapolated are projections, not measurements._\n")

    A("\n## Headline\n")
    A("Expert offload to CXL is **not** a general-purpose win. It is viable in "
      "a specific regime, and we can state the boundary of that regime in "
      "closed form:\n")
    A("```\nB* = (1 - h) * E * F / (BW_cxl * k * eps)\n```\n")
    A("the batch size above which the expert fetch is hidden behind compute. "
      "`h` is the fraction of expert *fetches* avoided by the HBM hot tier, "
      "`E` experts per layer, `k` activated per token, `F` per-GPU effective "
      "FLOP/s, `eps` the tolerated stall. B* does not depend on expert size, "
      "and it does not depend on how many GPUs you deploy, because compute "
      "and per-node CXL bandwidth scale together. It depends almost entirely "
      "on the hit rate.\n")

    A("\n## 1. The capacity problem\n")
    sub = cap[cap.expansion == "hbm_cxl_local"]
    A(sub[["model", "total_GB", "gpus_hbm_only", "gpus_with_cxl",
           "gpu_reduction", "capacity_multiplier"]].to_markdown(index=False))
    A(f"\nRouted experts are "
      f"{min(m.expert_fraction for m in models.values()):.1%}-"
      f"{max(m.expert_fraction for m in models.values()):.1%} of every "
      "checkpoint here, and any one token touches a small fraction of them. "
      "That is the part CXL can hold.\n")

    A("\n## 2. Batching amortises the fetch, but coverage saturates\n")
    piv = br.pivot(index="batch_size", columns="model", values="bytes_per_token_MB")
    A(piv.round(2).to_markdown())
    A("\nPer-token traffic falls roughly as 1/B. But the reason it stops "
      "falling is the important part:\n")
    piv2 = br.pivot(index="batch_size", columns="model", values="coverage_frac")
    A(piv2.round(3).to_markdown())
    A("\nOnce a batch touches every expert at every layer, the HBM tier stops "
      "behaving like a cache and becomes a fixed partition: every cold expert "
      "is refetched on **every** batch. Temporal locality is gone; only the "
      "popularity split matters.\n")

    A("\n## 3. The crossover\n")
    piv3 = cross[cross.target_resident_frac.isin([0.90, 0.99])].pivot_table(
        index="batch_size", columns=["model", "target_resident_frac"],
        values="overhead")
    A(piv3.round(3).to_markdown())
    A("\nOverhead falls as 1/B at fixed residency. Crossing below 10% needs "
      "either a very large batch or a very high hit rate.\n")

    A("\n## 4. Closed-form critical batch size (eps = 10%)\n")
    c = crit[crit.target_overhead == 0.10].pivot(
        index="expert_hit_rate", columns="model", values="critical_batch_tokens")
    A(c.to_markdown())
    A("\nRead this as the design requirement. A hot tier that avoids 99% of "
      "expert fetches puts OLMoE and Mixtral inside the range of a large "
      "prefill batch; DeepSeek-V3 needs 99.9%. Decode-sized batches are "
      "never in range, for any model here.\n")

    A("\n## 5. Tiering policy\n")
    g = sweep[sweep.memory == "hbm_cxl_local"]
    best = g.loc[g.groupby(["model", "batch_size"]).overhead.idxmin()]
    A(best[["model", "batch_size", "policy", "gpus", "hit_rate",
            "expert_hit_rate", "overhead"]].round(4).to_markdown(index=False))
    A("\nStatic popularity placement is competitive with LRU and sampled LFU, "
      "and at large batch it wins outright: with full coverage there is no "
      "recency signal left for a dynamic policy to exploit. That is a useful "
      "result for CXL specifically, because static placement is decided once "
      "at load time and costs nothing on the critical path.\n")

    A("\n## 6. Prefetching\n")
    A(pf[pf.prefetch_width > 0][["model", "prefetch_depth", "prefetch_width",
        "prefetch_coverage", "overhead"]].round(4).to_markdown(index=False))
    A("\nPrefetch helps only to the extent the popularity prior predicts the "
      "miss, and only if the transfer fits in the lookahead window. "
      "Deeper lookahead widens the window linearly.\n")

    A("\n## 7. Sensitivity to router skew\n")
    A(skewsens[["zipf_s", "batch_size", "hit_rate", "expert_hit_rate",
                "overhead"]].round(4).to_markdown(index=False))
    A("\nAt s=0 (uniform routing) a 50% resident tier serves 50% of routings "
      "and nothing more: tiering buys exactly its capacity fraction. Every "
      "result above rests on real routers being skewed. **Measuring that on a "
      "real checkpoint is the highest-value remaining experiment**, and it is "
      "what `memoe/hooks.py` exists for.\n")

    A("\n## 8. Sensitivity to CXL bandwidth\n")
    A(bwsens[["model", "cxl_bw_gbs", "extrapolated", "overhead"]]
      .round(4).to_markdown(index=False))
    A("\nOverhead is inversely proportional to bandwidth, with no threshold "
      "effect. Within the measured 18-52 GB/s band the qualitative picture "
      "does not change.\n")

    A("\n## Threats to validity\n")
    A("- Routing traces are **synthetic** (Zipf s=1.0, mild drift). Real "
      "router skew varies by layer and by input domain, and real routers are "
      "load-balanced during training, which pushes *against* skew. This is "
      "the weakest assumption in the project.\n"
      "- CXL 3.0 pooling figures are **modeled**; no shipping switch silicon "
      "exists to measure. Direct-attached CXL 2.0 figures are within the "
      "published measured range.\n"
      "- The compute model counts routed-expert GEMMs only, at 40% MFU. "
      "Attention and dense layers add compute that would hide more transfer, "
      "so reported overheads are conservative.\n"
      "- Prefetch uses a static popularity prior. A predictor conditioned on "
      "layer L-1 routing should do better; we have not built one.\n"
      "- We model bandwidth and capacity, not contention: concurrent KV-cache "
      "traffic over the same link is not simulated.\n")
    (RES / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("  -> results/REPORT.md", flush=True)


if __name__ == "__main__":
    main()
