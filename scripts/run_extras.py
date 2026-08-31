#!/usr/bin/env python3
"""Three studies the MoE analysis does not cover.

  1. KV cache vs expert weights as CXL targets (the Transformer half).
  2. Memory pooling across nodes.
  3. Latency sensitivity and transfer granularity.

Run from the repo root:  python scripts/run_extras.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memoe import *
from memoe.analysis import GB, MB

RES = ROOT / "results"; TAB = RES / "tables_extra"; FIG = RES / "figures_extra"
for d in (TAB, FIG): d.mkdir(parents=True, exist_ok=True)
MODELS = ["olmoe", "mixtral_8x7b", "qwen3_235b", "deepseek_v3"]


def save(df, n):
    df.to_csv(TAB / f"{n}.csv", index=False)
    print(f"  -> tables_extra/{n}.csv ({len(df)} rows)", flush=True)
    return df


# ---------------------------------------------------------------- 1. KV cache
def kv_study(models, gpu, mem):
    """Arithmetic intensity decides what belongs on a slow tier.

    A memory tier with bandwidth BW feeding a device with F FLOP/s is
    balanced at an intensity of F/BW FLOPs per byte. Anything below that
    ratio is bandwidth-bound on that tier.

    Expert weights:  each fetched expert serves B*k/E tokens, so intensity
                     is (6 d f B k/E) / (6 d f) = B*k/E -- it GROWS with
                     the batch. That is why batching rescues expert offload.

    KV cache in decode: each cached element is read once per step and used
                     by g = n_heads/n_kv_heads query heads, giving an
                     intensity of about g FLOPs per byte, FIXED. Batching
                     does not help, because every sequence has its own KV.
    """
    rows = []
    for m in models:
        g = (m.n_heads / m.n_kv_heads) if m.n_kv_heads else 1.0
        kv_tok = m.kv_bytes_per_token()
        for name, bw in [("HBM3e", mem.hbm.bandwidth_gbs),
                         ("CXL 1ch", 16.88), ("CXL 4ch", 67.5)]:
            balance = gpu.eff_flops / (bw * 1e9)
            rows.append({
                "model": m.name, "tier": name, "bw_gbs": bw,
                "balance_flops_per_byte": round(balance),
                "kv_intensity": round(g, 1),
                "kv_bandwidth_bound": bool(g < balance),
                "kv_slowdown_vs_hbm": round(mem.hbm.bandwidth_gbs / bw, 1),
                "kv_KB_per_token": round(kv_tok / 1024, 1),
                "batch_for_expert_parity": int(np.ceil(balance * m.n_experts
                                                       / (m.top_k + m.n_shared_experts))),
            })
    df = save(pd.DataFrame(rows), "e01_kv_vs_experts")

    # what a KV offload actually costs during decode
    rows = []
    for m in models:
        g = (m.n_heads / m.n_kv_heads) if m.n_kv_heads else 1.0
        for ctx in (4096, 32768, 131072):
            kv = m.kv_bytes_per_token() * ctx
            for name, bw in [("HBM3e", mem.hbm.bandwidth_gbs),
                             ("CXL 1ch", 16.88), ("CXL 4ch", 67.5)]:
                t_read = kv / (bw * 1e9)
                t_flops = 2 * g * kv / gpu.eff_flops
                rows.append({"model": m.name, "context": ctx, "tier": name,
                             "kv_GB_per_seq": round(kv / GB, 3),
                             "read_ms_per_step": round(t_read * 1e3, 3),
                             "compute_ms_per_step": round(t_flops * 1e3, 4),
                             "bound": "memory" if t_read > t_flops else "compute",
                             "steps_per_s": round(1 / max(t_read, t_flops))})
    save(pd.DataFrame(rows), "e02_kv_decode_cost")
    return df


# ------------------------------------------------------------------ 2. pooling
def pooling_study(models, gpu):
    """N nodes behind one switched pool.

    Capacity: nodes running the same checkpoint share ONE copy of the cold
    experts instead of N, so effective capacity per node scales with N.
    Bandwidth: those N nodes contend for the pool's link, so per-node
    bandwidth falls as 1/N.

    Since B* is inversely proportional to bandwidth, pooling multiplies both
    the capacity benefit and the critical batch size by N. It buys capacity
    with batch size, at a fixed exchange rate.
    """
    rows = []
    POOL_BW, POOL_GB = 67.5, 2048.0
    for m in models:
        mem = load_memory("cxl_measured")
        for N in (1, 2, 4, 8, 16):
            per_node_bw = POOL_BW / N
            mem.cxl.bandwidth_gbs = per_node_bw
            sim = Simulator(m, mem, gpu, n_gpus=1)
            b_local = Simulator(m, load_memory("cxl_measured"), gpu,
                                n_gpus=1).critical_batch(0.90, 0.10)
            rows.append({
                "model": m.name, "nodes": N,
                "pool_GB": POOL_GB, "per_node_bw_gbs": round(per_node_bw, 2),
                "dedicated_GB_needed": round(N * m.routed_expert_bytes / GB, 1),
                "pooled_GB_needed": round(m.routed_expert_bytes / GB, 1),
                "capacity_saving": N,
                "critical_batch_h90": int(sim.critical_batch(0.90, 0.10)),
                "vs_single_node": round(sim.critical_batch(0.90, 0.10)
                                        / max(1, b_local), 2),
                "pool_fits_model": bool(POOL_GB * GB >= m.routed_expert_bytes),
            })
    return save(pd.DataFrame(rows), "e03_pooling")


# ---------------------------------------------------- 3. latency & granularity
def latency_study(models, gpu):
    """Latency is irrelevant at expert granularity and dominant below it.

    One 12 MB expert at 67.5 GB/s takes 178 us to move; a 300 ns link
    latency is 0.17% of that. But if a placement engine fetched experts in
    small pieces, per-transfer latency would be paid on every piece.
    """
    rows = []
    for m in models:
        for lat in (200, 300, 400, 600, 800):
            for bw in (16.88, 67.5):
                for gran_kb in (4, 64, 512, 4096, None):   # None = whole expert
                    size = m.expert_bytes if gran_kb is None else gran_kb * 1024
                    n = m.expert_bytes / size
                    t_bw = m.expert_bytes / (bw * 1e9)
                    t_lat = n * lat * 1e-9
                    rows.append({
                        "model": m.name, "latency_ns": lat, "bw_gbs": bw,
                        "granularity": "whole expert" if gran_kb is None
                                       else f"{gran_kb} KB",
                        "transfers": int(n),
                        "transfer_ms": round(t_bw * 1e3, 4),
                        "latency_ms": round(t_lat * 1e3, 4),
                        "latency_share": round(t_lat / (t_bw + t_lat), 4)})
    return save(pd.DataFrame(rows), "e04_latency_granularity")


def figures(kv, pool, lat, models, gpu, mem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .3, "axes.spines.top": False,
                         "axes.spines.right": False})
    print("\n[figures]")

    # roofline: intensity vs batch, experts rise, KV is flat
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    B = np.logspace(0, 6, 60)
    for m in models[:3]:
        ax.plot(B, B * (m.top_k + m.n_shared_experts) / m.n_experts,
                label=f"{m.name} experts")
        g = (m.n_heads / m.n_kv_heads) if m.n_kv_heads else 1.0
        ax.axhline(g, ls="--", lw=1, alpha=.6)
    for name, bw, c in [("HBM3e", mem.hbm.bandwidth_gbs, "#888"),
                        ("CXL 4ch", 67.5, "#d62728"),
                        ("CXL 1ch", 16.88, "#ff7f0e")]:
        ax.axhline(gpu.eff_flops / (bw * 1e9), color=c, ls=":", lw=1.4)
        ax.text(1.3, gpu.eff_flops / (bw * 1e9) * 1.15, f"{name} balance",
                fontsize=6, color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("batch size (tokens)")
    ax.set_ylabel("arithmetic intensity (FLOP / byte)")
    ax.set_title("Why experts offload and KV does not\n"
                 "(dashed = KV intensity, fixed at the GQA group size)",
                 fontsize=10)
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(FIG / "e1_roofline.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, g in pool.groupby("model"):
        ax.plot(g.nodes, g.critical_batch_h90, marker="o", ms=4, label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("nodes sharing the pool")
    ax.set_ylabel("critical batch B* (tokens, h=0.9)")
    ax.set_title("Pooling buys capacity with batch size, 1:1")
    ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(FIG / "e2_pooling.png"); plt.close(fig)

    sub = lat[(lat.model == "OLMoE-1B-7B") & (lat.bw_gbs == 67.5)]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for gran, g in sub.groupby("granularity"):
        g = g.sort_values("latency_ns")
        ax.plot(g.latency_ns, g.latency_share * 100, marker="o", ms=4, label=gran)
    ax.set_xlabel("CXL access latency (ns)")
    ax.set_ylabel("% of transfer time spent on latency")
    ax.set_yscale("log")
    ax.set_title("Latency only matters if you fetch in small pieces")
    ax.legend(fontsize=6, title="granularity", title_fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "e3_latency.png"); plt.close(fig)
    print("  -> 3 figures in results/figures_extra/")


def main():
    gpu = load_gpu("h100"); mem = load_memory("cxl_measured")
    models = [load_model(n) for n in MODELS]
    print("\n[1] KV cache vs expert weights")
    kv = kv_study(models, gpu, mem)
    print("\n[2] memory pooling")
    pool = pooling_study(models, gpu)
    print("\n[3] latency and transfer granularity")
    lat = latency_study(models, gpu)
    figures(kv, pool, lat, models, gpu, mem)
    print("\nDONE.")


if __name__ == "__main__":
    main()
