#!/usr/bin/env python3
"""
Figures for the MEMoE-RT results section.

    uv run python scripts/plot_runtime.py

Reads results/runtime_bench_all.csv and writes four PNGs into
results/figures_runtime/. Every number comes from a measured run on the
Blackwell box; nothing here is modelled.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/runtime_bench_all.csv")
OUT = Path("results/figures_runtime")

INK = "#14171C"
MUTED = "#5A6472"
RULE = "#DCE0E6"
HBM = "#1B4FA8"
CXL = "#9A4C10"
GOOD = "#16653C"
BAD = "#A61B2B"

PLATEAU_GBS = 40.8       # measured PCIe 5.0 x16 plateau
BASE_WALL_4096 = 0.5875  # fully resident wall time at batch 4096


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.grid(axis="y", color="#EDEFF2", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)


def load():
    rows = []
    with SRC.open() as fh:
        for r in csv.DictReader(fh):
            for k in ("residency", "offloaded_fraction", "peak_vram_gb",
                      "wall_seconds", "tokens_per_second", "gb_streamed",
                      "stall_fraction", "achieved_gbs"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            for k in ("batch_tokens", "depth", "experts_offloaded"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def fig_residency(rows):
    d = sorted([r for r in rows if r["sweep"] == "residency"],
               key=lambda r: r["offloaded_fraction"])
    base = next(r for r in d if r["residency"] == 1.0)["tokens_per_second"]
    x = [100 * r["offloaded_fraction"] for r in d]
    y = [100 * r["tokens_per_second"] / base for r in d]
    v = [r["peak_vram_gb"] for r in d]

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    ax.plot(x, y, marker="o", color=CXL, linewidth=2.2, markersize=7,
            label="throughput retained")
    ax2 = ax.twinx()
    ax2.plot(x, v, marker="s", color=HBM, linewidth=1.8, markersize=6,
             linestyle="--", label="GPU memory used")
    ax2.set_ylabel("GPU memory, GB", fontsize=10, color=MUTED)
    ax2.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)

    ax.annotate("100% of experts off-GPU\n55% throughput, 4.4 GB",
                xy=(100, y[-1]), xytext=(-150, 26), textcoords="offset points",
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    style(ax, "share of experts held off-GPU, percent",
          "throughput as percent of fully resident",
          "Cost of offloading, batch 4,096")
    ax.set_ylim(40, 105)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "r1_residency.png")
    plt.close(fig)


def fig_batch(rows):
    off = sorted([r for r in rows if r["sweep"] == "batch" and r["residency"] == 0.3],
                 key=lambda r: r["batch_tokens"])
    bas = {r["batch_tokens"]: r["tokens_per_second"]
           for r in rows if r["sweep"] == "batch" and r["residency"] == 1.0}
    x = [r["batch_tokens"] for r in off]
    ret = [100 * r["tokens_per_second"] / bas[r["batch_tokens"]] for r in off]
    stall = [100 * r["stall_fraction"] for r in off]

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    ax.plot(x, ret, marker="o", color=GOOD, linewidth=2.2, markersize=7,
            label="throughput retained")
    ax.plot(x, stall, marker="v", color=BAD, linewidth=2.2, markersize=7,
            linestyle="--", label="time stalled on transfer")
    ax.axvspan(256, 1500, color="#FBEEEF", zorder=0)
    ax.axvspan(3500, 20000, color="#EAF3EE", zorder=0)
    ax.text(700, 84, "decode\nbatch sizes", fontsize=9, color=BAD, ha="center")
    ax.text(9000, 84, "prefill\nbatch sizes", fontsize=9, color=GOOD, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:,}" for b in x])
    style(ax, "tokens per batch", "percent",
          "The regime boundary, 70% of experts off-GPU throughout")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "r2_batch_regime.png")
    plt.close(fig)


def fig_depth(rows):
    d = sorted([r for r in rows if r["sweep"] == "depth" and r["depth"] > 0],
               key=lambda r: r["depth"])
    x = [str(r["depth"]) for r in d]
    y = [r["tokens_per_second"] for r in d]
    v = [r["peak_vram_gb"] for r in d]

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=170)
    ax.bar(x, y, color=CXL, width=0.55, label="throughput")
    ax.set_ylim(0, max(y) * 1.25)
    for i, (yy, vv) in enumerate(zip(y, v)):
        ax.text(i, yy + max(y) * 0.03, f"{yy:,.0f}", ha="center",
                fontsize=9, color=INK)
        ax.text(i, max(y) * 1.13, f"{vv:.1f} GB", ha="center",
                fontsize=9, color=HBM)
    ax.text(-0.45, max(y) * 1.20, "staging memory", fontsize=9, color=HBM)
    style(ax, "prefetch lookahead, layers", "tokens per second",
          "Deeper lookahead buys nothing once transfers are batched per layer")
    fig.tight_layout()
    fig.savefig(OUT / "r3_depth.png")
    plt.close(fig)


def fig_overlap(rows):
    """Measured wall time against the ideal-overlap model and the fitted one."""
    d = sorted([r for r in rows if r["sweep"] == "residency"],
               key=lambda r: r["offloaded_fraction"])
    x = [100 * r["offloaded_fraction"] for r in d]
    meas = [r["wall_seconds"] for r in d]
    xfer = [(r["gb_streamed"] / PLATEAU_GBS) for r in d]
    ideal = [max(BASE_WALL_4096, t) for t in xfer]
    exposed = [(m - BASE_WALL_4096) / t if t else 0.0
               for m, t in zip(meas, xfer)]
    k = sum(e for e in exposed[1:]) / len(exposed[1:])
    fitted = [BASE_WALL_4096 + k * t for t in xfer]

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    ax.plot(x, meas, marker="o", color=INK, linewidth=2.2, markersize=7,
            label="measured")
    ax.plot(x, ideal, color=HBM, linewidth=1.8, linestyle="--",
            label="model assuming perfect overlap")
    ax.plot(x, fitted, color=CXL, linewidth=1.8, linestyle=":",
            label=f"model with exposure {k:.2f}")
    style(ax, "share of experts held off-GPU, percent", "wall time, seconds",
          "Calibrating the overlap assumption")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "r4_overlap.png")
    plt.close(fig)
    print(f"fitted exposure constant: {k:.3f}")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_residency(rows)
    fig_batch(rows)
    fig_depth(rows)
    fig_overlap(rows)
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
