#!/usr/bin/env python3
"""
Figures for the DeepSeek-V2-Lite results.

    uv run python scripts/plot_deepseek.py

Reads results/runtime_deepseek_all.csv and writes two PNGs into
results/figures_runtime/. Every point is a measured run on the A10, a GPU that
cannot hold this model at all: 31.4 GB of weights against 23 GB of memory.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/runtime_deepseek_all.csv")
OUT = Path("results/figures_runtime")

INK, MUTED, RULE = "#14171C", "#5A6472", "#DCE0E6"
HBM, CXL, GOOD, BAD = "#1B4FA8", "#9A4C10", "#16653C", "#A61B2B"


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
            rows.append({k: (float(v) if k not in ("model",) else v)
                         for k, v in r.items()})
    return sorted(rows, key=lambda r: r["batch_tokens"])


def fig_crossover(rows):
    b = [r["batch_tokens"] for r in rows]
    stall = [100 * r["stall_fraction"] for r in rows]
    tps = [r["tokens_per_second"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=170)
    ax.plot(b, stall, marker="v", color=BAD, linewidth=2.4, markersize=8,
            linestyle="--", label="time stalled on transfer")
    ax.set_ylim(-3, 60)
    ax2 = ax.twinx()
    ax2.plot(b, tps, marker="o", color=GOOD, linewidth=2.4, markersize=8,
             label="throughput")
    ax2.set_ylabel("tokens per second", fontsize=10, color=MUTED)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["top"].set_visible(False)

    ax.axvspan(5200, 8600, color="#FDF3D8", zorder=0)
    ax.annotate("crossover", xy=(6700, 52), fontsize=10, color="#6B4D06",
                ha="center")

    ax.set_xscale("log")
    ax.set_xticks(b)
    ax.set_xticklabels([f"{int(x):,}" for x in b], rotation=0)
    style(ax, "tokens per batch", "percent of time stalled",
          "DeepSeek-V2-Lite: all 64 experts per layer held off-GPU")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "rt5_deepseek_crossover.png")
    plt.close(fig)


def fig_link(rows):
    """Achieved link bandwidth peaks exactly at the crossover."""
    b = [r["batch_tokens"] for r in rows]
    gbs = [r["achieved_gbs"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=170)
    ax.plot(b, gbs, marker="s", color=CXL, linewidth=2.4, markersize=8)
    peak = max(range(len(gbs)), key=lambda i: gbs[i])
    ax.annotate(f"link saturated, {gbs[peak]:.1f} GB/s",
                xy=(b[peak], gbs[peak]), xytext=(10, 12),
                textcoords="offset points", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.set_xscale("log")
    ax.set_xticks(b)
    ax.set_xticklabels([f"{int(x):,}" for x in b])
    style(ax, "tokens per batch", "achieved link bandwidth, GB/s",
          "Below the crossover the link is the limit; above it, compute is")
    fig.tight_layout()
    fig.savefig(OUT / "rt6_deepseek_link.png")
    plt.close(fig)


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_crossover(rows)
    fig_link(rows)
    print(f"wrote 2 figures to {OUT}")


if __name__ == "__main__":
    main()
