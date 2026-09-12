#!/usr/bin/env python3
"""
Figures for the serving-loop section.

    uv run python scripts/plot_serve.py

Reads results/serve_r{4,8,16,32}.csv and results/serve_bench.csv and writes
two PNGs into results/figures_serve/. Both are referenced by paper/memoe.tex
as s1_decode_tpot.png and s2_phase_cost.png. Every number is measured.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/figures_serve")
CONCURRENCY = [4, 8, 16, 32]

INK = "#14171C"
MUTED = "#5A6472"
RULE = "#DCE0E6"
HBM = "#1B4FA8"
CXL = "#9A4C10"


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


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as fh:
        return {r["config"]: r for r in csv.DictReader(fh)}


def fig_tpot():
    res, off = [], []
    for c in CONCURRENCY:
        rows = read(Path(f"results/serve_r{c}.csv"))
        res.append(float(rows["resident"]["tpot_mean_ms"]))
        off.append(float(rows["offload"]["tpot_mean_ms"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(CONCURRENCY, off, marker="o", color=CXL, linewidth=2,
            label="experts off-GPU")
    ax.plot(CONCURRENCY, res, marker="o", color=HBM, linewidth=2,
            label="fully resident")
    for x, y in zip(CONCURRENCY, off):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=CXL)
    for x, y in zip(CONCURRENCY, res):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=HBM)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY)
    ax.set_xticklabels([str(c) for c in CONCURRENCY])
    ax.set_ylim(0, max(off) * 1.18)
    style(ax, "concurrent requests", "time per output token (ms)",
          "Decode cost does not amortise")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "s1_decode_tpot.png", dpi=200)
    plt.close(fig)


def fig_phase():
    rows = read(Path("results/serve_bench.csv"))
    r, o = rows["resident"], rows["offload"]
    phases = ["prefill", "decode"]
    base = [float(r["prefill_seconds"]), float(r["decode_seconds"])]
    tier = [float(o["prefill_seconds"]), float(o["decode_seconds"])]
    over = [(t / b - 1) * 100 for t, b in zip(tier, base)]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars = ax.bar(phases, over, width=0.45, color=[HBM, CXL])
    for bar, pct, b, t in zip(bars, over, base, tier):
        ax.annotate(f"+{pct:.0f}%\n{b:.2f}s to {t:.2f}s",
                    (bar.get_x() + bar.get_width() / 2, pct),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, max(over) * 1.25)
    style(ax, "", "added time under offload (%)",
          "Same configuration, two phases")
    fig.tight_layout()
    fig.savefig(OUT / "s2_phase_cost.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_tpot()
    fig_phase()
    print(f"-> {OUT}/s1_decode_tpot.png, {OUT}/s2_phase_cost.png")


if __name__ == "__main__":
    main()
