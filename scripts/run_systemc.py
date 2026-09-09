#!/usr/bin/env python3
"""drives the systemc model and writes its table and figure.

systemc only lets you call sc_start once per process, so every operating point
is a separate run of the binary instead of a loop. slow, but it's a few seconds
either way.

    python scripts/run_systemc.py
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "systemc" / "memoe_tlm.cpp"
BIN = ROOT / "systemc" / "memoe_tlm"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures_extra"

COLS = ["batch", "residency", "depth", "tier_bw_gbs", "wall_s", "compute_s",
        "stall_s", "stall_fraction", "tokens_per_second", "staging_gb",
        "b_star"]


def build() -> bool:
    if BIN.exists() and BIN.stat().st_mtime > SRC.stat().st_mtime:
        return True
    print("building the SystemC model ...")
    r = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-o", str(BIN), str(SRC), "-lsystemc"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("build failed. SystemC headers not found?\n"
              "  Debian/Ubuntu:  apt-get install libsystemc-dev\n"
              "  from source:    https://github.com/accellera-official/systemc\n")
        print(r.stderr[:800])
        return False
    return True


def point(**kw) -> dict:
    args = [str(BIN)]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    out = subprocess.run(args, capture_output=True, text=True).stdout.strip()
    line = out.splitlines()[-1]
    vals = line.split(",")
    row = dict(zip(COLS, vals))
    return {k: (float(v) if "." in v or "e" in v.lower() else int(v))
            for k, v in row.items()}


def main() -> int:
    if not build():
        return 1

    rows = []

    # the crossover. this is the one that matters
    for b in (256, 512, 1024, 2048, 4096, 4626, 8192, 16384, 32768):
        rows.append({"sweep": "batch", **point(batch=b, residency=0.30)})

    # how much it costs to push more experts off the gpu
    for r in (1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0):
        rows.append({"sweep": "residency", **point(batch=4096, residency=r)})

    # depth, at a point where transfer and compute are close enough to matter
    for d in (1, 2, 4, 8):
        rows.append({"sweep": "depth",
                     **point(batch=4096, residency=0.30, depth=d)})

    # single-channel cxl, pcie, and the scaled 4-channel figure
    for bw in (16.88, 40.8, 67.5):
        rows.append({"sweep": "bandwidth",
                     **point(batch=4096, residency=0.30, bw=bw)})

    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / "15_systemc_sweep.csv"
    df.to_csv(out, index=False)
    print(f"  -> {out.relative_to(ROOT)}  ({len(df)} rows)")

    bs = df[df.sweep == "batch"]
    bstar = float(bs.b_star.iloc[0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

    ax[0].semilogx(bs.batch, bs.stall_fraction * 100, "o-", color="#2E7D8A")
    ax[0].axvline(bstar, ls="--", color="#C2703D", lw=1)
    ax[0].annotate(f"$B^*$ = {bstar:,.0f}", xy=(bstar, 55),
                   xytext=(6, 0), textcoords="offset points",
                   color="#C2703D", fontsize=9)
    ax[0].set_xlabel("batch (tokens)")
    ax[0].set_ylabel("stall, % of wall time")
    ax[0].set_title("SystemC TLM: stall against batch", fontsize=10)
    ax[0].grid(alpha=0.3)

    ax[1].semilogx(bs.batch, bs.tokens_per_second, "o-", color="#2E7D8A")
    ax[1].axvline(bstar, ls="--", color="#C2703D", lw=1)
    ax[1].set_xlabel("batch (tokens)")
    ax[1].set_ylabel("tokens / s")
    ax[1].set_title("Throughput saturates above $B^*$", fontsize=10)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "e5_systemc_crossover.png", dpi=150)
    print(f"  -> {(FIG / 'e5_systemc_crossover.png').relative_to(ROOT)}")

    # report the agreement between the three independent implementations.
    above = bs[bs.batch >= bstar]
    fill = above.stall_s.iloc[0]
    print(f"\nclosed form B*            = {bstar:,.0f} tokens")
    print(f"SystemC stall at B*       = {float(bs[bs.batch == 4626].stall_fraction.iloc[0]):.1%}")
    print(f"residual stall above B*   = {fill * 1e3:.2f} ms, constant "
          f"(= one layer transfer: pipeline fill, not steady-state stall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
