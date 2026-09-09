#!/usr/bin/env python3
"""
Turn the gem5 sweep into a latency-versus-utilisation curve.

    uv run python scripts/parse_gem5.py

Reads results/gem5/load*/stats.txt, writes results/gem5_knee.csv and
results/figures_extra/e4_knee.png, and prints where the knee sits.

gem5 renamed its stats to snake_case around v21.2, so every lookup here tries
both spellings rather than assuming a version.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEAK_GBS = 16.88
STATS_DIR = Path("results/gem5")
CSV_OUT = Path("results/gem5_knee.csv")
FIG_OUT = Path("results/figures_extra/e4_knee.png")

# each entry is a list of spellings to try, newest first.
KEYS = {
    "sim_seconds": ["simSeconds", "sim_seconds"],
    "read_latency_mean": [
        "system.monitor.readLatencyHist::mean",
        "system.monitor.read_latency_hist::mean",
    ],
    "monitor_bw": [
        "system.monitor.readBandwidthHist::mean",
        "system.monitor.read_bandwidth_hist::mean",
    ],
    "bytes_read": [
        "system.mem_ctrl.bytesRead::total",
        "system.mem_ctrl.bytes_read::total",
        "system.mem_ctrl.bytesRead",
        "system.mem_ctrl.bytes_read",
    ],
    "read_reqs": [
        "system.mem_ctrl.numReads::total",
        "system.mem_ctrl.num_reads::total",
    ],
}


def load_stats(path: Path) -> dict[str, float]:
    stats: dict[str, float] = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        try:
            stats[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return stats


def pick(stats: dict[str, float], names: list[str]) -> float | None:
    for n in names:
        if n in stats:
            return stats[n]
    return None


def main() -> None:
    dirs = sorted(STATS_DIR.glob("load*"), key=lambda p: int(p.name[4:]))
    if not dirs:
        raise SystemExit(f"no runs under {STATS_DIR}. Run gem5/sweep_knee.sh first.")

    rows = []
    missing_latency = False
    for d in dirs:
        f = d / "stats.txt"
        if not f.exists():
            print(f"skipping {d.name}, no stats.txt")
            continue
        s = load_stats(f)
        offered_pct = int(d.name[4:])

        secs = pick(s, KEYS["sim_seconds"])
        by = pick(s, KEYS["bytes_read"])
        lat_ticks = pick(s, KEYS["read_latency_mean"])
        mon_bw = pick(s, KEYS["monitor_bw"])

        if lat_ticks is None:
            missing_latency = True
            mon = [k for k in s if k.startswith("system.monitor")]
            if mon and not rows:
                print("could not find a latency histogram. Monitor stats present:")
                for k in mon[:25]:
                    print("   ", k)

        # commMonitor reports bandwidth directly and is the more reliable
        # source; fall back to the memory controller byte counter.
        if mon_bw:
            achieved = mon_bw / 1e9
        else:
            achieved = (by / secs / 1e9) if (by and secs) else None
        rows.append(dict(
            offered_pct=offered_pct,
            offered_gbs=round(offered_pct / 100 * PEAK_GBS, 3),
            achieved_gbs=round(achieved, 3) if achieved else "",
            utilisation_pct=round(100 * achieved / PEAK_GBS, 2) if achieved else "",
            mean_read_latency_ns=round(lat_ticks / 1000, 2) if lat_ticks else "",
        ))

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV_OUT} ({len(rows)} points)")

    pts = [(r["utilisation_pct"], r["mean_read_latency_ns"]) for r in rows
           if r["utilisation_pct"] != "" and r["mean_read_latency_ns"] != ""]
    if not pts:
        print("no latency data, so no figure. Fix the stat name above and rerun.")
        return

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    base = min(y)

    knee = None
    for u, lat in pts:
        if lat >= 2 * base:
            knee = u
            break

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    ax.plot(x, y, marker="o", color="#9A4C10", linewidth=2)
    ax.axhline(base, color="#DCE0E6", linewidth=1, linestyle="--")
    if knee is not None:
        ax.axvline(knee, color="#A61B2B", linewidth=1.2, linestyle=":")
        ax.annotate(f"latency doubles at {knee:.0f}% utilisation",
                    xy=(knee, 2 * base), xytext=(6, 10),
                    textcoords="offset points", fontsize=10, color="#A61B2B")
    ax.set_xlabel("achieved bandwidth as share of the 16.88 GB/s tier, percent")
    ax.set_ylabel("mean read latency, ns")
    ax.set_title("Queueing under load on the CXL tier", loc="left", fontsize=12)
    ax.grid(axis="y", color="#EDEFF2")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT)
    print(f"wrote {FIG_OUT}")

    print()
    print(f"unloaded latency      {base:.1f} ns")
    if knee:
        print(f"latency doubles at    {knee:.0f}% utilisation")
        if knee < 80:
            print("  -> below 80%. Our overhead numbers are optimistic and the report")
            print("     needs a derating factor. This is a real result, say so.")
        else:
            print("  -> at or above 80%. The analytical model holds in the regime we use.")
    else:
        print("latency never doubled inside the swept range. Extend the sweep past 110%.")
    if missing_latency:
        print("\nsome runs had no latency histogram. Check CommMonitor stat names.")


if __name__ == "__main__":
    main()
