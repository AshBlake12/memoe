#!/usr/bin/env python3
"""residency sweep, one point per process.

why this exists: the expert pool is pinned host memory, and pytorch's pinned
allocator caches those blocks rather than handing them back to the OS. building
the tier several times in one process therefore accumulates ~12-18 GB of shmem
per build and the OOM killer takes you out on the fourth. torch.cuda.empty_cache()
doesn't touch it, and neither does dropping the references.

so: fork per point. slower, finishes.

    python scripts/sweep_residency.py                       # 7 points, 3 trials
    python scripts/sweep_residency.py --trials 5 --batch 8192

writes results/runtime_bench_trials.csv (every run) and
results/runtime_bench.csv (median per point), same columns as before so the
existing plot scripts keep working.
"""
import argparse
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def one(residency, batch, depth, seq, reps, use_uv):
    """run a single point in a fresh process, return its result dict or None."""
    cmd = (["uv", "run", "python"] if use_uv else [sys.executable]) + [
        "scripts/bench_runtime.py",
        "--residency", str(residency),
        "--batch", str(batch),
        "--depth", str(depth),
        "--seq", str(seq),
        "--reps", str(reps),
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    # the bench prints one json block per point; take the last complete one
    blocks = re.findall(r"\{[^{}]*\"tokens_per_second\"[^{}]*\}", p.stdout, re.S)
    if not blocks:
        killed = p.returncode < 0 or "Killed" in p.stderr
        print(f"    no result (exit {p.returncode}"
              f"{', looks like the OOM killer' if killed else ''})")
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        for t in tail:
            print(f"      | {t[:110]}")
        return None
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        print("    could not parse the result block")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residency", type=float, nargs="+",
                    default=[1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-uv", action="store_true",
                    help="call python directly instead of `uv run python`")
    a = ap.parse_args()

    rows = []
    total = len(a.residency) * a.trials
    n = 0
    for t in range(a.trials):
        for r in a.residency:
            n += 1
            print(f"[{n}/{total}] residency {r:.2f}, trial {t + 1}", flush=True)
            got = one(r, a.batch, a.depth, a.seq, a.reps, not a.no_uv)
            if got:
                got["trial"] = t
                got["residency"] = r
                rows.append(got)
                print(f"    {got['tokens_per_second']:>9.1f} tok/s   "
                      f"{got['peak_vram_gb']:>5.2f} GB   "
                      f"stall {got.get('stall_fraction', 0):.1%}", flush=True)

    if not rows:
        print("\nnothing completed.")
        return 1

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    import csv
    cols = list(rows[0].keys())
    with (out / "runtime_bench_trials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # median per residency
    summary = []
    for r in sorted({x["residency"] for x in rows}, reverse=True):
        grp = [x for x in rows if x["residency"] == r]
        med = dict(grp[0])
        med.pop("trial", None)
        for k in ("tokens_per_second", "wall_seconds", "stall_seconds",
                  "stall_fraction", "achieved_gbs", "peak_vram_gb"):
            vals = [x[k] for x in grp if x.get(k) is not None]
            if vals:
                med[k] = round(st.median(vals), 4)
        tps = [x["tokens_per_second"] for x in grp]
        med["trials"] = len(grp)
        med["tps_spread_pct"] = round(100 * (max(tps) - min(tps)) / st.median(tps), 2)
        summary.append(med)

    with (out / "runtime_bench.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    base = next((x for x in summary if x["residency"] >= 0.999), None)
    print(f"\n{'resid':>6} {'median tok/s':>13} {'spread':>7} {'vs full':>8} "
          f"{'vram':>7} {'trials':>7}")
    for x in summary:
        rel = (f"{x['tokens_per_second'] / base['tokens_per_second']:>7.1%}"
               if base else "      -")
        print(f"{x['residency']:>6.2f} {x['tokens_per_second']:>13.1f} "
              f"{x['tps_spread_pct']:>6.1f}% {rel} "
              f"{x['peak_vram_gb']:>6.2f}G {x['trials']:>7}")

    worst = max(x["tps_spread_pct"] for x in summary)
    print(f"\nworst spread across trials: {worst:.1f}%")

    bad = []
    for p, q in zip(summary, summary[1:]):
        if q["tokens_per_second"] > p["tokens_per_second"]:
            gap = 100 * (q["tokens_per_second"] - p["tokens_per_second"]) \
                  / p["tokens_per_second"]
            bad.append((p["residency"], q["residency"], gap,
                        max(p["tps_spread_pct"], q["tps_spread_pct"])))

    if not bad:
        print("monotonicity: OK, throughput falls as residency falls")
    else:
        print("monotonicity: still inverted at")
        for x, y, gap, spread in bad:
            verdict = ("inside the trial spread, so it is noise"
                       if gap <= spread else
                       "LARGER than the trial spread, worth investigating")
            print(f"  {x:.2f} -> {y:.2f}: +{gap:.1f}% ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
