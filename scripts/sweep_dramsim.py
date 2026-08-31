#!/usr/bin/env python3
"""Sweep DRAMSim3 across CXL-plausible DRAM back-ends.

A CXL Type-3 expander is DRAM behind a controller. Its sustained bandwidth is
set by the speed grade and channel count the vendor put behind the link, so
that is exactly what we sweep. The result is a MEASURED bandwidth curve for
the CXL tier instead of a cited range.

Usage (from the repo root):
    python scripts/sweep_dramsim.py --dramsim ~/DRAMsim3
"""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path

LINE = 64
GRADES = ["DDR4_8Gb_x8_1866", "DDR4_8Gb_x8_2133", "DDR4_8Gb_x8_2400",
          "DDR4_8Gb_x8_2666", "DDR4_8Gb_x8_2933", "DDR4_8Gb_x8_3200"]
CHANNELS = [1, 2, 4]


def set_channels(src, dst, n):
    txt = Path(src).read_text()
    if re.search(r"^\s*channels\s*=", txt, re.M):
        txt = re.sub(r"^(\s*channels\s*=\s*)\d+", rf"\g<1>{n}", txt, flags=re.M)
    else:
        txt = txt.replace("[system]", f"[system]\nchannels = {n}", 1)
    Path(dst).write_text(txt)


def tck_ns(ini):
    m = re.search(r"^\s*tCK\s*=\s*([0-9.]+)", Path(ini).read_text(), re.M)
    return float(m.group(1))


def run(dramsim, ini, trace, outdir, cycles):
    outdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(Path(dramsim) / "build" / "dramsim3main"), str(ini),
                        "-c", str(cycles), "-t", str(trace), "-o", str(outdir)],
                       capture_output=True, text=True)
    j = outdir / "dramsim3.json"
    if not j.exists():
        print("  FAILED:", r.stderr.strip()[:200]); return None
    d = json.loads(j.read_text())
    ch = [v for v in d.values() if isinstance(v, dict) and "num_reads_done" in v]
    reads = sum(c["num_reads_done"] for c in ch)
    if not reads:
        return None
    ia = sum(c["average_interarrival"] * c["num_reads_done"] for c in ch) / reads
    lat = sum(c["average_read_latency"] * c["num_reads_done"] for c in ch) / reads
    tck = tck_ns(ini)
    bw = LINE / (ia * tck * 1e-9) / 1e9 * len(ch)
    ceiling = LINE / (tck * 1e-9) / 1e9          # 1 request/cycle trace-CPU limit
    return {"channels": len(ch), "tCK_ns": tck, "reads": reads,
            "sustained_gbs": round(bw, 2), "latency_ns": round(lat * tck, 1),
            "issue_ceiling_gbs": round(ceiling, 1),
            "issue_limited": bw > 0.9 * ceiling}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dramsim", default=str(Path.home() / "DRAMsim3"))
    ap.add_argument("--trace", default="results/dramsim/expert_fetch.trace")
    ap.add_argument("--out", default="results/dramsim/sweep")
    ap.add_argument("--cycles", type=int, default=3000000)
    a = ap.parse_args()

    cfgdir = Path(a.out) / "configs"; cfgdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in GRADES:
        src = Path(a.dramsim) / "configs" / f"{g}.ini"
        if not src.exists():
            print(f"skip {g} (no config)"); continue
        for n in CHANNELS:
            ini = cfgdir / f"{g}_ch{n}.ini"
            set_channels(src, ini, n)
            res = run(a.dramsim, ini, a.trace, Path(a.out) / f"{g}_ch{n}", a.cycles)
            if not res: continue
            res.update(grade=g.replace("DDR4_8Gb_x8_", "DDR4-"), requested_channels=n)
            rows.append(res)
            flag = "  ISSUE-LIMITED" if res["issue_limited"] else ""
            print(f"  {res['grade']:>9} x{res['channels']}ch  "
                  f"{res['sustained_gbs']:6.1f} GB/s  "
                  f"{res['latency_ns']:6.1f} ns{flag}", flush=True)

    out = Path(a.out) / "sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(Path(a.out) / "sweep.csv", index=False)
    except ImportError:
        pass
    print(f"\n-> {out}  ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
