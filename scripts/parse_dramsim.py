#!/usr/bin/env python3
"""Extract real sustained bandwidth and latency from a DRAMSim3 run.

DRAMSim3's own `average_bandwidth` divides bytes moved by the FULL `-c`
cycle budget, including every idle cycle after the trace has drained. Run a
12 MiB trace with -c 3000000 and the figure is diluted by whatever fraction
of those cycles were spent doing nothing. Use `average_interarrival` instead:
the mean cycles between accepted requests, which is the rate the memory system
sustained while it was busy.

    sustained BW = 64 B / (average_interarrival * tCK)
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

LINE = 64


def tck_ns(ini_path):
    txt = Path(ini_path).read_text()
    m = re.search(r"^\s*tCK\s*=\s*([0-9.]+)", txt, re.M)
    if not m:
        raise SystemExit(f"no tCK in {ini_path}")
    return float(m.group(1))


def summarise(json_path, ini_path, label):
    d = json.loads(Path(json_path).read_text())
    tck = tck_ns(ini_path)
    chans = [v for v in d.values() if isinstance(v, dict) and "num_reads_done" in v]
    if not chans:
        raise SystemExit(f"no channel stats in {json_path}")

    reads = sum(c["num_reads_done"] for c in chans)
    ia = sum(c["average_interarrival"] * c["num_reads_done"] for c in chans) / reads
    lat_cyc = sum(c["average_read_latency"] * c["num_reads_done"] for c in chans) / reads

    per_chan_bw = LINE / (ia * tck * 1e-9) / 1e9
    total_bw = per_chan_bw * len(chans)
    naive = sum(c["average_bandwidth"] for c in chans)

    print(f"\n=== {label}  ({Path(ini_path).name})")
    print(f"  channels                {len(chans)}")
    print(f"  tCK                     {tck} ns")
    print(f"  reads completed         {reads:,}")
    print(f"  bytes moved             {reads*LINE/2**20:.1f} MiB")
    print(f"  mean interarrival       {ia:.2f} cycles")
    print(f"  SUSTAINED BANDWIDTH     {total_bw:.1f} GB/s   <-- use this")
    print(f"  (DRAMSim3 reported      {naive:.1f} GB/s, diluted by idle cycles)")
    print(f"  loaded read latency     {lat_cyc*tck:.1f} ns  ({lat_cyc:.0f} cycles)")
    return {"label": label, "channels": len(chans), "reads": reads,
            "sustained_gbs": round(total_bw, 2),
            "loaded_latency_ns": round(lat_cyc * tck, 1)}


if __name__ == "__main__":
    T = sys.argv[1] if len(sys.argv) > 1 else "results/dramsim"
    D = sys.argv[2] if len(sys.argv) > 2 else str(Path.home() / "DRAMsim3")
    out = [
        summarise(f"{T}/cxl/dramsim3.json",
                  f"{D}/configs/DDR4_8Gb_x8_3200.ini", "CXL tier (DDR4-3200)"),
        summarise(f"{T}/hbm/dramsim3.json",
                  f"{D}/configs/HBM2_8Gb_x128.ini", "HBM tier (HBM2)"),
    ]
    Path(f"{T}/calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {T}/calibration.json")
    c, h = out
    print(f"\nHBM : CXL bandwidth ratio = "
          f"{h['sustained_gbs']/c['sustained_gbs']:.1f}x")
