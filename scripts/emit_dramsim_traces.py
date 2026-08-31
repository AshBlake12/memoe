#!/usr/bin/env python3
"""Emit DRAMSim3 address traces for the two access patterns MEMoE cares about.

  expert_fetch : bulk sequential read of one expert tensor. This is what a
                 CXL miss actually looks like -- one large contiguous pull.
  kv_stream    : strided read of a KV cache slice; no reuse, small records.
  random       : 64 B random reads, the pathological case, for contrast.

DRAMSim3's trace CPU issues a request as soon as the memory system will accept
it, so setting every added_cycle to 0 applies maximum pressure and measures
sustained bandwidth rather than a latency-bound trickle.
"""
import argparse, random
from pathlib import Path

LINE = 64


def emit(path, n_req, stride=LINE, base=0x10000000, mode="seq", seed=0):
    rng = random.Random(seed)
    span = n_req * stride
    with open(path, "w") as f:
        for i in range(n_req):
            if mode == "seq":
                addr = base + i * stride
            else:
                addr = base + (rng.randrange(span // LINE) * LINE)
            f.write(f"0x{addr:x} READ 0\n")
    print(f"{path}  {n_req} requests  {n_req*LINE/2**20:.1f} MiB touched")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dramsim")
    ap.add_argument("--expert-mb", type=float, default=12.0,
                    help="expert size in MiB (OLMoE=12, Qwen3=36, DeepSeek=84)")
    a = ap.parse_args()
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)

    n = int(a.expert_mb * 2**20 // LINE)
    emit(d / "expert_fetch.trace", n, LINE, mode="seq")
    emit(d / "kv_stream.trace", n // 4, 256, mode="seq")     # strided, sparse
    emit(d / "random.trace", n // 4, LINE, mode="rand")
