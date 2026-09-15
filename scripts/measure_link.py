#!/usr/bin/env python3
"""Measure host-to-device bandwidth, which is what the tier delivers.

    uv run --no-project --python 3.11 --with "torch>=2.5.0" \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        python scripts/measure_link.py

Transfers pinned host memory to the GPU over a separate stream, the same way
MEMoE-RT moves a layer of experts, and reports sustained bandwidth per transfer
size. Small transfers are latency-bound, large ones reach the link plateau. Put
the plateau in your GPU config as link_gbs, and the whole-layer transfer size of
your model will sit on or near it.

Takes about ten seconds on a new machine.
"""

from __future__ import annotations

import argparse

import torch

SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def measure(size_mb: int, device: torch.device, reps: int) -> float:
    """Sustained GB/s for one transfer size, best of `reps`."""
    n = size_mb * 1024 * 1024
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)

    best = 0.0
    for _ in range(reps):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(stream):
            start.record(stream)
            dev.copy_(host, non_blocking=True)
            end.record(stream)
        torch.cuda.synchronize()
        gbs = n / 1e9 / (start.elapsed_time(end) / 1000.0)
        best = max(best, gbs)

    del host, dev
    torch.cuda.empty_cache()
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES_MB,
                   help="transfer sizes in MB")
    a = p.parse_args()

    device = torch.device(a.device)
    name = torch.cuda.get_device_name(device)
    gen = torch.cuda.get_device_properties(device)
    print(f"{name}, {gen.total_memory / 1e9:.1f} GB\n")
    print(f"{'transfer':>10} {'GB/s':>9}")

    results = []
    for mb in a.sizes:
        gbs = measure(mb, device, a.reps)
        results.append((mb, gbs))
        print(f"{mb:>7} MB {gbs:>9.2f}")

    plateau = max(g for _, g in results)
    knee = next(mb for mb, g in results if g >= 0.95 * plateau)
    print(f"\nplateau {plateau:.2f} GB/s, reached by {knee} MB transfers")
    print(f"put link_gbs: {plateau:.1f} in your configs/gpu/*.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
