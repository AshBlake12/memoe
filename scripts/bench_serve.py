#!/usr/bin/env python3
"""
Measure what expert offload costs in a serving setting.

    uv run python scripts/bench_serve.py --requests 32 --prompt 2048 --gen 64

Runs the same workload three ways and writes results/serve_bench.csv:

  resident     every expert in GPU memory, the conventional deployment
  offload      every expert streamed, one engine serving both phases
  phase_aware  prefill measured under offload, decode measured resident

The third is the disaggregated arrangement. Prefill workers and decode workers
are separate processes on separate hardware in such a deployment, so we measure
each phase in the configuration it would occupy and report both. We do not model
the KV handoff between them; that cost is real and we say so rather than
inventing a number for it.

The comparison that matters is not offload against resident overall. It is
time-to-first-token under offload, which should be close to resident because
prefill is a large batch, against time-per-output-token under offload, which
should be far worse because decode is not.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from memoe.serve import ServeEngine, make_workload

CKPT = "allenai/OLMoE-1B-7B-0924-Instruct"


def load(ckpt, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    print(f"loading {ckpt}")
    m = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    m.eval()
    return m


def build(args, residency: float):
    """Return (model, tier). residency 1.0 means nothing is offloaded."""
    from memoe.runtime import tier_model
    m = load(args.ckpt)
    if residency >= 0.999:
        m.to(args.device)
        return m, None
    tier = tier_model(m, residency=residency, depth=args.depth,
                      device=args.device)
    m.to(args.device)
    return m, tier


def run_one(args, residency, label):
    m, tier = build(args, residency)
    eng = ServeEngine(m, tier=tier, device=args.device,
                      max_running=args.max_running,
                      max_prefill_tokens=args.max_prefill_tokens)
    reqs = make_workload(args.requests, args.prompt, args.gen)
    st = eng.run(reqs, label)
    row = st.summary()
    row["residency"] = residency
    print(json.dumps(row, indent=2))
    del m, tier, eng
    gc.collect(); torch.cuda.empty_cache()
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--max-running", type=int, default=32)
    p.add_argument("--max-prefill-tokens", type=int, default=16384)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--residency", type=float, default=0.0,
                   help="residency used for the offloaded configurations")
    a = p.parse_args()

    rows = []
    print("\n=== resident: conventional deployment ===")
    res = run_one(a, 1.0, "resident")
    rows.append(res)

    print("\n=== offload: one engine, both phases ===")
    off = run_one(a, a.residency, "offload")
    rows.append(off)

    # phase-aware: prefill from the offloaded run, decode from the resident one.
    pa = {
        "config": "phase_aware",
        "residency": f"{a.residency} prefill / 1.0 decode",
        "requests": res["requests"],
        "prompt_tokens": res["prompt_tokens"],
        "output_tokens": res["output_tokens"],
        "wall_seconds": round(off["prefill_seconds"] + res["decode_seconds"], 3),
        "output_tokens_per_second": round(
            res["output_tokens"] / (off["prefill_seconds"] + res["decode_seconds"]), 1),
        "ttft_mean_ms": off["ttft_mean_ms"],
        "ttft_p95_ms": off["ttft_p95_ms"],
        "tpot_mean_ms": res["tpot_mean_ms"],
        "tpot_p95_ms": res["tpot_p95_ms"],
        "prefill_seconds": off["prefill_seconds"],
        "decode_seconds": res["decode_seconds"],
        "prefill_stall_seconds": off["prefill_stall_seconds"],
        "decode_stall_seconds": 0.0,
        "peak_vram_gb": f"{off['peak_vram_gb']} prefill / {res['peak_vram_gb']} decode",
    }
    rows.append(pa)

    out = Path("results/serve_bench.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(res.keys())
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    print("\n--- comparison ---")
    print(f"{'config':<14}{'TTFT ms':>10}{'TPOT ms':>10}{'tok/s':>10}{'VRAM GB':>26}")
    for r in rows:
        print(f"{r['config']:<14}{r['ttft_mean_ms']:>10.1f}{r['tpot_mean_ms']:>10.2f}"
              f"{r['output_tokens_per_second']:>10.1f}{str(r['peak_vram_gb']):>26}")

    if res["tpot_mean_ms"]:
        print(f"\ndecode is {off['tpot_mean_ms'] / res['tpot_mean_ms']:.2f}x slower "
              f"under offload; prefill is "
              f"{off['ttft_mean_ms'] / res['ttft_mean_ms']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
