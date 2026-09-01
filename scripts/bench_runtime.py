#!/usr/bin/env python3
"""
Benchmark MEMoE-RT against a fully resident baseline.

    uv run python scripts/bench_runtime.py --check
    uv run python scripts/bench_runtime.py --batch 4096 --residency 1.0 0.6 0.4 0.2

Writes results/runtime_bench.csv.

Run --check first. It compares tiered output against the reference
implementation on the same input. If that does not pass, no timing number
below it means anything.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

from memoe.runtime import discover_moe_blocks, tier_model

CKPT = "allenai/OLMoE-1B-7B-0924-Instruct"


def load(ckpt: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    print(f"loading {ckpt} to CPU")
    m = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    m.eval()
    return m


def make_input(model, batch_tokens: int, seq_len: int, device):
    vocab = model.config.vocab_size
    n_seq = max(1, batch_tokens // seq_len)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (n_seq, seq_len), generator=g)
    return ids.to(device), n_seq * seq_len


@torch.no_grad()
def timed_forward(model, ids, reps: int, tier=None):
    torch.cuda.synchronize()
    model(ids[:1, :8])  # warm kernels
    if tier is not None:
        tier.reset_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        if tier is not None:
            tier.warmup()
        model(ids)
        if tier is not None:
            torch.cuda.synchronize()
            tier.collect_stalls()
    return time.perf_counter() - t0


@torch.no_grad()
def correctness(args) -> int:
    device = torch.device(args.device)
    print("=== correctness ===")
    ref_model = load(args.ckpt).to(device)
    ids, _ = make_input(ref_model, 512, 128, device)
    ref = ref_model(ids).logits.float().cpu()
    del ref_model
    gc.collect(); torch.cuda.empty_cache()

    m = load(args.ckpt)
    tier = tier_model(m, residency=0.25, depth=args.depth, device=device)
    m.to(device)
    tier.warmup()
    got = m(ids).logits.float().cpu()

    diff = (ref - got).abs()
    rel = (diff.max() / ref.abs().max()).item()
    print(f"max abs diff {diff.max().item():.5f}   relative {rel:.2e}")
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(f"argmax agreement {agree:.4%}")
    ok = agree > 0.90
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@torch.no_grad()
def bench(args) -> int:
    device = torch.device(args.device)
    rows = []
    for r in args.residency:
        print(f"\n=== residency {r:.2f} ===")
        m = load(args.ckpt)
        specs = discover_moe_blocks(m)
        n_exp, layers = specs[0].n_experts, len(specs)
        tier = tier_model(m, residency=r, depth=args.depth, device=device)
        m.to(device)
        ids, tokens = make_input(m, args.batch, args.seq, device)

        secs = timed_forward(m, ids, args.reps, tier)
        s = tier.stats.summary(secs, tokens * args.reps)

        offload_frac = tier.stats.offloaded_experts / n_exp
        gb_resident = torch.cuda.max_memory_allocated(device) / 1e9
        row = dict(
            residency=r,
            offloaded_fraction=round(offload_frac, 3),
            experts_offloaded=tier.stats.offloaded_experts,
            layers=layers,
            batch_tokens=tokens,
            peak_vram_gb=round(gb_resident, 2),
            **s,
        )
        rows.append(row)
        print(json.dumps(row, indent=2))

        del m, tier
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    out = Path("results/runtime_bench.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")

    base = next((x for x in rows if x["residency"] >= 0.999), None)
    if base:
        print("\nthroughput relative to fully resident:")
        for x in rows:
            rel = x["tokens_per_second"] / base["tokens_per_second"]
            print(f"  residency {x['residency']:.2f}  "
                  f"{x['tokens_per_second']:>9.1f} tok/s  "
                  f"{rel:>6.1%}  stall {x['stall_fraction']:.1%}  "
                  f"vram {x['peak_vram_gb']:.1f} GB")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4096, help="tokens per forward")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--depth", type=int, default=2, help="layers of lookahead")
    p.add_argument("--residency", type=float, nargs="+",
                   default=[1.0, 0.6, 0.4, 0.2])
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    return correctness(args) if args.check else bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
