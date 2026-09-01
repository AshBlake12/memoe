#!/usr/bin/env python3
"""
MEMoE-RT on DeepSeek-V2-Lite: a second architecture, on a GPU that cannot hold
the model at all.

    uv run --python 3.11 --with "transformers==4.44.2" --with torch \
        --with accelerate python scripts/bench_deepseek.py --check
    uv run ... python scripts/bench_deepseek.py --batch 4096 --residency 0.0

DeepSeek-V2-Lite is 15.7B parameters, about 31 GB in bfloat16. The expert pool
alone is 28.8 GB across 26 MoE layers of 64 experts at 17.3 MB each. The A10 we
run it on has 23 GB, so the fully resident baseline does not exist: this model
cannot run on this GPU without offload. We therefore report absolute throughput
and the highest residency that fits, rather than a retention percentage.

It also differs from OLMoE in ways that make it a real test rather than a
repeat: top-6 of 64 instead of top-8, two shared experts that fire on every
token, a dense first layer, and multi-head latent attention.

The model ships its own modelling code written against transformers 4.x. We
load it from a local copy at ~/dsmod, patched to drop an unconditional
flash_attn import that 5.x-era environments cannot satisfy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

CKPT = "deepseek-ai/DeepSeek-V2-Lite"


def load(dsmod_path: str, ckpt: str, dtype=torch.bfloat16):
    sys.path.insert(0, os.path.dirname(dsmod_path.rstrip("/")))
    from dsmod.modeling_deepseek import DeepseekV2ForCausalLM
    from dsmod.configuration_deepseek import DeepseekV2Config

    cfg = DeepseekV2Config.from_pretrained(ckpt)
    cfg._attn_implementation = "eager"
    print(f"loading {ckpt} to CPU")
    m = DeepseekV2ForCausalLM.from_pretrained(
        ckpt, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True)
    m.eval()
    return m


def report_footprint(model) -> dict:
    from memoe.runtime_ml import discover_modulelist_blocks
    specs = discover_modulelist_blocks(model)
    s0 = specs[0]
    per_expert = 3 * s0.d_model * s0.d_ff * 2          # bytes, bf16
    pool = per_expert * s0.n_experts * len(specs)
    total = sum(p.numel() for p in model.parameters()) * 2
    info = {
        "moe_layers": len(specs),
        "experts_per_layer": s0.n_experts,
        "expert_mb": round(per_expert / 1e6, 2),
        "expert_pool_gb": round(pool / 1e9, 2),
        "total_weights_gb": round(total / 1e9, 2),
        "experts_share": round(100 * pool / total, 1),
    }
    print(json.dumps(info, indent=2))
    return info


@torch.no_grad()
def check(args) -> int:
    """
    Correctness on this model cannot be checked against a fully resident
    baseline, because a fully resident baseline does not fit on this GPU. We
    instead compare a high-residency run against a low-residency one: both use
    the streaming path, so agreement shows the transfer and view binding are
    correct.
    """
    import gc
    from memoe.runtime_ml import tier_model_modulelist

    dev = torch.device(args.device)
    m = load(args.dsmod, args.ckpt)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, m.config.vocab_size, (2, 128), generator=g).to(dev)

    tier = tier_model_modulelist(m, residency=0.5, depth=args.depth, device=dev)
    tier.warmup()
    a = m(ids).logits.float().cpu()
    del m, tier
    gc.collect(); torch.cuda.empty_cache()

    m = load(args.dsmod, args.ckpt)
    tier = tier_model_modulelist(m, residency=0.0, depth=args.depth, device=dev)
    tier.warmup()
    b = m(ids).logits.float().cpu()

    diff = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"max abs diff {diff:.5f}   argmax agreement {agree:.4%}")
    ok = agree > 0.90
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@torch.no_grad()
def bench(args) -> int:
    import gc
    from memoe.runtime_ml import tier_model_modulelist

    dev = torch.device(args.device)
    rows = []
    for r in args.residency:
        print(f"\n=== residency {r:.2f} ===")
        m = load(args.dsmod, args.ckpt)
        info = report_footprint(m)
        tier = tier_model_modulelist(m, residency=r, depth=args.depth, device=dev)

        g = torch.Generator().manual_seed(0)
        n_seq = max(1, args.batch // args.seq)
        ids = torch.randint(0, m.config.vocab_size, (n_seq, args.seq),
                            generator=g).to(dev)
        tokens = n_seq * args.seq

        m.model(ids[:1, :8])
        torch.cuda.synchronize()
        tier.reset_stats()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            tier.warmup()
            m.model(ids)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        tier.collect_stalls()

        row = dict(model="DeepSeek-V2-Lite", residency=r,
                   experts_offloaded=tier.stats.offloaded_experts,
                   offloaded_fraction=round(
                       tier.stats.offloaded_experts / info["experts_per_layer"], 3),
                   moe_layers=info["moe_layers"],
                   expert_pool_gb=info["expert_pool_gb"],
                   peak_vram_gb=round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
                   **tier.stats.summary(secs, tokens * args.reps))
        rows.append(row)
        print(json.dumps(row, indent=2))

        del m, tier
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    out = Path("results/runtime_deepseek.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--dsmod", default=os.path.expanduser("~/dsmod"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--residency", type=float, nargs="+", default=[0.0])
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    return check(a) if a.check else bench(a)


if __name__ == "__main__":
    raise SystemExit(main())
