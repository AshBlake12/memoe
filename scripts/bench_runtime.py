#!/usr/bin/env python3
"""
Benchmark MEMoE-RT against a fully resident baseline.

    uv run python scripts/bench_runtime.py --check
    uv run python scripts/bench_runtime.py --batch 4096 --residency 1.0 0.6 0.4 0.2
    uv run python scripts/bench_runtime.py --trials 3   # for reportable numbers

Writes results/runtime_bench.csv (median across trials) and, when --trials > 1,
results/runtime_bench_trials.csv (every trial, nothing discarded).

--reps and --trials are not the same thing and the difference matters. --reps
runs the forward a few times inside one timed loop on one model load. --trials
tears the model down and does the whole sweep again. only the second one sees
load-to-load variance, and that is where the noise actually lives.

one trial cannot tell you whether a 2% difference is real. the run ends with a
monotonicity check that will say so if it isn't.

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

    def tiered(residency):
        m = load(args.ckpt)
        tier = tier_model(m, residency=residency, depth=args.depth, device=device)
        m.to(device)
        tier.warmup()
        out = m(ids).logits.float().cpu()
        del m, tier
        gc.collect(); torch.cuda.empty_cache()
        return out

    got = tiered(0.25)
    diff = (ref - got).abs()
    rel = (diff.max() / ref.abs().max()).item()
    print(f"vs reference:  max abs diff {diff.max().item():.5f}   relative {rel:.2e}")
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    top5 = (torch.topk(ref, 5, -1).indices == got.argmax(-1, keepdim=True)).any(-1)
    top5 = top5.float().mean().item()
    print(f"vs reference:  argmax agreement {agree:.4%}   in reference top-5 {top5:.4%}")

    # the reference gap is bf16 rounding in our MoE forward, and it is identical
    # at every residency; what offload itself must do is change nothing at all
    inv = (tiered(1.0) - got).abs().max().item()
    print(f"offload invariance: 25% vs 100% resident, max abs diff {inv:.5f}")
    ok = inv == 0.0 and top5 > 0.99
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@torch.no_grad()
def bench(args) -> int:
    device = torch.device(args.device)
    rows = []
    for trial in range(args.trials):
        if args.trials > 1:
            print(f"\n########## trial {trial + 1} of {args.trials} ##########")
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
                trial=trial,
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep every trial. this is the raw record, nothing gets thrown away
    trials_out = out_dir / "runtime_bench_trials.csv"
    with trials_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = _summarise(rows, args.trials)

    out = out_dir / "runtime_bench.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nwrote {out}  ({args.trials} trial(s))")
    if args.trials > 1:
        print(f"wrote {trials_out}  (per-trial raw)")

    base = next((x for x in summary if x["residency"] >= 0.999), None)
    print("\nthroughput relative to fully resident:")
    for x in summary:
        rel = x["tokens_per_second"] / base["tokens_per_second"] if base else float("nan")
        spread = (f"  spread {x['tps_spread_pct']:>4.1f}%"
                  if x["tps_spread_pct"] is not None else "")
        print(f"  residency {x['residency']:.2f}  "
              f"{x['tokens_per_second']:>9.1f} tok/s  "
              f"{rel:>6.1%}  stall {x['stall_fraction']:.1%}  "
              f"vram {x['peak_vram_gb']:.1f} GB{spread}")

    _report_monotonicity(summary, args.trials)
    return 0


def _summarise(rows, trials):
    """median across trials, per residency point.

    median not mean. one slow trial from a stray allocation or a clock blip
    shouldn't drag the reported number around, and with three trials the median
    is just the middle one.
    """
    import statistics as st

    out = []
    for r in sorted({x["residency"] for x in rows}, reverse=True):
        grp = [x for x in rows if x["residency"] == r]
        tps = [x["tokens_per_second"] for x in grp]
        med = dict(grp[0])
        med.pop("trial", None)
        for k in ("tokens_per_second", "wall_seconds", "stall_seconds",
                  "stall_fraction", "achieved_gbs", "peak_vram_gb"):
            vals = [x[k] for x in grp if x.get(k) is not None]
            if vals:
                med[k] = round(st.median(vals), 4)
        med["trials"] = len(grp)
        med["tps_spread_pct"] = (round(100 * (max(tps) - min(tps)) / st.median(tps), 2)
                                 if trials > 1 else None)
        out.append(med)
    return out


def _report_monotonicity(summary, trials):
    """throughput should fall as more experts move off the GPU, never rise.

    if it rises, the noise is bigger than the effect at those two points and
    neither of them belongs in a figure yet. run more trials.
    """
    bad = []
    for a, b in zip(summary, summary[1:]):          # descending residency
        if b["tokens_per_second"] > a["tokens_per_second"]:
            gap = 100 * (b["tokens_per_second"] - a["tokens_per_second"]) \
                  / a["tokens_per_second"]
            bad.append((a["residency"], b["residency"], gap))
    if not bad:
        print("\nmonotonicity: OK (throughput falls as residency falls)")
        return
    print("\nmonotonicity: INVERTED at " +
          ", ".join(f"{a:.2f}->{b:.2f} (+{g:.1f}%)" for a, b, g in bad))
    if trials < 3:
        print("  Re-run with --trials 3. With one trial an inversion this size "
              "is not distinguishable from run-to-run noise.")
    else:
        print("  Inversion persists across trials. Compare against "
              "tps_spread_pct before reporting either point.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4096, help="tokens per forward")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--reps", type=int, default=3,
                   help="forwards inside one timed loop (averaged)")
    p.add_argument("--trials", type=int, default=1,
                   help="independent repeats of the whole sweep, model reloaded "
                        "each time; results/runtime_bench.csv holds the median. "
                        "Use 3 for anything that goes in a figure.")
    p.add_argument("--depth", type=int, default=2, help="layers of lookahead")
    p.add_argument("--residency", type=float, nargs="+",
                   default=[1.0, 0.6, 0.4, 0.2])
    p.add_argument("--check", action="store_true")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    return correctness(args) if args.check else bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
