#!/usr/bin/env python3
"""Work out how to run a given MoE checkpoint on a given GPU.

    uv run python scripts/plan_offload.py --model olmoe --gpu a10
    uv run python scripts/plan_offload.py --model mixtral_8x7b --gpu a10 --batch 4096 16384
    uv run python scripts/plan_offload.py --model deepseek_v2_lite --gpu a10 \
        --calibrate results/runtime_deepseek_all.csv --at-batch 8192

Answers three questions, in the order you need them:

  1. Does the checkpoint fit, and how much of the expert pool has to move off
     the GPU for it to fit?
  2. Above what batch size does that transfer hide behind compute? This is B*
     from the report, evaluated for your model, card and link.
  3. What do you type to run it?

CPU only. It reads the same YAML configs as the rest of the analysis, so a new
model is a config file rather than a code change. See docs/PORTING.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memoe import load_gpu, load_model                      # noqa: E402
from memoe.memory import GPU, MemorySystem, Tier            # noqa: E402
from memoe.sim import Simulator                             # noqa: E402

GB = 1e9


@dataclass
class Calibration:
    flops: float
    reserve_gb: float | None
    batch: int


def layer_bytes(model) -> float:
    return model.n_experts * model.expert_bytes


def on_gpu_bytes(model, residency: float, depth: int) -> float:
    """Weights MEMoE-RT keeps on the GPU: non-expert weights and shared experts,
    the resident routed experts, and the staging ring of depth+1 layers."""
    per_layer = layer_bytes(model)
    resident = residency * model.n_moe_layers * per_layer
    staging = 0.0 if residency >= 1.0 else (depth + 1) * (1.0 - residency) * per_layer
    return model.resident_bytes + resident + staging


def calibrate(csv_path: Path, model, at_batch: int | None) -> Calibration:
    """Effective FLOP/s, and activation memory, from a measured run.

    Tokens per second rises with batch size, so calibrate at a batch near the
    one you plan to run (--at-batch). A fully resident row is used if there is
    one at that batch; otherwise the row with the least stall, with stall time
    subtracted so only compute is counted. That is how you calibrate a model
    too large to run resident.

    If the row records peak VRAM, whatever is left after the weights and the
    staging ring is taken as the activation and KV reserve for that batch.
    """
    rows = [r for r in csv.DictReader(csv_path.open())
            if r.get("tokens") and r.get("wall_seconds")]
    if not rows:
        raise SystemExit(f"no usable rows in {csv_path}")

    def batch(r):
        return int(float(r.get("batch_tokens") or r["tokens"]))

    def stall(r):
        return float(r.get("stall_seconds") or "inf")

    target = at_batch or max(batch(r) for r in rows)
    nearest = min(abs(batch(r) - target) for r in rows)
    rows = [r for r in rows if abs(batch(r) - target) == nearest]
    resident = [r for r in rows if float(r.get("residency", 0)) >= 0.999]
    r = resident[0] if resident else min(rows, key=stall)

    tokens = float(r["tokens"])
    compute_s = float(r["wall_seconds"]) - (0.0 if resident else stall(r))
    if compute_s <= 0:
        raise SystemExit("row has no compute time left after stall")
    per_expert_token = 2 * model.mats_per_expert * model.d_model * model.d_ff_expert
    flops = (model.n_moe_layers * per_expert_token
             * (model.top_k + model.n_shared_experts) * tokens) / compute_s

    reserve = None
    if r.get("peak_vram_gb"):
        depth = int(float(r.get("depth") or 2))      # both launchers default to 2
        used = on_gpu_bytes(model, float(r["residency"]), depth) / GB
        reserve = max(0.0, float(r["peak_vram_gb"]) - used)

    kind = "resident" if resident else "tiered, stall removed"
    print(f"calibrated on {csv_path.name}, batch {batch(r):,} ({kind}): "
          f"{flops / 1e12:.1f} TFLOP/s"
          + (f", {reserve:.1f} GB of activations and KV" if reserve is not None else ""))
    return Calibration(flops, reserve, batch(r))


def residency_that_fits(model, gpu_gb: float, reserve_gb: float, depth: int,
                        headroom_gb: float) -> float:
    """Largest share of experts that can stay resident, staging ring included."""
    budget = (gpu_gb - reserve_gb - headroom_gb) * GB
    for step in range(101):
        r = 1.0 - step / 100.0
        if on_gpu_bytes(model, r, depth) <= budget:
            return round(r, 2)
    return -1.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="configs/models name or a YAML path")
    p.add_argument("--gpu", required=True, help="configs/gpu name or a YAML path")
    p.add_argument("--link-gbs", type=float, help="host-to-device GB/s; overrides the GPU config")
    p.add_argument("--flops", type=float, help="effective FLOP/s; overrides the GPU config")
    p.add_argument("--calibrate", type=Path, help="CSV from a run to derive compute and memory from")
    p.add_argument("--at-batch", type=int, help="calibrate on the row closest to this batch")
    p.add_argument("--batch", type=int, nargs="+", default=[512, 2048, 8192, 16384],
                   help="batch sizes, in tokens, you intend to run")
    p.add_argument("--residency", type=float, nargs="+",
                   help="residency levels to evaluate; default picks a spread")
    p.add_argument("--depth", type=int, default=1, help="layers of lookahead")
    p.add_argument("--reserve-gb", type=float,
                   help="GPU memory for activations and KV cache (default: from "
                        "--calibrate, else 2 GB)")
    p.add_argument("--headroom-gb", type=float, default=1.5,
                   help="CUDA context and cuBLAS workspace, which peak VRAM "
                        "readings do not include")
    p.add_argument("--stall-budget", type=float, default=1.0,
                   help="tolerated stall as a fraction of compute; 1.0 is the "
                        "crossover where transfer time equals compute time")
    a = p.parse_args()

    model = load_model(a.model)
    gpu = load_gpu(a.gpu)
    cal = calibrate(a.calibrate, model, a.at_batch) if a.calibrate else None
    flops = a.flops or (cal.flops if cal else gpu.eff_flops)
    link = a.link_gbs or gpu.link_gbs
    reserve = a.reserve_gb if a.reserve_gb is not None else (
        cal.reserve_gb if cal and cal.reserve_gb is not None else 2.0)
    if not flops:
        raise SystemExit(f"{gpu.name} has no compute figure. Pass --flops, or "
                         "--calibrate with a CSV from a run on this card.")
    if not link:
        raise SystemExit(f"{gpu.name} has no link_gbs. Measure it with "
                         "scripts/measure_link.py, or pass --link-gbs.")

    pool_gb = model.routed_expert_bytes / GB
    print(f"\n{model.name} on {gpu.name}")
    print(f"  weights            {model.total_bytes / GB:>8.2f} GB")
    print(f"  routed experts     {pool_gb:>8.2f} GB  ({model.expert_fraction:.1%} of the "
          f"model, {model.n_experts} per layer x {model.n_moe_layers} MoE layers)")
    print(f"  always on GPU      {model.resident_bytes / GB:>8.2f} GB  "
          "(attention, dense layers, shared experts, embeddings)")
    print(f"  one layer          {layer_bytes(model) / GB:>8.2f} GB  (one transfer)")
    print(f"  GPU memory         {gpu.hbm_gb:>8.2f} GB, {reserve:.1f} GB kept for "
          f"activations and KV, {a.headroom_gb:.1f} GB for the CUDA context")
    print(f"  link               {link:>8.2f} GB/s")
    print(f"  compute            {flops / 1e12:>8.2f} TFLOP/s effective")

    r_fit = residency_that_fits(model, gpu.hbm_gb, reserve, a.depth, a.headroom_gb)
    if r_fit < 0:
        print("\n  Even with every expert off-GPU the rest of the model does not fit.")
        print("  Use a larger card, or shard the non-expert weights across GPUs.")
        return 1
    if r_fit >= 1.0:
        print(f"\n  Fits fully resident. Offloading frees up to {pool_gb:.1f} GB for "
              "KV cache and longer contexts.")
    else:
        print(f"\n  Fits with residency up to {r_fit:.2f}: at least "
              f"{1 - r_fit:.0%} of experts stream from host memory.")

    mem = MemorySystem(name="planned", hbm=Tier("GPU", gpu.hbm_gb, 0.0, 0.0),
                       cxl=Tier("host over link", 0.0, link, 0.0))
    sim = Simulator(model, mem, GPU(name=gpu.name, eff_flops_measured=flops,
                                    hbm_gb=gpu.hbm_gb))
    levels = a.residency or sorted({0.0, 0.25, 0.5, min(r_fit, 1.0)})

    print(f"\n  {'residency':>9} {'fits':>5} {'GPU GB':>7} {'host GB':>8} "
          f"{'B* tokens':>10}   (stall budget {a.stall_budget:.0%} of compute)")
    for r in levels:
        gpu_gb = on_gpu_bytes(model, r, a.depth) / GB + reserve
        host_gb = (1.0 - r) * pool_gb
        bstar = sim.critical_batch(expert_hit_rate=r, overhead=a.stall_budget)
        fits = "yes" if r <= r_fit else "no"
        print(f"  {r:>9.2f} {fits:>5} {gpu_gb:>7.1f} {host_gb:>8.1f} {bstar:>10,.0f}")

    pick = max((r for r in levels if r <= r_fit), default=0.0)
    if pick >= 1.0:
        pick = 0.0
    bstar = sim.critical_batch(expert_hit_rate=pick, overhead=a.stall_budget)
    print(f"\n  At residency {pick:.2f}:")
    for b in sorted(a.batch):
        verdict = ("transfer hides behind compute" if b >= bstar
                   else f"below the crossover, {bstar / b:.1f}x short")
        print(f"    batch {b:>7,}   {verdict}")

    print(f"\n  Run it (see docs/PORTING.md for the launcher environment):")
    print(f"    bash scripts/run_runtime.sh --ckpt <checkpoint> --check")
    print(f"    bash scripts/run_runtime.sh --ckpt <checkpoint> --batch {max(a.batch)} "
          f"--residency 1.0 {pick:.2f} --depth {a.depth} --trials 3 --out-dir results/port")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
