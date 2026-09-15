# Running MEMoE-RT on another MoE model or GPU

MEMoE-RT isn't tied to OLMoE or to the two GPUs in the report. Porting it takes a
config file for the model, two measurements of the GPU, and one planning run. After
that you run the same correctness check and benchmark as the report. This guide
goes through it in order, using DeepSeek-V2-Lite on an NVIDIA A10 as the example
wherever a real number helps.

| Step | Tool | What you get |
|---|---|---|
| 0. Smoke test the machine | `scripts/make_tiny_moe.py` | a small checkpoint to run `--check` on, no download |
| 1. Describe the model | `configs/models/<name>.yaml` | expert pool size, what stays on the GPU |
| 2. Describe the GPU | `scripts/measure_link.py`, `plan_offload.py --calibrate` | link bandwidth, effective compute, activation memory |
| 3. Plan | `scripts/plan_offload.py` | largest residency that fits, crossover batch B*, the command |
| 4. Check | `scripts/run_runtime.sh --check` | PASS when offload changes nothing |
| 5. Measure | `scripts/run_runtime.sh --trials 3` | throughput, stall and peak memory per residency |

You need a CUDA GPU, [uv](https://docs.astral.sh/uv/), and enough host RAM to hold
the experts you offload (the planner prints this) on top of loading the checkpoint.
Offloaded experts sit in pinned memory, so if your system caps locked memory
(`ulimit -l`), raise it.

## 0. Smoke test the machine

Build a small random checkpoint and run the correctness check on it. It takes a
couple of minutes, downloads no model, and exercises the same code as a real run.

```bash
uv run --no-project --python 3.11 --with "transformers>=5.0.0" --with "torch>=2.5.0" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    python scripts/make_tiny_moe.py --arch qwen3_moe --out /tmp/tiny-qwen3moe
bash scripts/run_runtime.sh --ckpt /tmp/tiny-qwen3moe --check      # expect PASS
```

Ignore throughput on a model this small, since kernel overhead dominates. It is for
`--check` only.

## 1. Describe the model

Create `configs/models/<name>.yaml`. Every value comes from the checkpoint's
`config.json`:

| Our key | Mixtral | Qwen3-MoE | DeepSeek-V2/V3 |
|---|---|---|---|
| `d_model` | `hidden_size` | `hidden_size` | `hidden_size` |
| `d_ff_expert` | `intermediate_size` | `moe_intermediate_size` | `moe_intermediate_size` |
| `n_experts` | `num_local_experts` | `num_experts` | `n_routed_experts` |
| `top_k` | `num_experts_per_tok` | `num_experts_per_tok` | `num_experts_per_tok` |
| `n_layers` | `num_hidden_layers` | `num_hidden_layers` | `num_hidden_layers` |
| `n_moe_layers` | all layers | layers not in `mlp_only_layers` | `num_hidden_layers - first_k_dense_replace` |
| `n_shared_experts` | 0 | 0 | `n_shared_experts` |
| `d_ff_dense` | 0 | 0 | `intermediate_size` |
| `n_heads`, `n_kv_heads`, `d_head` | attention fields | attention fields | attention fields |

Then check the footprint against the parameter count on the model card:

```bash
uv run python -c "from memoe import load_model; print(load_model('deepseek_v2_lite').summary())"
```

If the total is off, the attention formula doesn't match the architecture (DeepSeek's
multi-head latent attention is the usual case). Set `nonexpert_params_override` to the
published total minus routed and shared expert parameters. `deepseek_v2_lite.yaml`
does this, and its footprint matches the live checkpoint: 31.41 GB of weights and a
28.79 GB routed pool. A test in `tests/test_memoe.py` keeps it that way.

## 2. Describe the GPU

Two numbers decide the plan: how fast the link moves bytes, and how fast the card
computes.

**Link.** Measure host-to-device bandwidth in about ten seconds:

```bash
uv run --no-project --python 3.11 --with "torch>=2.5.0" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    python scripts/measure_link.py
```

On the A10 this plateaus at 25.23 GB/s from 4 MB transfers upward. A whole layer of
experts is hundreds of megabytes, well onto the plateau.

**Compute.** Run the model once and let the planner derive effective FLOP/s from the
CSV. Throughput per token rises with batch size, so calibrate at a batch close to the
one you will serve.

- If the model fits resident, run the unmodified model:
  `bash scripts/run_runtime.sh --ckpt <checkpoint> --batch 8192 --residency 1.0 --out-dir results/port`
- If it doesn't fit, run fully offloaded instead. The planner subtracts stall time so
  only compute is counted:
  `bash scripts/run_runtime.sh --ckpt <checkpoint> --batch 8192 --residency 0.0 --out-dir results/port`

```bash
uv run python scripts/plan_offload.py --model deepseek_v2_lite --gpu a10 \
    --calibrate results/runtime_deepseek_all.csv --at-batch 8192
# calibrated on runtime_deepseek_all.csv, batch 8,192 (tiered, stall removed):
#   23.7 TFLOP/s, 4.6 GB of activations and KV
```

The same run also gives the activation and KV memory at that batch, from peak VRAM
minus weights and staging buffers.

Write both numbers into `configs/gpu/<name>.yaml`:

```yaml
name: A10
hbm_gb: 23.0
link_gbs: 25.2
eff_flops_measured: 14600000000000
```

`a10.yaml` and `rtx_pro_4500.yaml` are the two measured cards from the report.

## 3. Plan

```bash
uv run python scripts/plan_offload.py --model deepseek_v2_lite --gpu a10 \
    --calibrate results/runtime_deepseek_all.csv --at-batch 8192 --batch 4096 6144 8192
```

```
DeepSeek-V2-Lite on A10
  weights               31.41 GB
  routed experts        28.79 GB  (91.7% of the model, 64 per layer x 26 MoE layers)
  always on GPU          2.62 GB  (attention, dense layers, shared experts, embeddings)
  ...
  Fits with residency up to 0.45: at least 55% of experts stream from host memory.

  residency  fits  GPU GB  host GB  B* tokens   (stall budget 100% of compute)
       0.00   yes     9.4     28.8      7,509
       0.25   yes    16.1     21.6      5,631
       0.45   yes    21.4     15.8      4,130
```

How to read it:

- **residency** is the share of each layer's experts kept on the GPU. Only the amount
  matters, not which experts, so the runtime keeps the lowest indices.
- **GPU GB** and **host GB** are what that residency costs on each side. The planner
  also leaves 1.5 GB for the CUDA context and cuBLAS workspace (`--headroom-gb`),
  since peak VRAM readings don't include them.
- **B\*** is the batch, in tokens, above which transfer hides behind compute. Serve
  above it. Measured on the A10, DeepSeek-V2-Lite's stall drops from 23% at 6,144
  tokens to 0.01% at 8,192, and the plan puts B\* at 7,509.
- `--stall-budget 0.1` gives the batch at which stall is 10% of compute instead of
  the crossover.

As a rule of thumb, prefill batches sit above B\* and decode batches below it. On a
serving deployment, put offload on the prefill workers.

## 4. Pick the launcher environment

`scripts/run_runtime.sh` and `scripts/run_serve.sh` build a throwaway environment with
`uv run --no-project`. Four variables choose what goes into it:

| Variable | Default | Change it when |
|---|---|---|
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cu121` | your driver or GPU needs other CUDA wheels |
| `TORCH` | `torch>=2.5.0` | you pin a torch version |
| `TRANSFORMERS` | `transformers>=5.0.0` | the checkpoint needs an older transformers |
| `PY` | `3.11` | you need a different Python |

| GPU generation | CUDA wheels |
|---|---|
| Ampere and Ada (A10, A100, L4, RTX 30/40) | cu121 or later, matching the driver |
| Hopper (H100, H200) | cu121 or later |
| Blackwell (RTX 50, RTX PRO) | cu128, with torch 2.7 or later |

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128 TORCH="torch>=2.7.0" \
    bash scripts/run_runtime.sh --ckpt <checkpoint> --check
```

The runtime picks its code path from how the checkpoint stores experts:

| Layout | Examples | How MEMoE-RT attaches |
|---|---|---|
| fused tensors (`gate_up_proj`, `down_proj`) | OLMoE, Qwen3-MoE, Mixtral on transformers 5.x | replaces the MoE block with a tiered one |
| `ModuleList` of experts | Mixtral on transformers 4.x, DeepSeek-V2 remote code | points expert weights at the staging ring, drives transfers from hooks |

Models that ship their own code through `trust_remote_code` sometimes import
packages you don't have, `flash_attn` being the common one. `scripts/setup_dsmod.sh`
shows the fix: copy the modelling files locally and guard the import.

## 5. Check correctness

```bash
bash scripts/run_runtime.sh --ckpt <checkpoint> --check
```

This runs the unmodified model, then the tiered model at 25% and 100% residency, on
identical input. PASS means offloaded output matches the 100%-resident run exactly
(max abs diff 0), and at least 99% of tiered top-1 tokens fall inside the unmodified
model's top 5. The small gap to the unmodified model is bfloat16 rounding in the
MoE forward pass, and it doesn't change with residency.

If the model is too large to load resident, compare two residencies that do fit:

```bash
bash scripts/run_runtime.sh --ckpt <checkpoint> --check --check-residency 0.0 0.25
```

## 6. Measure

```bash
bash scripts/run_runtime.sh --ckpt <checkpoint> --batch 16384 \
    --residency 1.0 0.45 0.0 --depth 1 --trials 3 --out-dir results/port
```

Use `--trials 3` for any number you report. Each trial reloads the model, and the
run ends by checking that throughput falls as residency falls. Use depth 1: one layer
of lookahead matched depth 8 on throughput with 41% less staging memory in the report.
`results/port/runtime_bench.csv` has the medians, and `runtime_bench_trials.csv` has
every trial.

Run the batch sweep across B\*: stall should fall sharply between a batch below the
planned B\* and one above it, as it does in the report on both GPUs. For a serving
loop, `run_serve.sh` takes the same `--ckpt`, `--residency` and `--depth` flags.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUBLAS_STATUS_EXECUTION_FAILED` or CUDA OOM | GPU memory full; cuBLAS can't get workspace | lower `--residency` or the batch; rerun the planner with the measured reserve |
| `Found no MoE blocks` / `Unrecognised expert layout` | expert weights have names the runtime doesn't know | add the projection names to `_TRIPLES` in `memoe/runtime_ml.py` (it knows `gate_proj/up_proj/down_proj` and `w1/w3/w2`) |
| host OOM or `pin_memory` fails | not enough host RAM, or a locked-memory limit | offload less, add RAM, or raise `ulimit -l` |
| `no kernel image is available` / `sm_120 is not compatible` | torch built without your GPU's architecture | set `TORCH_INDEX` to cu128 and `TORCH="torch>=2.7.0"` for Blackwell |
| `ImportError` from remote code | the checkpoint's modelling code imports an optional package | copy it locally and guard the import, as `setup_dsmod.sh` does |
| offloaded run faster than resident | tiny model, overhead-dominated | measure a real checkpoint at a real batch |
| repeated residency points get OOM-killed | pinned host memory is cached between builds in one process | `scripts/sweep_residency.py` runs one process per point |

## Verified on the A10

| Checkpoint | Layout | transformers | `--check` |
|---|---|---|---|
| OLMoE-1B-7B | fused | 5.x | PASS; invariance exact, 99.41% top-5 containment |
| Qwen3-MoE, small random | fused | 5.x | PASS; invariance exact, 100% top-5 containment |
| Mixtral, small random | fused | 5.x | PASS; invariance exact, 100% top-5 containment |
| Mixtral, small random | ModuleList | 4.44 | PASS; bit-identical to the unmodified model |
| DeepSeek-V2-Lite | ModuleList | 4.44 (remote code) | PASS via `run_deepseek.sh --check`; 25% vs 0% resident identical; full sweep in the report |

The output of these checks is in `results/porting_checks_a10.txt`.
