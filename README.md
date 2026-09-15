# MEMoE: MoE-Oriented Memory Expansion

Team King Bob (Shashank, Shreyansh Jain, Kanav Sharma) · Nebula 2026 · Software

Can the expert weights of a Mixture-of-Experts model live outside GPU memory, and
what does it cost in throughput?

Yes, inside a regime whose boundary we can write in closed form and have measured
on real hardware. `results/REPORT.md` has the numbers. `results/dashboard.html`
shows them interactively and needs nothing installed.

## The result

Per MoE layer per batch, with `E` experts, `k` activated per token, expert-fetch
hit rate `h`, per-GPU effective compute `F` and tier bandwidth `BW`, the fetch
hides behind compute once

```
B  >  B*  =  (1 - h) · E · F / (BW · k · ε)
```

where `ε` is the tolerated stall as a fraction of compute. `B*` does not depend on
expert size (a bigger expert costs more to fetch and does proportionally more
work) and does not depend on GPU count (compute and per-node bandwidth scale
together). That leaves the hit rate, which enters linearly.

On OLMoE the closed form gives a crossover of 4,626 tokens. The runtime's measured
transfer and compute times put it at 4,645, and an independent SystemC model puts
it at 4,626.

So offload is a prefill technique. With every expert off-GPU at a batch of 16,384
tokens, OLMoE keeps 91.2% of resident throughput in 2.44x less GPU memory. At
decode batch sizes the same configuration costs most of it.

## Three findings that changed the design

1. **Batching amortises the fetch, but coverage saturates.** Past a few hundred
   tokens a batch touches nearly every expert at every layer. The GPU tier
   stops being a cache and becomes a fixed partition.
2. **Popularity buys nothing at serving scale.** Fetch hit rate equals resident
   fraction exactly, whatever the routing distribution. LRU, LFU and static
   placement come out the same, so MEMoE-RT has no ranking in it at all.
3. **The wrong profile is worse than none.** Prose and code hot sets overlap
   15.2%, below the 25% expected by chance, while each workload overlaps 84.9%
   with itself. A placement profiled on prose serves code worse than random.

Put together, these pushed us away from popularity-based tiering and toward
bandwidth-window prefetch scheduling plus a capacity decision.

## Deliverables

Where each item in the problem statement is addressed.

| Deliverable | Code | Report |
|---|---|---|
| HBM + CXL memory architecture model | `memoe/memory.py`, `configs/memory/` | §3 |
| CXL-based Transformer & MoE workload analysis | `memoe/hooks.py`, `results/traces/` | §4, §6 |
| Bandwidth, latency & scalability evaluation | `scripts/sweep_dramsim.py`, `gem5/` | §3, §13 |
| Memory tiering & expert placement strategy | `memoe/policy.py`, `memoe/sim.py` | §4, §5, §7 |
| CXL memory expansion & pooling study | `memoe/analysis.py`, `scripts/run_extras.py` | §12 |
| Simulation framework, HBM-only vs HBM+CXL | `memoe/sim.py`, `systemc/` | §8 |
| Interactive demo / dashboard | `scripts/build_dashboard.py` | `results/dashboard.html` |
| Final report with recommendations | | §15 |

Tools from the brief: Python, C/C++ (`systemc/`), gem5, SystemC, DRAMSim3,
QEMU with CXL support, open-source MoE traces, pandas, matplotlib, plotly.

## What is in here

```
memoe/
  moe.py        analytical footprint model (validated against published counts)
  memory.py     HBM / CXL tier and GPU compute specs
  trace.py      routing traces; Gumbel-top-k Zipf synthesiser
  skew.py       entropy, Gini, fitted Zipf exponent, coverage curve
  policy.py     static-popularity, balanced-static, LRU, sampled-LFU
  sim.py        trace-driven tiered-memory simulator + closed-form B*
  analysis.py   capacity, batch-regime, hit-rate and prefetch-depth analyses
  hooks.py      architecture-agnostic router capture for real checkpoints
  runtime.py    MEMoE-RT: tiered execution for HF-style MoE blocks
  runtime_ml.py second path for ModuleList-style experts (DeepSeek)
  serve.py      serving loop with real KV cache and continuous batching
  config.py     YAML loading (utf-8-sig; see "Windows" below)

configs/        model / memory / gpu YAML (add a file here to plan a new model or GPU)
scripts/        see below
gem5/           channel-scaling config and sweep
systemc/        TLM-2.0 model of the tiered expert path
tests/          38 tests
docs/           SETUP.md (building everything without root), PORTING.md (new models and GPUs), related work
paper/          LaTeX source of the report: main.tex, diagrams/ (pdflatex main.tex)
results/        tables*/ figures*/ traces/ dramsim/ gem5/ qemu/ REPORT.md dashboard.html
```

### Scripts

| | |
|---|---|
| `reproduce.sh` | everything below the runtime, one command, CPU only |
| `run_all.py` | main tables and figures, `REPORT.md` |
| `run_extras.py` | KV-vs-experts, pooling, transfer granularity |
| `run_real.py` | results from the captured OLMoE traces |
| `frontier.py` | binary search for maximum offloadable fraction |
| `build_dashboard.py` | writes `results/dashboard.html` |
| `capture_traces.py` | hook a live checkpoint, capture routing across 4 domains |
| `hotset_overlap.py` | cross-domain hot-set overlap |
| `plan_offload.py` | plan a model on a GPU: what fits, crossover batch B*, the command to run |
| `measure_link.py` | **GPU** host-to-device bandwidth, the `link_gbs` a GPU config needs |
| `bench_runtime.py` | **GPU** MEMoE-RT on any MoE checkpoint (OLMoE by default): correctness, residency, batch, depth |
| `sweep_residency.py` | **GPU** residency sweep, one process per point (avoids pinned-memory OOM) |
| `bench_serve.py` | **GPU** serving loop, KV cache, continuous batching |
| `bench_deepseek.py` | **GPU** MEMoE-RT on DeepSeek-V2-Lite |
| `run_runtime.sh`, `run_serve.sh`, `run_deepseek.sh` | launchers that pin the Python/torch environment each benchmark needs |
| `setup_dsmod.sh` | builds the patched DeepSeek modelling package |
| `plot_runtime.py`, `plot_deepseek.py`, `plot_serve.py` | figures from the measured runtime and serving CSVs |
| `emit_dramsim_traces.py`, `sweep_dramsim.py`, `parse_dramsim.py` | DRAMSim3 tier characterisation |
| `run_systemc.py` | SystemC sweep, writes `results/tables/15_systemc_sweep.csv` |
| `run_a10.sh` | one command per component on the A10 machine |
| `gem5/run_channels.sh` | gem5 1/2/4/8-channel sweep, config in `gem5/channel_scaling.py` |

`docs/SETUP.md` covers building gem5, SystemC, DRAMSim3 and QEMU without root and
running every component on one A10 machine. `docs/PORTING.md` shows how to run
MEMoE-RT on a different MoE model or a different GPU.

## Setup with uv

The project is managed with [uv](https://docs.astral.sh/uv/). Dependencies are
declared in `pyproject.toml` and pinned in `uv.lock`, so everyone gets the same
versions.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
# Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex
uv --version
```

No root needed. uv installs into `~/.local/bin` and downloads a Python for you if
the machine doesn't have 3.10 or newer.

### 2. Create the environment

```bash
git clone https://github.com/AshBlake12/memoe.git && cd memoe
uv sync --extra dev          # creates .venv from uv.lock, includes pytest
```

Pass `--extra dev`. A plain `uv sync` installs only the core packages and will
remove pytest if it's already in the environment.

### 3. Run things

Prefix commands with `uv run` and they execute inside `.venv`; you never need to
activate it.

```bash
uv run python -m pytest tests -q          # 38 passed
uv run bash scripts/reproduce.sh          # ~8 min, CPU only, no downloads
uv run python scripts/build_dashboard.py  # just the dashboard
```

### What gets installed

| Group | Packages | Used for |
|---|---|---|
| core (`uv sync`) | numpy, pandas, pyyaml, matplotlib, tabulate, plotly | analysis, simulator, tables, figures, dashboard |
| `--extra dev` | pytest | the test suite |
| `--extra runtime` | torch, transformers, accelerate, datasets | GPU runtime and trace capture, if you manage your own CUDA setup |

The GPU benchmarks don't use the project environment. Each launcher builds a
throwaway Python 3.11 environment with `uv run --no-project` and the exact CUDA
12.1 wheels its model needs, so nothing has to be installed by hand:

| Launcher | Python | Packages |
|---|---|---|
| `scripts/run_runtime.sh` (OLMoE) and `scripts/run_serve.sh` | 3.11 | torch>=2.5.0 (cu121), transformers>=5.0.0, accelerate, numpy, pandas, pyyaml |
| `scripts/run_deepseek.sh` (DeepSeek-V2-Lite) | 3.11 | torch==2.4.1 (cu121), transformers==4.44.2, accelerate, numpy, pandas, pyyaml |

DRAMSim3, SystemC, gem5 and QEMU are native builds, covered in `docs/SETUP.md`.
If you'd rather use pip, `requirements.txt` lists the core and test packages.

## Reproducing

```bash
uv run bash scripts/reproduce.sh
```

Every CSV and `results/REPORT.md` in this repo regenerates byte-identical from a
clean checkout. Nothing in the analysis pipeline needs a GPU, a checkpoint or a
network fetch.

The runtime benchmarks are separate because they need hardware:

```bash
bash scripts/run_runtime.sh --check       # correctness first
bash scripts/run_runtime.sh --trials 3    # then timing
```

`--check` compares tiered output against the unmodified reference on identical
input. If it fails, ignore every timing number after it.

Use `--trials 3` for any number that goes in a figure. `--reps` averages
forwards inside one timed loop on one model load. `--trials` reloads the model and
repeats the whole sweep, and only that captures load-to-load variance.
`results/runtime_bench.csv` holds the median, `runtime_bench_trials.csv` holds
every trial, and the run ends with a monotonicity check: throughput must not rise
as more experts move off the GPU. If it does, the spread is larger than the effect
at those points.

For DeepSeek-V2-Lite, run `scripts/setup_dsmod.sh` first (the checkpoint's remote
code has an unconditional `flash_attn` import that has to be guarded), then
`scripts/run_deepseek.sh --check`.

## Hardware these numbers come from

| | |
|---|---|
| `ramesh` | RTX PRO 4500 Blackwell, PCIe 5.0, 32.6 GB: OLMoE runtime and serving |
| `bhaskar` | A10, PCIe 4.0, 23 GB, 125 GB host RAM: DeepSeek, gem5, QEMU |

DeepSeek-V2-Lite is 31.4 GB in bfloat16 and the A10 has 23 GB, so there is no
fully resident baseline. That path reports absolute throughput instead of a
retention percentage.

## Simulation and the CXL software path

**DRAMSim3** characterises the tier: 16.88 GB/s sustained and 179.4 ns loaded
device latency on DDR4-3200, from a 12 MiB sequential expert-fetch trace.
DRAMSim3's reported bandwidth is diluted by idle cycles, so we compute it from
`average_interarrival` instead. The tier config adds 120 ns for the controller and
link hop, which puts modelled CXL latency at ~300 ns, inside the published
200-400 ns band.

**gem5** measures channel scaling at ~4x for four channels. That is what backs
the scaled four-channel bandwidth figure. The figure comes from measured scaling,
not from a four-channel CXL device, because we didn't have one.

**QEMU** runs the Linux CXL stack end to end. On a stock Ubuntu kernel everything
works up to region commit, which is gated on a cache-invalidation check that
returns false whenever `X86_FEATURE_HYPERVISOR` is set. We compiled Linux 6.8
with `CONFIG_CXL_REGION_INVALIDATION_TEST=y` using a user-local toolchain (no root
anywhere in the build), and the region commits: 2 GB at `0x490000000`, interleave
ways 1, target `decoder2.0`, with the bypass logged in dmesg. Capture in
`results/qemu/cxl_committed.txt`.

QEMU validates the software path. Every quantitative CXL number here comes from
DRAMSim3, gem5 and the analytical model.

## Substituting PCIe for CXL

Host DRAM over PCIe is the standard stand-in for a CXL tier: a large, slow,
byte-addressable tier behind a link. At PCIe 5.0 x16 the measured plateau of
40.8 GB/s sits between our single-channel CXL figure of 16.88 GB/s and the scaled
four-channel 67.5 GB/s. MEMoE-RT moves whole layers in bulk, which depends on
bandwidth rather than per-access latency, so the problem has the same structure,
and the structure is what the analytical model describes.

## Constraints enforced by tests

A reviewer will check these first, so they are assertions in the test suite.

- CXL bandwidth is always the measured sustained 18-52 GB/s range, never the
  theoretical link rate. `test_cxl_bandwidth_within_measured_range`
- CXL 3.0 pooling is modelled. Any config using it is flagged
  `measured: false` and every result carries a MODELED note.
  `test_pooled_cxl_is_flagged_as_modeled`
- Extrapolated bandwidth points (>52 GB/s) are marked and drawn dotted.
- The footprint model has to reproduce published parameter counts for all four
  checkpoints before anything else uses it.
- The compute model counts routed-expert GEMMs only. Attention and dense layers
  would hide more transfer, so reported overheads are conservative.

## Windows note

PowerShell 5.1's `Set-Content -Encoding UTF8` writes a BOM, which turns the first
YAML key into `﻿name` and breaks parsing in a way that looks like a missing
field. All config reads use `read_text(encoding="utf-8-sig")`, and
`test_bom_prefixed_yaml_still_parses` feeds the loader a BOM-prefixed file so this
can't come back.
