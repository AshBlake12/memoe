# MEMoE — MoE-Oriented Memory Expansion

**Team King Bob** (Shashank, Shreyansh Jain, Kanav Sharma) · Nebula 2026 · Software

Can the expert weights of a Mixture-of-Experts model live outside GPU memory, and
what does it cost in throughput?

The answer is yes, inside a regime whose boundary we can state in closed form and
have measured on real hardware. `results/REPORT.md` has the numbers;
`results/dashboard.html` presents them interactively and needs nothing installed.

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
together). What is left is the hit rate, and it enters linearly.

Measured against the runtime, the closed form predicts the crossover to within
0.5% on OLMoE.

**What follows from that.** Offload is a prefill technique. At prefill batch sizes
the entire expert pool can sit off-GPU for about 7% throughput; at decode batch
sizes the same configuration costs the majority of it.

## Three findings that changed the design

1. **Batching amortises the fetch, but coverage saturates.** Past a few hundred
   tokens a batch touches essentially every expert at every layer. The GPU tier
   stops being a cache and becomes a fixed partition.
2. **Popularity buys nothing at serving scale.** Fetch hit rate equals resident
   fraction exactly, regardless of routing distribution. LRU, LFU and static
   placement are equivalent, so MEMoE-RT contains no ranking anywhere.
3. **Hot expert sets are anti-correlated across workloads.** Prose vs. code
   overlap is 15.2% against 25% chance, with an 84.9% within-domain control. A
   placement profiled on one domain does worse than random on another.

Together these move the recommendation away from popularity-based tiering toward
bandwidth-window prefetch scheduling plus a capacity decision.

## Deliverables

Where each item in the problem statement is addressed.

| Deliverable | Code | Report |
|---|---|---|
| HBM + CXL memory architecture model | `memoe/memory.py`, `configs/memory/` | §3 |
| CXL-based Transformer & MoE workload analysis | `memoe/hooks.py`, `results/traces/` | §4, §6 |
| Bandwidth, latency & scalability evaluation | `scripts/sweep_dramsim.py`, `scripts/parse_gem5.py` | §3, §13 |
| Memory tiering & expert placement strategy | `memoe/policy.py`, `memoe/sim.py` | §4, §5, §7 |
| CXL memory expansion & pooling study | `memoe/analysis.py` | §12 |
| Simulation framework, HBM-only vs HBM+CXL | `memoe/sim.py`, `systemc/` | §8 |
| Interactive demo / dashboard | `scripts/build_dashboard.py` | `results/dashboard.html` |
| Final report with recommendations | — | §15 |

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

configs/        model / memory / gpu YAML
scripts/        see below
tests/          36 tests
results/        tables*/ figures*/ traces/ qemu/ REPORT.md dashboard.html
paper/          LaTeX source (working copy lives in Overleaf)
```

### Scripts

| | |
|---|---|
| `reproduce.sh` | everything below the runtime, one command, CPU only |
| `run_all.py` | main tables and figures, `REPORT.md` |
| `run_extras.py` | KV-vs-experts, pooling, transfer granularity |
| `run_real.py` | results from the captured OLMoE traces |
| `frontier.py` | binary search for maximum offloadable fraction |
| `make_paper_tables.py` | LaTeX tables straight from `results/tables/` |
| `build_dashboard.py` | writes `results/dashboard.html` |
| `capture_traces.py` | hook a live checkpoint, capture routing across 4 domains |
| `hotset_overlap.py` | cross-domain hot-set overlap |
| `bench_runtime.py` | **GPU** MEMoE-RT on OLMoE: correctness, residency, batch, depth |
| `bench_serve.py` | **GPU** serving loop, KV cache, continuous batching |
| `bench_deepseek.py` | **GPU** MEMoE-RT on DeepSeek-V2-Lite |
| `run_deepseek.sh` | launcher pinning the environment DeepSeek needs |
| `setup_dsmod.sh` | builds the patched DeepSeek modelling package |
| `emit_dramsim_traces.py`, `sweep_dramsim.py`, `parse_dramsim.py` | DRAMSim3 tier characterisation |
| `parse_gem5.py` | gem5 channel-scaling results |
| `demo_a10.sh` | one command per demo shot on the A10 machine |
| `gem5/run_channels.sh` | gem5 1/2/4/8-channel sweep, config in `gem5/channel_scaling.py` |

Building gem5, SystemC, DRAMSim3 and QEMU without root and running every
component on one A10 machine: `docs/SETUP.md`. Video script: `docs/demo-script-a10.md`.

## Reproducing

```bash
pip install -r requirements.txt
scripts/reproduce.sh          # ~8 min, CPU only, no downloads
```

Every CSV, `paper/generated_tables.tex` and `results/REPORT.md` in this repo
regenerate **byte-identical** from a clean checkout. Nothing in the analysis
pipeline depends on a GPU, a checkpoint or a network fetch.

The runtime benchmarks are separate because they need hardware:

```bash
pip install -e ".[runtime]"
python scripts/bench_runtime.py --check       # correctness first
python scripts/bench_runtime.py --trials 3    # then timing
```

`--check` compares tiered output against the unmodified reference on identical
input. If it does not pass, no timing number below it means anything.

**Use `--trials 3` for any number that goes in a figure.** `--reps` averages
forwards inside one timed loop on one model load; `--trials` reloads the model and
repeats the whole sweep, which is the only thing that captures load-to-load
variance. `results/runtime_bench.csv` holds the median, `runtime_bench_trials.csv`
holds every trial, and the run ends with a monotonicity check: throughput must not
rise as more experts move off the GPU, and if it does, the spread is larger than
the effect at those points.

For DeepSeek-V2-Lite, run `scripts/setup_dsmod.sh` first — the checkpoint's remote
code has an unconditional `flash_attn` import that has to be guarded — then
`scripts/run_deepseek.sh --check`.

## Hardware these numbers come from

| | |
|---|---|
| `ramesh` | RTX PRO 4500 Blackwell, PCIe 5.0, 32.6 GB — OLMoE runtime and serving |
| `bhaskar` | A10, PCIe 4.0, 23 GB, 125 GB host RAM — DeepSeek, gem5, QEMU |

DeepSeek-V2-Lite is 31.4 GB in bfloat16 on a 23 GB GPU: the fully resident
baseline does not exist, so that path reports absolute throughput rather than a
retention percentage.

## Simulation and the CXL software path

**DRAMSim3** characterises the tier: 16.88 GB/s sustained and 179.4 ns loaded
device latency on DDR4-3200, from a 12 MiB sequential expert-fetch trace. The
reported bandwidth is diluted by idle cycles, so the honest figure comes from
`average_interarrival`. The tier config adds 120 ns for controller and link hop,
putting modelled CXL latency at ~300 ns, inside the published 200–400 ns band.

**gem5** measures channel scaling at ~4x for four channels, which is what
validates the scaled four-channel bandwidth figure. That figure is derived from
measured scaling; it is not measured on a four-channel CXL device, because no such
device was available.

**QEMU** exercises the Linux CXL stack end to end. On a stock Ubuntu kernel
everything works up to region commit, which is gated on a cache-invalidation check
that returns false whenever `X86_FEATURE_HYPERVISOR` is set. We compiled Linux 6.8
with `CONFIG_CXL_REGION_INVALIDATION_TEST=y` using a user-local toolchain (no root
anywhere in the build), and the region commits: 2 GB at `0x490000000`, interleave
ways 1, target `decoder2.0`, with the bypass logged in dmesg. Capture in
`results/qemu/cxl_committed.txt`.

Two limits worth stating before anyone else does. The region was committed but not
mapped and driven with traffic, so the claim is about region management, not a
working memory tier. And QEMU's CXL support is functional emulation with no timing
model — nothing from it is performance evidence. Every quantitative CXL number
here comes from DRAMSim3 and the analytical model.

## Substituting PCIe for CXL

We do not have CXL silicon. Host DRAM over PCIe is the standard substitute: a
large, slow, byte-addressable tier behind a link. At PCIe 5.0 x16 the measured
plateau of 40.8 GB/s sits between our single-channel CXL figure of 16.88 GB/s and
the scaled four-channel 67.5 GB/s. It is not CXL — different latency, no cache
coherence — but it reproduces the structure of the problem, which is what the
analytical model is about.

## Honesty constraints, enforced by tests

These are the things a reviewer checks first, so they are assertions rather than
intentions.

- CXL bandwidth is always the **measured sustained** 18–52 GB/s range, never the
  theoretical link rate. `test_cxl_bandwidth_within_measured_range`
- CXL 3.0 pooling is pre-production; any config using it is flagged
  `measured: false` and every result carries a MODELED note.
  `test_pooled_cxl_is_flagged_as_modeled`
- Extrapolated bandwidth points (>52 GB/s) are marked and drawn dotted.
- The footprint model reproduces published parameter counts for all four
  checkpoints before it is trusted for anything else.
- The compute model counts routed-expert GEMMs only. Attention and dense layers
  would hide more transfer, so reported overheads are conservative.

## Windows note

PowerShell 5.1's `Set-Content -Encoding UTF8` writes a BOM, which turns the first
YAML key into `\ufeffname` and breaks parsing in a way that looks like a missing
field. All config reads use `read_text(encoding="utf-8-sig")`, and
`test_bom_prefixed_yaml_still_parses` feeds the loader a BOM-prefixed file so this
cannot regress.
