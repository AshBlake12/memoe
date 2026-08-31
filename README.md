# MEMoE — MoE-Oriented Memory Expansion

**Team King Bob** (Shashank, Shreyansh Jain, Kanav Sharma) · Nebula 2026 · Software

An analysis and simulation framework for answering one question:

> Can CXL-attached memory hold the expert weights of a large Mixture-of-Experts
> model, and what does it cost in throughput?

The short answer is that it can, but only inside a regime we can now describe
exactly. See `results/REPORT.md` for the numbers.

## The result

Per MoE layer per batch, with `E` experts, `k` activated per token, expert size
`S`, expert-fetch hit rate `h`, per-GPU effective compute `F` and CXL bandwidth
`BW`, the stall is hidden behind compute once

```
B  >  B*  =  (1 - h) · E · F / (BW · k · ε)
```

where `ε` is the tolerated stall as a fraction of compute. Two things about
`B*` matter:

* **It does not depend on expert size.** A bigger expert costs more to fetch
  and also does proportionally more work. They cancel.
* **It does not depend on GPU count.** Both compute and per-node CXL bandwidth
  scale with the deployment.

What is left is the hit rate, and it enters linearly. Getting from 90% to 99%
expert-fetch hit rate cuts the required batch size by 10x. That makes the hot
tier the entire design problem, and it makes real router skew the number the
whole project rests on.

## Why the obvious framing is wrong

Batching amortises the expert fetch across tokens — per-token CXL traffic falls
as roughly 1/B. That much is true and is what motivated the project. But the
reason it stops falling is the more important half: past a few hundred tokens a
batch touches *every* expert at *every* layer. At that point the HBM tier is no
longer a cache. It is a fixed partition, and every cold expert is refetched on
every single batch. Temporal locality is gone; only the popularity split
matters, which is why a static placement decided once at load time beats LRU
and LFU at serving batch sizes.

## Layout

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
  config.py     YAML loading (utf-8-sig; see "Windows" below)
configs/        model / memory / gpu YAML
scripts/
  run_all.py    every table and figure in results/
  frontier.py   binary search for the maximum offloadable fraction
tests/          36 tests
results/        tables/*.csv, figures/*.png, REPORT.md
paper/          LaTeX source
```

## Running it

```bash
pip install numpy pandas pyyaml matplotlib tabulate pytest
python -m pytest tests -q
python scripts/run_all.py       # ~6 min, writes results/
python scripts/frontier.py
```

## Capturing real routing traces

`memoe/hooks.py` attaches forward hooks to anything that looks like a router
(name matched against `gate`, `router`, `wg`, `switch`, `gating`) and whose
output width equals the expert count, so it does not need to know the model
class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from memoe.hooks import capture

mid = "allenai/OLMoE-1B-7B-0924-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype="bfloat16",
                                             device_map="cuda")
with capture(model, n_experts=64, top_k=8) as cap:
    for prompt in prompts:
        model(**tok(prompt, return_tensors="pt").to("cuda"))

cap.trace(model_name="OLMoE-1B-7B").save("results/olmoe_trace.npz")
```

Then swap it in anywhere a synthetic trace is used:

```python
from memoe import RoutingTrace
tr = RoutingTrace.load("results/olmoe_trace.npz")   # tr.is_real == True
```

Use a spread of input domains. Router skew is the assumption the whole project
leans on, and skew measured on one corpus will overstate what a mixed serving
workload sees.

## Honesty constraints we held to

These are deliberate and are enforced by tests, because they are the things a
reviewer will check first.

* CXL bandwidth is always the **measured sustained** 18–52 GB/s range, never
  the theoretical link rate. `test_cxl_bandwidth_within_measured_range`.
* CXL 3.0 pooling is pre-production. Any config using it is flagged
  `measured: false`, and every result carries a MODELED note.
  `test_pooled_cxl_is_flagged_as_modeled`.
* Extrapolated bandwidth points (>52 GB/s) are marked and drawn dotted.
* The footprint model reproduces published parameter counts for all four
  checkpoints before it is trusted for anything else.
* The compute model counts routed-expert GEMMs only. Attention and dense
  layers would hide more transfer, so the reported overheads are conservative.

## Windows note

PowerShell 5.1's `Set-Content -Encoding UTF8` writes a BOM, which turns the
first YAML key into `\ufeffname` and breaks parsing in a way that looks like a
missing field. All config reads use `read_text(encoding="utf-8-sig")`, and
`test_bom_prefixed_yaml_still_parses` feeds the loader a BOM-prefixed file so
this cannot regress.
