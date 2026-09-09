# demo video narration script

eleven and a half minutes. **[bracketed]** text is a screen direction, everything
else is read aloud word for word.

the shape: about two and a half minutes of setup, then eight minutes on what we
built and what it does. every graph gets explained — axes first, then what to look
at, then what it means. no code on screen, only runs and plots.

record the screen first, narrate over it afterwards.

---

## before you film

**run this on ramesh and check the result:**

```bash
python scripts/bench_runtime.py --trials 3
```

the residency sweep currently has an inversion in it: 15,062 tokens per second at
residency 0.4, and 15,342 at 0.3. throughput going *up* as more experts leave the
GPU. it's 1.9 percent and it's noise, but it's visible in `rt1_residency.png` and
it will be the first thing a careful viewer asks about. three trials with medians
either flattens it or tells you it's real.

if it's still there when you film, say so on camera. the line is written into
section 4 below. owning it costs four seconds and beats being caught by it.

**machines:**

| | GPU | memory | what it proves |
|---|---|---|---|
| **ramesh** | RTX PRO 4500 Blackwell, PCIe 5.0 | 32.6 GB | a model that fits, in 2.44x less memory |
| **bhaskar** | A10, PCIe 4.0 | 23.0 GB | a model that does not fit, running anyway |

DRAMSim3, gem5, SystemC and QEMU all run on bhaskar.

**also copy the gem5 output in, it's gitignored:**

```bash
mkdir -p results/gem5 && cp <your stats.txt> results/gem5/ && python scripts/parse_gem5.py
```

---

## 0:00 — the problem  ·  45s

**[SCREEN: dashboard top, "You buy sixteen GPUs to store a model that needs two".]**

DeepSeek-V3 has 671 billion parameters and uses about 37 billion on any given
token. The rest still have to sit where the GPU can reach them, so you buy sixteen
H100s to store a model whose arithmetic would fit on two.

Ninety-seven percent of that checkpoint is expert weights, idle almost all of the
time. The question we set out to answer is whether they can live on something
slower and cheaper, and what that costs you in throughput.

The short answer is that it depends entirely on batch size, and we can tell you
where the line is.

---

## 0:45 — three measurements that decided the design  ·  1m 45s

**[SCREEN: results/figures_real/r1_coverage.png, full screen.]**

Start with the obvious design, which is a cache. Keep popular experts on the GPU,
fetch the rest on demand.

This plot kills it. Horizontal axis is batch size in tokens, log scale. Vertical is
the fraction of all experts that a single batch touches at a single layer. One line
per workload — prose, math, code, chat.

Every line goes to one. By about a thousand tokens they're all at ninety-seven
percent or above. At serving batch sizes a batch touches essentially every expert
in every layer, so there is no working set to cache. The set is everything. That
turns the GPU tier from a cache into a fixed partition, and a partition is a
capacity decision, not a prediction problem.

**[SCREEN: results/figures/f4_policy.png.]**

Here's the same conclusion from the policy side. Four placement strategies —
static popularity, balanced static, LRU, sampled LFU — plotted against resident
fraction. They lie on top of each other. The hit rate you get is exactly the
fraction you keep resident, no matter which policy you use, and no matter how
skewed the routing is.

That's why MEMoE-RT has no ranking in it anywhere. Ranking would be code that
computes something and changes nothing.

**[SCREEN: results/figures_real/r2_overlap.png.]**

Could you at least tune the placement per workload? This is the overlap between
the hot expert sets of two domains. The bright diagonal is a control: split one
workload in half and the halves agree eighty-five percent of the time, so the
measurement is sound.

Now look off the diagonal. Prose against code is fifteen percent, where random
chance is twenty-five. Prose and code don't merely fail to share experts, they
avoid each other.

**[SCREEN: results/figures_real/r3_penalty.png.]**

Which produces this. Rows are the workload a placement was profiled on, columns
are what it was then asked to serve. On the diagonal, 0.756. Off it, 0.519, barely
above the 0.500 you'd get from picking at random. The worst cell is prose profiled,
code served: 0.383. Worse than random.

So: no cache, no predictor, no ranking, and never tune placement on a workload you
aren't going to serve. That's the design, and all three parts of it are deletions.

---

## 2:30 — what we built  ·  1m

**[SCREEN: dashboard, "How it actually works". drag the batch slider slowly from
lowest to highest, then back.]**

MEMoE-RT holds the non-resident experts in pinned host memory, laid out
contiguously by layer. When layer L is computing, the whole expert set for layer
L plus one is already in flight on a separate copy stream, landing in a ring of
staging buffers. Layer L uses slot L mod d plus one, so the addresses are fixed at
load time and never recomputed.

One transfer per layer, not one per expert. That matters more than it sounds like
it should, and there's a plot for it later.

Watch the cost as I move batch size. Small batch, offload is expensive. Large
batch, nearly free. Same configuration, same hardware, same code. Only the batch
changed.

**[SCREEN: dashboard, "The line that decides everything".]**

That crossover has a closed form. It depends on tier bandwidth, expert count, and
how many experts each token activates. It does not depend on expert size, because
a bigger expert costs proportionally more to move and does proportionally more work
once it lands. And it does not depend on GPU count, because compute and per-node
bandwidth scale together.

Calibrated only against the fully resident baseline, it predicts a crossover at
4,626 tokens.

---

## 3:30 — it runs, on hardware  ·  2m 30s

**[SCREEN: ramesh. run: nvidia-smi --query-gpu=name,memory.total --format=csv]**

A Blackwell card, thirty-two gigabytes. OLMoE fits on it comfortably, which is
exactly why it's the right model for measuring what offload costs — there's a real
baseline to compare against.

**[SCREEN: run: python scripts/bench_runtime.py --check]**

Correctness first. This runs the tiered model and the untouched reference on
identical input and compares the outputs. If this doesn't pass, no timing number
underneath it means anything.

**[SCREEN: run: python scripts/bench_runtime.py --batch 16384 --residency 0.0
--depth 1 — keep the GPU memory pane in frame.]**

Now every expert in every layer is outside GPU memory. Ninety-three percent of the
model's parameters are across a PCIe link, streamed in one layer ahead of use.

Twenty-four thousand six hundred eighty-four tokens per second, against a resident
baseline of twenty-six thousand eight hundred seventy-three. Ninety-two percent of
the throughput for forty-one percent of the memory. Time stalled waiting on a
transfer: under one percent.

**[SCREEN: results/figures_runtime/rt1_residency.png.]**

Here's the whole sweep. Horizontal axis is the fraction of experts kept on the GPU,
running from fully resident on the left to fully offloaded on the right. Two
vertical axes: throughput, and peak GPU memory.

Memory falls cleanly and predictably — that's just arithmetic, you're removing
weights. Throughput falls much more slowly, and that gap between the two curves is
the entire result. You give up a little speed for a lot of capacity.

**[if the inversion is still present, say this:]** You'll notice a small kink
around thirty and forty percent, where throughput goes slightly up as we offload
more. That's about two percent, and it's run-to-run variance rather than a real
effect — we report medians across three trials and the two points sit inside each
other's spread.

**[SCREEN: results/figures_runtime/rt2_batch_regime.png.]**

Now the same thing against batch size, with offload fixed at seventy percent.
Horizontal axis is batch, log scale. The rising line is throughput retained against
the matched baseline; the falling line is the fraction of time stalled on transfer.

At 512 tokens we keep 38 percent — offload is a disaster there. At 16,384 we keep
94 percent and stall for less than a tenth of a percent. Same configuration at both
ends. The dotted vertical is where the closed form said the crossover would be, and
the measured curve crosses at 4,645 against a prediction of 4,626. Half a percent,
on a quantity the model was never fitted to.

**[SCREEN: results/figures_runtime/rt3_depth.png.]**

One result that corrected us. This is prefetch lookahead depth against throughput
and staging memory. Our trace simulator had recommended fetching four to eight
layers ahead. On hardware, depth one is enough: throughput is flat across the
range, while staging memory grows forty-one percent by depth eight.

The reason is the design change. Once you move a whole layer as one transfer
instead of chasing individual experts, one layer of lookahead is all the pipeline
can absorb. We report the correction rather than the original recommendation.

**[SCREEN: results/figures_runtime/rt4_overlap.png.]**

And this is the honesty check on our own model. The analytical version assumes
transfer hides perfectly under compute. Measured, about sixty percent hides and
forty percent is exposed. The fitted line is what we use everywhere else, so the
model is calibrated against reality rather than against an assumption.

---

## 6:00 — the model that does not fit  ·  1m 30s

**[SCREEN: bhaskar. run: nvidia-smi --query-gpu=name,memory.total --format=csv]**

Different machine. An A10, twenty-three gigabytes, on a slower PCIe generation.

**[SCREEN: run: scripts/run_deepseek.sh --batch 16384]**

DeepSeek-V2-Lite is 31.4 gigabytes of weights. There is no baseline to compare
against here, because the model does not fit on this card. It cannot run at all.

On the last machine we saved memory. On this one we bought a capability that
didn't exist. It runs in 15.1 gigabytes at 7,241 tokens per second.

And this is a genuinely different architecture, not a rerun. Different top-k,
shared experts alongside the routed ones, a dense first layer, and multi-head
latent attention instead of ordinary attention. Our runtime handles it through the
same tiering path with no architecture-specific transfer logic.

**[SCREEN: results/figures_runtime/rt5_deepseek_crossover.png.]**

Same plot as before, different model, different machine, different link. Batch on
the horizontal, cost of offload on the vertical. The curve has the same shape and
the tipping point lands where the same formula puts it.

That's the part worth pausing on. The formula was derived from one model on one
machine, and it transferred to a different model on different hardware without
being refitted.

**[SCREEN: results/figures_runtime/rt6_deepseek_link.png.]**

And here's the sensitivity to link speed on that machine. Slower link, higher
required batch, in exactly the inverse proportion the closed form predicts.

---

## 7:30 — where it belongs in a real server  ·  1m 15s

**[SCREEN: dashboard, "And it fits how servers actually work", the TPOT plot.]**

A serving request has two halves and they land on opposite sides of that line.

This is time per output token against concurrency. The offloaded line is flat at
325 milliseconds while concurrency grows eightfold. The resident line climbs. A
flat line is what a bandwidth-bound cost looks like once it's already been paid —
more concurrent work rides along on the same transfer.

**[SCREEN: dashboard, "Put it in the right place and it is free".]**

Which gives a deployment rule. Offload on a prefill worker costs seven percent.
The same configuration on a decode worker costs six hundred. Same code, same
hardware — placement is the whole difference. Prefill-decode disaggregation is
already where serving stacks are heading, and this says exactly which side the
slow tier belongs on.

**[SCREEN: results/figures_extra/e1_roofline.png.]**

The same arithmetic tells you what may not move. This is arithmetic intensity
against the balance point of the tier. Expert weights sit above the line, so they
can be streamed. The KV cache sits hundreds of times below it, because it's read
every single step. One long Qwen3 sequence reads nearly six gigabytes per decode
step: 1.9 milliseconds from HBM, 93 from a CXL tier. Experts move. KV stays.

---

## 8:45 — four checks that don't share code  ·  2m

**[SCREEN: results/figures/f7_bandwidth.png.]**

All of this rests on one bandwidth number, so it's worth showing how hard we
pushed on it. This is required batch size against tier bandwidth. It's an inverse
curve — halve the bandwidth, double the batch you need. The solid section is the
measured range and the dotted section is extrapolated, marked so nobody mistakes
one for the other.

**[SCREEN: dashboard, gem5 comparison table.]**

We measured the tier twice with different tools. DRAMSim3 says 14.96 gigabytes per
second on a DDR4-2400 channel. gem5, independently, says 13.86. Seven percent
apart, and neither reaches the 19.2 the DIMM is sold as. Two simulators with
different memory models landing in the same place is much harder to argue with than
either alone.

**[SCREEN: bhaskar. run: cd systemc && ./memoe_tlm --batch 4626 --residency 0.3]**

Then a third check, because the formula was fitted against the runtime and those
two aren't independent witnesses. This is a SystemC TLM model of the same data
path, sharing no code with either. It contains no formula and was never told where
the crossover is.

At batch 4,625 it puts one layer's transfer at 13.814 milliseconds and one layer's
compute at 13.813. **[SCREEN: hold on that output, three full seconds.]** That
equality is precisely what the closed form asserts, reached from the other
direction by a discrete-event simulation.

**[SCREEN: results/figures_extra/e5_systemc_crossover.png.]**

Its full sweep, for completeness. Stall against batch on the left, throughput on
the right, with the predicted crossover marked. Above that line the leftover stall
is constant at 13.81 milliseconds — exactly one layer transfer, which is the first
layer, the one nothing precedes and nothing can prefetch. Steady-state stall above
the crossover is zero.

It also independently reproduced the depth result: one, two, four and eight layers
of lookahead give identical throughput while staging memory grows from 1.1 to 5.1
gigabytes. Our simulator was wrong about that, the hardware corrected it, and this
third model agrees with the hardware.

**[SCREEN: qemu guest. run: cxl list -M, then cat
/sys/bus/cxl/devices/region0/commit, then dmesg | grep -i "bypassing cpu_cache"]**

Fourth, the software path. We don't have CXL silicon, so we exercised the Linux CXL
stack in QEMU. It enumerates, the drivers bind, the decoders program. Region commit
fails on a stock kernel because it's gated on a cache invalidation check that
returns false inside any virtual machine. We built a 6.8 kernel with the documented
bypass, no root anywhere on the host, and it commits — two gigabytes, and the
kernel logs which path it took.

Two things we are not claiming. We committed that region, we did not map it and
push traffic through it. And QEMU has no timing model, so nothing there is a
performance number. Every quantitative CXL figure in this project came from
DRAMSim3 and the analytical model.

---

## 10:45 — expansion, pooling, and one practical detail  ·  45s

**[SCREEN: results/figures_extra/e2_pooling.png.]**

Two things a CXL tier gives you that a bigger GPU doesn't. Expansion is what
everything so far measured. Pooling is this plot: nodes sharing one memory pool on
the horizontal, capacity saved and required batch size on the vertical.

Both scale linearly, which means the trade is exactly one for one. Sixteen nodes
save you six point three terabytes and demand a batch of one and a half million
tokens, which is not an operating point. Two to four nodes is the useful region.

**[SCREEN: results/figures_extra/e3_latency.png.]**

And the detail that shaped the whole runtime. Transfer size on the horizontal, the
share of time spent waiting rather than moving data on the vertical. At a whole
twelve-megabyte expert, waiting is 0.16 percent. At four kilobytes, it's eighty-
three percent. Fetch experts whole. Never fetch fragments.

---

## 11:30 — close  ·  30s

**[SCREEN: ramesh. run: scripts/reproduce.sh — cut the moment "36 passed" appears.]**

One command, CPU only, no GPU and no downloads. Every table and every figure you've
seen regenerates byte-identical from a clean checkout.

**[SCREEN: dashboard, last card, "What we would tell you to build".]**

Expert offload is a prefill technique. At prefill batch sizes you can put an entire
expert pool on a tier fifty times slower than HBM and lose seven percent. At decode
batch sizes the same configuration takes most of your throughput. The line between
them is a formula you can evaluate before buying any hardware, and it held across
two model architectures, two machines, two link speeds, and a model that fit
alongside one that didn't.

**[SCREEN: hold two seconds. cut. no logo, no music, no sign-off.]**

---

## if a run fails while recording

don't re-record the narration, cut to the stored file and keep reading.

| fails | show instead |
|---|---|
| `bench_runtime.py --check` | `results/runtime_bench_all.csv` |
| the 16,384 run | `results/runtime_bench.csv` |
| `run_deepseek.sh` | `results/runtime_deepseek_all.csv` |
| `./memoe_tlm` | `results/tables/15_systemc_sweep.csv` |
| qemu guest | `results/qemu/cxl_committed.txt` |
| `reproduce.sh` | `results/REPORT.md` |

---

## reading notes

- every plot: axes, then what to look at, then what it means. never just read the
  title out
- pause a beat after each number
- lines to land hardest: "ninety-two percent of the throughput for forty-one
  percent of the memory", "it cannot run at all", and the 13.814 against 13.813
- say the two not-claiming lines flatly. in front of people who build CXL
  controllers, stating your limits is a strength
- if you're running long, cut the pooling and granularity section at 10:45 and the
  bandwidth plot at 8:45. never cut DeepSeek or the SystemC agreement
