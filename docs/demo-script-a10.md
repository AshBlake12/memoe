# MEMoE demo video script, filmed live on bhaskar (A10)

Team King Bob · Nebula 2026

Every live command below was run on bhaskar on 14 September 2026. The numbers in
the narration come from those runs, or from the figure or committed file named in
the shot. Each plot's description was checked against the rendered PNG.

- **LIVE** means filmed in a terminal on bhaskar.
- **CUTAWAY** means a figure from `results/` (also on GitHub), added while editing.
- **DASHBOARD** means `results/dashboard.html` open in a browser.

Setup and troubleshooting for every component are in `docs/SETUP.md`.

---

## 0. Deliverable map

When the first shot for a deliverable starts, show its name as a caption.

| # | Deliverable (from the brief) | Shots |
|---|---|---|
| 1 | HBM + CXL memory architecture model | 5, 6 |
| 2 | CXL-based Transformer & MoE workload analysis | 7, 8, 9, 19 |
| 3 | Bandwidth, latency & scalability evaluation | 5, 6, 20 |
| 4 | Memory tiering & expert placement strategy | 8, 9, 14 |
| 5 | CXL memory expansion & pooling study | 2, 16, 23 |
| 6 | Simulation framework, HBM-only vs HBM+CXL | 11, 21 |
| 7 | Interactive demo / dashboard | 1, 3, 10 |
| 8 | Final report with architecture recommendations | 24 |

---

## 1. Before filming

### Windows on bhaskar

| Label | What |
|---|---|
| T1 | terminal in `~/nebula/memoe-proj`, font 16pt or larger, 120 columns |
| T2 | `watch -n0.5 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` |
| T3 | QEMU console: `bash ~/cxlvm/run-cxl.sh` |
| T4 | guest shell: `ssh -p 2222 ubuntu@localhost` (password `memoe`) |
| B | browser with `results/dashboard.html`, full screen |

For the GPU shots, put T1 on the left two thirds of the screen and T2 on the right
third, so viewers can see GPU memory change during the run. If you film from a
laptop over ssh, copy `results/dashboard.html` to the laptop. It doesn't need a
server.

### Pre-flight (about 10 minutes, run all of it once)

```bash
cd ~/nebula/memoe-proj
nvidia-smi                                 # only the 270 MiB background job on the card
export SYSTEMC_HOME=$HOME/local/systemc
for s in gpu tests dramsim systemc gem5; do scripts/demo_a10.sh $s; done
scripts/demo_a10.sh olmoe-check            # PASS, under a minute
scripts/demo_a10.sh deepseek-check         # PASS, about a minute
```

Demo runs write to `/tmp/memoe-demo`, so filming never changes the committed
results. Boot the QEMU guest fresh before shot 22. If a region was created in an
earlier boot, `create-region` fails.

GPU throughput moves by 1 to 2 percent between runs. The narration uses the 03:05
run (13,391 and 12,239 tok/s, 91.4%). A rerun at 03:34 gave 13,332 and 12,042 tok/s
(90.3%). If your take prints different numbers, read the ones on screen. Memory and
stall figures stay the same.

### Editing notes

- Speed up checkpoint loading and VM boot 4 to 8x, but never cut a result line.
- Caption each key number as it is spoken.
- Record the screen first and narrate afterwards, one audio take per part.
- In shots with more than one figure, the switch point is marked `[next: file]`.

### Reading notes

- Numbers are written as digits. Read "GB/s" as "gigabytes per second", "B*" as
  "B star", and "ε" as "epsilon".
- Pause for a beat after each headline number.
- Read the two "what we don't claim" lines in shot 22 plainly. The judges build CXL
  hardware, so stating limits helps.

---

## 2. Shot list

Timings assume about 150 spoken words a minute. The full video runs about
13 minutes.

### Part one · the problem · 0:00 to 1:15

**Shot 1 · DASHBOARD · top, "Can CXL memory hold the experts?" · 20s**

A Mixture-of-Experts model, or MoE, splits each layer into many small networks
called experts, and a router sends each token to only a few of them. DeepSeek-V3 has
671 billion parameters, but each token uses about 37 billion. The rest still have to
sit where the GPU can reach them.

**Shot 2 · CUTAWAY · `results/figures/f8_gpu_reduction.png` · 30s**

This chart counts the GPUs needed just to hold the weights of four real models. Blue
gives each GPU its own 80 GB of HBM, the high-bandwidth memory packaged with the
chip. Orange adds 512 GB of CXL memory per GPU. DeepSeek-V3 goes from 16 GPUs to 3,
and Qwen3 from 6 to 1. Experts make up 93 to 97 percent of these models, and the
only way to get more HBM is to buy more GPUs.

**Shot 3 · DASHBOARD · scroll to "What we measured" · 25s**

We're Team King Bob, and this is MEMoE. We wanted to know whether expert weights can
live on CXL memory, and what that costs in throughput. We built a planner that sizes
the CXL tier before you buy it, a runtime called MEMoE-RT that runs models this way,
and four independent checks. The short answer is yes during prefill, and no during
decode.

### Part two · the machine and the memory tiers · 1:15 to 3:00

**Shot 4 · LIVE T1 · 20s**

```bash
scripts/demo_a10.sh gpu
```

Everything live in this video runs on one NVIDIA A10 with 23 GB of memory on PCIe
Gen 4. Remember that 23. A few plots come from our second machine, a 32 GB Blackwell
card on PCIe Gen 5, and we'll say when they do.

**Shot 5 · CUTAWAY · architecture diagram (`paper/diagrams/arch.tex`), then `results/figures_extra/e3_latency.png` · 60s**

Tier one is HBM on the GPU, which is the fastest memory and fixed in size. Tier two
is a CXL Type-3 expander: ordinary DDR memory behind a CXL link. You grow it by
adding cards, and a CXL switch lets several servers share one pool.

Expert weights live on tier two, packed into one continuous block per layer. While
the GPU computes layer L, a separate copy stream is already sending all of layer L
plus 1's experts into staging buffers on the GPU. That makes one transfer per layer
instead of one per expert. [next: `e3_latency.png`] Size matters because latency
dominates small transfers, but for a whole expert it's under half a percent of the
transfer time.

We don't have CXL hardware, so in the live runs host memory over PCIe stands in for
it. The latency is different, but the problem has the same shape.

**Shot 6 · LIVE T1 · 25s**

```bash
scripts/demo_a10.sh dramsim
```

How fast is a CXL tier in practice? Instead of trusting a datasheet, we simulated the
memory. DRAMSim3 is a cycle-accurate DRAM simulator, and here it replays real 12 MB
expert fetches. A DDR4-3200 channel is rated at 25.6 GB/s, but under this load it
sustains 16.9, at 179 nanoseconds per read. HBM2 reaches 63 GB/s at 61 nanoseconds.
Every calculation after this uses the measured 16.9.

### Part three · what MoE traffic looks like · 3:00 to 4:55

**Shot 7 · CUTAWAY · `results/figures/f2_batch_amortisation.png` · 45s**

Two terms first. Prefill is when the model reads the prompt, so thousands of tokens
go through in one batch. Decode is when it writes the reply, one token per request
per step, so batches are much smaller.

Both panels have batch size along the bottom. On the left, the expert bytes moved per
token fall roughly as one over the batch, since one transfer serves every token. On
the right is the catch: the share of a layer's experts one batch touches passes 90
percent by 256 tokens and reaches 100 by about a thousand. At serving batch sizes
every expert is in use, so there's nothing to cache. The GPU tier becomes a fixed
split.

**Shot 8 · CUTAWAY · `results/figures/f4_policy.png` · 30s**

So which policy should pick what stays in HBM? Up the side is the share of expert
reads served from HBM. LFU slides down, and LRU drops to zero past a few hundred
tokens because each batch evicts what the last one cached. Static placement, in red,
stays flat at exactly the share you keep in HBM, at every routing skew we tested. So
MEMoE-RT has no ranking code at all. It keeps the lowest-numbered experts on the GPU.

**Shot 9 · CUTAWAY · `results/figures_real/r2_overlap.png`, then `r3_penalty.png` · 40s**

Could you pick different experts for different workloads? We captured about 259,000
tokens of real OLMoE routing across prose, math, code and chat, and compared each
domain's most-used quarter of experts. Chance overlap is 25 percent. Prose and code
share only 15.

[next: `r3_penalty.png`] Here's what that does to a tuned placement. Rows are the
workload it was profiled on, columns the one it serves, and random placement scores
50. Matching workloads get 70 to 84. Profile on prose and serve code, and you get 38,
which is worse than random. So we size the split by capacity, never by workload.

### Part four · plan it before you buy it · 4:55 to 6:10

**Shot 10 · DASHBOARD · "The capacity model" calculator · 40s**

DO: choose DeepSeek-V3 under Model. Drag "Experts kept in HBM" down, then drag "CXL
bandwidth" from 16.88 up to 67.5.

This is the planner. A transfer only costs you if the GPU has to wait for it. With a
big enough batch, computing a layer takes longer than fetching the next layer's
experts, so the wait disappears. We call that batch size B*.

B* grows with E, the experts per layer, and F, the GPU's compute. It shrinks with
bandwidth, with k, the experts each token uses, and with ε, the stall you'll accept.
Expert size cancels out, because a bigger expert is slower to move but gives the GPU
more work. Keep fewer experts in HBM and B* climbs. Add bandwidth and it falls.

**Shot 11 · CUTAWAY · `results/figures_runtime/rt2_batch_regime.png`, then `rt4_overlap.png`, then `results/figures/f3_crossover.png` · 35s**

Does hardware agree? This is OLMoE on the Blackwell machine with 70 percent of
experts off the GPU. At 512 tokens it keeps 38 percent of its throughput, and at
about 16,000 it keeps 94. The formula predicted the crossover at 4,626 tokens, and
the hardware crossed at 4,645.

[next: `rt4_overlap.png`] One correction: the formula assumes transfers hide
completely behind compute, but about 40 percent of transfer time is still exposed.
[next: `f3_crossover.png`] Our simulator finds the same crossover for all four
models, against HBM only on the flat bottom line.

### Part five · running it: MEMoE-RT on OLMoE · 6:10 to 7:50

**Shot 12 · LIVE T1 + T2 · 35s (speed up loading)**

```bash
scripts/demo_a10.sh olmoe-check
```

Correctness comes first. OLMoE-1B-7B has 16 layers of 64 experts, and each token uses
8. This runs the original model, then our tiered version with 75 percent of experts
off the GPU and with none off. The two tiered runs match exactly, so offloading
changes nothing. Against the original, our top token is in its top five 99.4 percent
of the time. The 89.5 percent exact match is bf16 rounding in our MoE layer, and it's
the same at every offload level. PASS.

**CUT** Hold on PASS for two seconds.

**Shot 13 · LIVE T1 + T2 · 35s**

```bash
scripts/demo_a10.sh olmoe
```

Now speed, at a prefill batch of 16,384 tokens. With the whole model on the GPU we
get 13,391 tokens per second in 17.7 GB. Next, every expert moves to host memory and
streams over PCIe one layer ahead. Watch the memory column: 12,239 tokens per second
in 6.4 GB. That's 91.4 percent of the throughput with 2.75 times less GPU memory, and
the GPU waits under a tenth of a percent of the time. The Blackwell machine, with
twice the link speed, keeps 91.2 percent.

**CUT** Hold on the "throughput relative to fully resident" block.

**Shot 14 · CUTAWAY · `results/figures_runtime/rt1_residency.png`, then `rt3_depth.png` · 30s**

Batch size is what makes that work. At 4,096 tokens on the Blackwell machine, full
offload cuts memory from 15 GB to 4.4 but keeps only 55 percent of throughput.

[next: `rt3_depth.png`] How far ahead should you fetch? One, two, four and eight
layers all give 14 to 15 thousand tokens per second, while staging memory grows from
8.1 to 11.5 GB. Our simulator had said four to eight. The hardware says one, so we
use one.

### Part six · a model that doesn't fit: DeepSeek-V2-Lite · 7:50 to 8:55

**Shot 15 · LIVE T1 · 20s (speed up loading)**

```bash
scripts/demo_a10.sh deepseek-check
```

Now a very different model, DeepSeek-V2-Lite. Each token uses 6 of 64 routed experts
plus 2 shared ones, its first layer is dense, and it uses multi-head latent
attention. Only the routed experts move. Same check as before: the maximum difference
is zero, and top tokens agree 100 percent of the time. PASS.

**Shot 16 · LIVE T1 + T2 · 25s**

```bash
scripts/demo_a10.sh deepseek
```

This model has 31.4 GB of weights, and the card has 23, so on its own it can't run
here. With its 28.8 GB of experts in host memory, it runs in 15.2 GB at 7,021 tokens
per second, and the GPU waits a hundredth of a percent of the time. With OLMoE we
saved memory. Here we ran a model this card couldn't run at all.

**CUT** Hold on `tokens_per_second` and `peak_vram_gb`.

**Shot 17 · CUTAWAY · `results/figures_runtime/rt5_deepseek_crossover.png` · 20s**

Here's that model across batch sizes on the A10. Red dashed is time stalled, green is
throughput. Below 4,000 tokens nearly half the time is spent waiting. Past the shaded
crossover the stall drops to zero, and throughput climbs from about 750 to over 7,000
tokens per second.

### Part seven · where it fits in a serving stack · 8:55 to 9:30

**Shot 18 · CUTAWAY · `results/figures_serve/s1_decode_tpot.png`, then `s2_phase_cost.png` · 35s**

A real request has two phases, on opposite sides of B*. We built a serving loop with a
real KV cache and continuous batching on the Blackwell machine. In decode, offloaded
time per output token stays flat near 325 milliseconds, while resident rises from 15
to 52.

[next: `s2_phase_cost.png`] By phase, offloading adds 7 percent to prefill and 525
percent to decode. So put the slow tier on prefill workers and keep decode on HBM.
With that split we kept full throughput with a third of prefill-side GPU memory freed.

### Part eight · four independent checks · 9:30 to 11:50

**Shot 19 · CUTAWAY · `results/figures_extra/e1_roofline.png` · 30s**

What shouldn't move? The KV cache stores attention keys and values for earlier
tokens, and decode reads it every step. This plot shows compute per byte moved
against batch size. Above a dotted line, compute hides the transfer. Experts, the
solid lines, rise with batch and cross those lines. The KV cache, dashed at the
bottom, never reaches even HBM's line. So experts move, and the KV cache stays in HBM.

**Shot 20 · LIVE T1, then `results/figures/f7_bandwidth.png` · 25s**

```bash
scripts/demo_a10.sh gem5
```

Now the checks. DRAMSim3 was the first. The second is gem5, which has its own memory
model. One DDR4-2400 channel gives 13.86 GB/s, within 7 percent of DRAMSim3. Two
channels give 2.001 times that and four give 3.978, which is linear. At eight, the
shared bus can't return responses fast enough and gem5 stops. [next: `f7_bandwidth.png`]
That's why we specify bandwidth: stall falls in proportion to it.

**Shot 21 · LIVE T1 · 25s**

```bash
scripts/demo_a10.sh systemc
```

The third check is SystemC, a transaction-level model of the data path. It shares no
code with the runtime or the formula, and it isn't told where the crossover is. At
4,626 tokens it reports a stall of 13.81 milliseconds, exactly one layer of compute,
and a B* of 4,625. A separate simulation lands on the formula's number.

**CUT** Hold on the CSV line for three seconds.

**Shot 22 · LIVE T3 then T4 · 40s**

T3: `bash ~/cxlvm/run-cxl.sh` (speed up the boot). T4:

```bash
uname -r
sudo cxl list -M
sudo cxl create-region -m -t ram -d decoder0.0 -w 1 mem0
cat /sys/bus/cxl/devices/region0/commit
sudo dmesg | grep -i "bypassing cpu_cache"
```

The fourth check is the Linux CXL software stack. QEMU emulates a CXL Type-3
expander, and the guest runs a Linux 6.8 kernel we built without root. The 2 GB
device shows up, we create a region, and the kernel commits it. On a stock kernel
this fails in any virtual machine, because of a CPU cache check. Our kernel enables
Linux's built-in testing bypass, and dmesg shows it.

Two things we don't claim. We didn't send traffic through the region. And QEMU has no
timing model, so none of this is a performance number.

**Shot 23 · CUTAWAY · `results/figures_extra/e2_pooling.png` · 20s**

Last, pooling, where several servers share one CXL pool. Each time the number of
servers doubles, B* doubles. DeepSeek-V3 goes from under 200,000 tokens with its own
pool to nearly 3 million shared across 16. So 2 to 4 servers is the useful range.
This part is modelled, because no CXL 3.0 switch ships yet.

### Part nine · recommendations · 11:50 to 12:35

**Shot 24 · CUTAWAY · paper, Recommendations section · 30s**

Here's what we'd tell you to build. One: experts on CXL, active KV cache in HBM. Two:
offload during prefill, not decode. Three: size the HBM share by capacity, not
popularity. Four: move whole layers, one layer ahead. Five: specify expanders by
sustained bandwidth, not DIMM rating. Six: assume 40 percent of transfer time is
exposed. Seven: pool across a few servers, not many. And calculate B* for your models
before you buy hardware.

**Shot 25 · LIVE T1 · 15s**

```bash
scripts/demo_a10.sh tests
```

All 36 tests pass with one command, and `scripts/reproduce.sh` rebuilds every table
and analysis figure from a clean checkout on a CPU. Setup for everything in this video
is in the repository.

**CUT** On "36 passed". Cut to black. No music.

---

## 3. If a live shot fails

Don't re-record the narration. Cut to the committed file and keep reading.

| Shot | Fails | Show instead |
|---|---|---|
| 6 | dramsim | `results/dramsim/calibration.json` |
| 12 | olmoe-check | `docs/SETUP.md`, section 2 expected output |
| 13 | olmoe | `results/runtime_bench.csv` (Blackwell) |
| 15, 16 | deepseek | `results/runtime_deepseek_all.csv` |
| 20 | gem5 | `results/gem5/ch*/stats.txt`: `simSeconds`, `bytesRead` |
| 21 | systemc | `results/tables/15_systemc_sweep.csv`, batch 4626 row |
| 22 | QEMU | `results/qemu/cxl_committed.txt` |

## 4. Shorter cut (about 9 minutes)

Keep shots 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 16, 18, 20, 21, 22 and 24. Never cut
13, 16, 21 or 22. Those are the runtime, the model that doesn't fit, the independent
model and the CXL software stack.

## 5. Delivery

- 1080p H.264 mp4, under 500 MB, named `MEMoE_KingBob_Nebula2026_demo.mp4`
- Upload it unlisted, test the link in a private window, and put the link in the
  email body
- Email `nebula@asteralabs.com` with the subject `Nebula – Final Submission`.
  Include the team name, college and every member's name

## 6. Numbers that differ from the older scripts

`docs/demo-video-script*.md` quote Blackwell numbers for the live shots and describe
several figures incorrectly (`r1_coverage`, `f4_policy`, `f7_bandwidth`,
`e1_roofline`, `e2_pooling`, `e3_latency`). Use this script instead. The live
numbers here come from the A10:

| Measurement | Blackwell (paper) | A10 (this video) |
|---|---|---|
| OLMoE full offload, batch 16,384 | 91.2%, 2.44x less memory | 91.4%, 2.75x less memory |
| OLMoE argmax vs reference | 92.8% | 89.45% (top-5 99.41%, offload diff exactly 0) |
| DeepSeek-V2-Lite, batch 16,384 | 7,241 tok/s, 15.1 GB | 7,021 tok/s, 15.2 GB |
| gem5, four channels | 3.999x | 3.978x |
