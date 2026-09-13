# MEMoE demo video: product script, filmed live on bhaskar (A10)

**Team King Bob · Nebula 2026 · target 10 minutes**

Every command below was run on bhaskar on 14 September 2026. Every number in the
narration came out of that run, or is read off the figure or committed file named
in the shot. Every figure description was checked against the rendered PNG.

- **LIVE** is filmed in a terminal on bhaskar.
- **CUTAWAY** is a figure from `results/` (also on GitHub), dropped in while editing.
- **DASHBOARD** is `results/dashboard.html` in a browser.

Setup and troubleshooting for every component: `docs/SETUP.md`.

---

## 0. What the judges must see: deliverable map

Put the deliverable name on screen as a caption when its first shot starts.

| # | Deliverable (from the brief) | Shots |
|---|---|---|
| 1 | HBM + CXL memory architecture model | 5, 6 |
| 2 | CXL-based Transformer & MoE workload analysis | 7, 8, 9, 22 |
| 3 | Bandwidth, latency & scalability evaluation | 6, 10, 23, 24 |
| 4 | Memory tiering & expert placement strategy | 8, 9, 10, 15 |
| 5 | CXL memory expansion & pooling study | 2, 17, 26, 27 |
| 6 | Simulation framework, HBM-only vs HBM+CXL | 21, 25 |
| 7 | Interactive demo / dashboard | 1, 3, 11 |
| 8 | Final report with architecture recommendations | 28 |

---

## 1. Before filming

### Windows on bhaskar

| Label | What |
|---|---|
| **T1** | terminal in `~/nebula/memoe-proj`, font 16pt or larger, 120 columns |
| **T2** | `watch -n0.5 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` |
| **T3** | QEMU console: `bash ~/cxlvm/run-cxl.sh` |
| **T4** | guest shell: `ssh -p 2222 ubuntu@localhost` (password `memoe`) |
| **B** | browser with `results/dashboard.html`, full screen |

For GPU shots put T1 on the left two thirds and T2 on the right third so memory is
visibly changing during the run. If you film from a laptop over ssh, copy
`results/dashboard.html` to the laptop; it needs no server.

### Pre-flight (about 10 minutes, run all of it once)

```bash
cd ~/nebula/memoe-proj
nvidia-smi                                 # only the 270 MiB background job on the card
export SYSTEMC_HOME=$HOME/local/systemc
for s in gpu tests dramsim systemc gem5; do scripts/demo_a10.sh $s; done
scripts/demo_a10.sh olmoe-check            # PASS, under a minute
scripts/demo_a10.sh deepseek-check         # PASS, about a minute
```

Demo runs write to `/tmp/memoe-demo`, so filming never changes committed results.
Boot the QEMU guest fresh for shot 26: a region created in an earlier boot makes
`create-region` fail.

### Editing notes

- Speed up checkpoint loading and VM boot 4 to 8x; never cut a result line.
- Caption each key number when it is spoken.
- Record screen first, narrate afterwards, one audio take per part.
- For every plot: say the axes, then what to look at, then what it means.

---

## 2. Shot list

### Part one · the problem · 0:00 to 0:45

**Shot 1 · DASHBOARD · top, "Can CXL memory hold the experts?" · 15s**
DeepSeek-V3 has six hundred and seventy-one billion parameters and uses about
thirty-seven billion of them per token. The rest still have to sit where the GPU
can reach them.

**Shot 2 · CUTAWAY · `results/figures/f8_gpu_reduction.png` · 15s**
Here is what that costs. GPUs needed just to hold the weights, for four real
models. Blue is HBM only; orange is the same GPUs with five hundred and twelve
gigabytes of CXL memory each. DeepSeek-V3 goes from sixteen GPUs to three. Qwen3
from six to one. HBM is fast, but its capacity is welded to the GPU.

**Shot 3 · DASHBOARD · scroll to "What we measured" · 15s**
MEMoE is our answer, in three parts: a planner that tells you how much of an MoE
model can live on CXL before you buy anything, a runtime that runs it that way,
and a verification stack that checks the answer four independent ways. Everything
you are about to see runs live on one machine with a twenty-three gigabyte A10.

### Part two · the machine and the tiers · 0:45 to 1:45

**Shot 4 · LIVE T1 · 10s**
```bash
scripts/demo_a10.sh gpu
```
One NVIDIA A10: twenty-three gigabytes, PCIe generation four. Remember
twenty-three.

**Shot 5 · CUTAWAY · architecture diagram from the paper (`paper/diagrams/arch.tex`) · 20s**
The architecture. Tier one is HBM on the GPU package: fastest, fixed size. Tier two
is a CXL Type-3 expander: DDR4 behind a link, expandable by adding cards and
poolable across servers through a CXL switch. Expert weights live on tier two and
are streamed into a staging buffer on tier one, one layer before they are needed.

**Shot 6 · LIVE T1 · 30s**
```bash
scripts/demo_a10.sh dramsim
```
What does that CXL tier actually deliver? This is DRAMSim3, a cycle-accurate DRAM
simulator, replaying real expert fetches. A DDR4-3200 channel sustains sixteen
point nine gigabytes per second at one hundred and seventy-nine nanoseconds loaded
latency, not the twenty-five point six on the label, because bulk reads never reach
nameplate. HBM2 in the same simulator: sixty-three gigabytes per second at
sixty-one nanoseconds. Every number downstream uses the measured figure, never the
spec sheet. Two simulations, six seconds.

### Part three · what MoE traffic looks like · 1:45 to 3:05

**Shot 7 · CUTAWAY · `results/figures/f2_batch_amortisation.png` · 25s**
Two panels, both against batch size. Left: CXL bytes you have to move per token
falls by four orders of magnitude as the batch grows, because one transfer serves
every token in the batch. Right, the catch: the share of experts a batch touches in
a layer. Every model passes ninety percent by two hundred and fifty-six tokens and
reaches one hundred percent by a thousand. At serving batch sizes there is no hot
set to cache. The GPU tier becomes a fixed partition, and the problem becomes
capacity and scheduling, not prediction.

**Shot 8 · CUTAWAY · `results/figures/f4_policy.png` · 20s**
Placement policy, per model, against batch size; vertical is the share of expert
reads served from HBM. The classic caches break: LFU slides down, and LRU drops to
zero once batches reach a few hundred tokens, because every batch evicts everything.
Static placement, red, stays flat at exactly the share of experts you keep in HBM.
So MEMoE uses a static partition and contains no ranking or eviction logic at all.

**Shot 9 · CUTAWAY · `results/figures_real/r3_penalty.png` · 20s**
And tuning that partition to a workload is dangerous. Real OLMoE routing, half the
experts resident. Rows are the workload a placement was profiled on, columns the
workload it serves, random is fifty. On the diagonal, seventy to eighty-four.
Profile on prose and serve code: thirty-eight, worse than random. The placement
strategy is a capacity-sized partition, never tuned to one workload.

**Shot 10 · CUTAWAY · `results/figures_extra/e3_latency.png` · 20s**
Last design input. Horizontal is CXL access latency, two hundred to eight hundred
nanoseconds; vertical is the share of transfer time spent on latency rather than
moving data, log scale; one line per transfer size. Fetch four-kilobyte pieces and
latency is most of the time. Fetch a whole expert and it stays under half a percent
at any latency. So MEMoE moves a whole layer of experts as one transfer.

### Part four · plan it before you buy it · 3:05 to 3:50

**Shot 11 · DASHBOARD · "The capacity model" calculator · 30s**
DO: pick DeepSeek-V3 in Model. Drag "Experts kept in HBM" down, then drag "CXL
bandwidth" from 16.88 up to 67.5.
This is the planner. The expert fetch hides behind compute once the batch passes a
critical size, B star. It depends on how many experts you keep in HBM, how much
stall you tolerate, and tier bandwidth, and it does not depend on expert size or
GPU count. Keep fewer experts in HBM and B star climbs. Give the tier more bandwidth
and it falls. You size the CXL deployment here, before spending anything.

**Shot 12 · CUTAWAY · `results/figures_runtime/rt2_batch_regime.png` · 20s**
And the planner is right. OLMoE on our Blackwell machine with seventy percent of
experts off the GPU. Horizontal is batch, the red band is decode batch sizes, the
green band prefill. At five hundred and twelve tokens we keep thirty-eight percent
of throughput; at sixteen thousand, ninety-four, with stall near zero. The formula
predicted the crossover at four thousand six hundred and twenty-six tokens; the
hardware crossed at four thousand six hundred and forty-five. Half a percent.

### Part five · run it: MEMoE-RT on OLMoE · 3:50 to 5:30

**Shot 13 · LIVE T1 + T2 · 25s (speed up loading)**
```bash
scripts/demo_a10.sh olmoe-check
```
Correctness before speed. This runs the untouched model, then our tiered model with
seventy-five percent of experts off the GPU, then with none off. Offloading changes
the output by exactly zero. Against the original implementation, the tiered model's
top token is in the reference top five ninety-nine point four percent of the time;
the remaining difference is bf16 rounding in our MoE layer, and it is identical
whether zero or all experts are offloaded. PASS.
**CUT** hold on PASS two seconds.

**Shot 14 · LIVE T1 + T2 · 35s**
```bash
scripts/demo_a10.sh olmoe
```
Now throughput at a prefill batch of sixteen thousand tokens. Fully resident:
thirteen thousand three hundred and ninety-one tokens per second in seventeen point
seven gigabytes. Now every expert of all sixteen layers sits in host memory and is
streamed over PCIe one layer ahead: twelve thousand two hundred and thirty-nine
tokens per second in six point four gigabytes. Ninety-one point four percent of the
throughput, two point seven five times less GPU memory, stall under a tenth of a
percent. Our PCIe 5 Blackwell machine keeps ninety-one point two percent: half the
link speed here, same answer, because at this batch the transfer hides behind
compute.
**CUT** hold on the "throughput relative to fully resident" block.

**Shot 15 · CUTAWAY · `results/figures_runtime/rt1_residency.png`, then `rt3_depth.png` · 30s**
Why the batch matters: the same experiment at four thousand tokens. Horizontal is
the share of experts off the GPU. Memory, blue, falls from fifteen gigabytes to four
point four; throughput, brown, falls to fifty-five percent. Same code as the live
run, one quarter of the batch, and the cost is five times larger. That is B star at
work.
Then lookahead depth: fetching one, two, four or eight layers ahead gives fourteen
to fifteen thousand tokens per second every time, while staging memory grows from
eight point one to eleven point five gigabytes. Our simulator had recommended four
to eight; hardware says one. We kept the correction.

### Part six · run what does not fit: DeepSeek-V2-Lite · 5:30 to 6:40

**Shot 16 · LIVE T1 · 15s (speed up loading)**
```bash
scripts/demo_a10.sh deepseek-check
```
A very different architecture: DeepSeek-V2-Lite, top-six of sixty-four experts,
shared experts, multi-head latent attention. Seventy-five percent of experts off the
GPU against one hundred percent off: max difference zero, argmax agreement one
hundred percent. PASS.

**Shot 17 · LIVE T1 + T2 · 35s**
```bash
scripts/demo_a10.sh deepseek
```
This model is thirty-one point four gigabytes of weights. The card is twenty-three.
It cannot run here at all without a second tier. With its whole twenty-eight point
eight gigabyte expert pool in host memory it runs in fifteen point two gigabytes,
at seven thousand and twenty-one tokens per second, stalling one hundredth of a
percent. On OLMoE we saved memory. Here we created a capability that did not exist.
**CUT** hold on `tokens_per_second` and `peak_vram_gb`.

**Shot 18 · CUTAWAY · `results/figures_runtime/rt5_deepseek_crossover.png` · 20s**
The same model swept across batch on this A10, all sixty-four experts per layer
off the GPU. Red dashed is time stalled on transfer, green is throughput. Up to four
thousand tokens nearly half the time is spent waiting. Through the shaded crossover
the stall collapses to zero, and throughput climbs from about seven hundred and fifty
tokens per second to over seven thousand. The same regime change as OLMoE, on a
different model and a slower link.

### Part seven · where it belongs in a serving stack · 6:40 to 7:30

**Shot 19 · CUTAWAY · `results/figures_serve/s1_decode_tpot.png` · 15s**
A serving request has two phases on opposite sides of B star. Time per output token
against concurrent requests: offloaded decode is flat around three hundred and
twenty-five milliseconds; resident decode rises from fifteen to fifty-two.

**Shot 20 · CUTAWAY · `results/figures_serve/s2_phase_cost.png` · 20s**
Same configuration, two phases. Offload adds seven percent to prefill, two point
four nine seconds to two point six seven, and five hundred and twenty-five percent
to decode, three point two eight to twenty point four nine. So the deployment rule:
slow tier on prefill workers, HBM on decode workers. Split that way we measured full
throughput with a third of the prefill-side GPU memory gone.

**Shot 21 · CUTAWAY · `results/figures/f3_crossover.png` · 20s**
The simulation framework's core comparison. One panel per model; horizontal is batch
size, vertical is stall as a percent of compute, log scale; each line is a share of
experts kept in HBM, and the flat bottom line, one hundred percent, is HBM-only.
The dashed line is ten percent overhead. The HBM plus CXL splits drop under it past
a critical batch, and the less you keep in HBM, the larger that batch.

### Part eight · verify it four independent ways · 7:30 to 9:20

**Shot 22 · CUTAWAY · `results/figures_extra/e1_roofline.png` · 20s**
First, what must not move. Horizontal is batch size, vertical is arithmetic
intensity, compute per byte moved, both log scale. The dotted lines are the balance
points of HBM, four-channel CXL and single-channel CXL: above a line, compute hides
the transfer. Expert intensity, the solid lines, rises with batch and crosses them.
KV intensity, dashed at the bottom, is fixed by the attention layout and never
reaches even HBM's line, because the cache is read every decode step. Experts move;
KV stays in HBM.

**Shot 23 · LIVE T1 · 30s**
```bash
scripts/demo_a10.sh gem5
```
Check one, gem5: one linear read generator into one, two, four and eight DDR4-2400
channels. One channel, thirteen point eight six gigabytes per second, seven percent
from DRAMSim3's figure for the same grade: two simulators, different memory models,
same answer. Two channels two point zero zero one times, four channels three point
nine eight times: linear, which is what licenses our four-channel CXL figure. At
eight the shared bus cannot drain responses and gem5 refuses. Scaling is not free
forever.

**Shot 24 · CUTAWAY · `results/figures/f7_bandwidth.png` · 15s**
Why bandwidth is the number to specify. Stall overhead against CXL sustained
bandwidth, at ninety-five percent resident and a two-thousand-token batch. The grey
band is the range measured on real CXL devices; dotted is extrapolated. Double the
bandwidth and the stall roughly halves, with no threshold where it stops mattering.

**Shot 25 · LIVE T1 · 25s**
```bash
scripts/demo_a10.sh systemc
```
Check two, a SystemC transaction-level model of the data path. It shares no code
with the runtime or the formula and is never told where the crossover is. At a batch
of four thousand six hundred and twenty-six it reports stall of thirteen point eight
one milliseconds, exactly one layer of compute, and B star of four thousand six
hundred and twenty-five. A discrete-event simulation landing on the closed form from
the other direction.
**CUT** hold three seconds on the CSV line.

**Shot 26 · LIVE T3 then T4 · 40s**
T3: `bash ~/cxlvm/run-cxl.sh` (speed up the boot). T4:
```bash
uname -r
sudo cxl list -M
sudo cxl create-region -m -t ram -d decoder0.0 -w 1 mem0
cat /sys/bus/cxl/devices/region0/commit
sudo dmesg | grep -i "bypassing cpu_cache"
```
Check three, the software path. QEMU emulating a CXL Type-3 memory expander, booted
on a Linux 6.8 kernel we built without root. The two-gigabyte device enumerates. We
create a region and the kernel commits it: commit equals one. On a stock kernel this
fails inside every virtual machine on a CPU cache check; our kernel enables the
documented bypass, and dmesg shows it. Two things we do not claim: the region is not
onlined as system memory, and QEMU has no timing model, so nothing here is a
performance number.

**Shot 27 · CUTAWAY · `results/figures_extra/e2_pooling.png` · 20s**
And pooling. Horizontal is how many servers share one CXL pool; vertical is the
critical batch B star, log scale. Every doubling of servers doubles B star:
DeepSeek-V3 goes from under two hundred thousand tokens with a private pool to
nearly three million shared by sixteen. Pooling buys capacity with batch size one
for one, so two to four servers is the useful region. These figures are modelled,
because no CXL 3.0 switch ships yet.

### Part nine · recommendations · 9:20 to 10:00

**Shot 28 · CUTAWAY · paper, Recommendations section · 30s**
What we would tell you to build. One: experts on CXL, active KV in HBM. Two: expert
offload is a prefill technique. Three: no popularity tiering engine; choose the
partition by capacity. Four: move whole layers, one layer ahead. Five: specify
expanders by sustained bandwidth, not DIMM rating. Six: pool across a few servers,
not many. And size all of it with B star before you buy the hardware.

**Shot 29 · LIVE T1 · 10s**
```bash
scripts/demo_a10.sh tests
```
Thirty-six tests in one command, and `scripts/reproduce.sh` regenerates every table
and figure from a clean checkout on a CPU. Setup for everything you have seen is in
the repository.
**CUT** on "36 passed". Black. No music.

---

## 3. If a live shot fails

Do not re-record narration; cut to the committed file and keep reading.

| Shot | Fails | Show instead |
|---|---|---|
| 6 | dramsim | `results/dramsim/calibration.json` |
| 13 | olmoe-check | `docs/SETUP.md`, section 2 expected output |
| 14 | olmoe | `results/runtime_bench.csv` (Blackwell) |
| 16, 17 | deepseek | `results/runtime_deepseek_all.csv` |
| 23 | gem5 | `results/gem5/ch*/stats.txt`: `simSeconds`, `bytesRead` |
| 25 | systemc | `results/tables/15_systemc_sweep.csv`, batch 4626 row |
| 26 | QEMU | `results/qemu/cxl_committed.txt` |

## 4. Six-minute cut

Keep 1, 2, 3, 4, 6, 7, 9, 11, 13, 14, 17, 20, 23, 25, 26, 28. Never cut 14, 17, 25 or
26: the runtime, the model that does not fit, the independent model, and the CXL
stack.

## 5. Delivery

- 1080p H.264 mp4, under 500 MB, `MEMoE_KingBob_Nebula2026_demo.mp4`
- Upload unlisted, test the link in a private window, put it in the email body
- Email `nebula@asteralabs.com`, subject `Nebula – Final Submission`, with team name,
  college and all member names

## 6. Numbers that differ from the older scripts

`docs/demo-video-script*.md` quote Blackwell numbers for live shots and describe
several figures incorrectly (`r1_coverage`, `f4_policy`, `f7_bandwidth`,
`e1_roofline`, `e2_pooling`, `e3_latency`). Use this script. Live numbers here are
the A10 ones:

| Measurement | Blackwell (paper) | A10 (this video) |
|---|---|---|
| OLMoE full offload, batch 16,384 | 91.2%, 2.44x less memory | 91.4%, 2.75x less memory |
| OLMoE argmax vs reference | 92.8% | 89.45% (top-5 99.41%, offload diff exactly 0) |
| DeepSeek-V2-Lite, batch 16,384 | 7,241 tok/s, 15.1 GB | 7,021 tok/s, 15.2 GB |
| gem5, four channels | 3.999x | 3.978x |
