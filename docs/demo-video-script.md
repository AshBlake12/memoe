# MEMoE demo video: full production script

Target runtime 8 minutes 15 seconds. Every number below is read from a file in
this repo, and the file is named in the shot so you can check it before filming.

Each shot has:

- **WHERE** which window is full screen
- **DO** the exact command, or the exact scroll position
- **SAY** read word for word, this is the narration
- **CUT** when to move on

Record the screen first with no audio, then narrate over the cut. Do not try to
talk and type at the same time.

---

## 1. Production setup

**Capture.** 1920x1080, 30 fps, OBS or the built-in recorder. One display only.
Terminal font at 16pt minimum. Hide the taskbar, close mail and chat.

**Audio.** Record narration separately in a quiet room, one take per section, not
one take for the whole thing. No music.

**Windows, referred to by label throughout.**

| Label | What | Where |
|---|---|---|
| **D** | `results/dashboard.html` in a browser, full screen, no bookmarks bar | laptop |
| **P** | the compiled paper PDF | laptop |
| **R1** | ssh to ramesh, repo root, wide | Blackwell box |
| **R2** | ssh to ramesh, running `watch -n0.5 nvidia-smi --query-gpu=memory.used --format=csv` | Blackwell box |
| **B1** | ssh to bhaskar, repo root | A10 box |
| **B2** | the QEMU guest console, booted on the custom 6.8 kernel | A10 box |
| **B3** | ssh to bhaskar, `systemc/` directory | A10 box |

For the hardware shots put R1 on the left two thirds and R2 on the right third, so
GPU memory falls on camera while the run happens.

**Order to film.** Part four first, because it is the only part that can fail on
camera. Then five, seven, six, three, two, eight. Narrate last.

---

## 2. Pre-flight checklist

Run all of this the day before. Every item here is a shot that breaks if it fails.

```bash
# on ramesh
cd ~/nebula/memoe-proj
nvidia-smi --query-gpu=name,memory.total --format=csv   # shot 11
python scripts/bench_runtime.py --check                 # shot 12

# on bhaskar
cd ~/nebula/memoe-proj
bash scripts/run_deepseek.sh --batch 16384              # shot 18
cd systemc && make && ./memoe_tlm --batch 4626 --residency 0.3   # shot 27
# in the QEMU guest
cxl list -M && cat /sys/bus/cxl/devices/region0/commit   # shot 29
```

Files that must exist, because shots point straight at them:

```
results/figures_real/r1_coverage.png            shot 4
results/figures/f4_policy.png                   shot 5
results/figures_real/r2_overlap.png             shot 6
results/figures_real/r3_penalty.png             shot 7
results/figures_runtime/rt1_residency.png       shot 14
results/figures_runtime/rt2_batch_regime.png    shot 15
results/figures_runtime/rt3_depth.png           shot 16
results/figures_runtime/rt5_deepseek_crossover.png  shot 20
results/figures_serve/s1_decode_tpot.png        shot 22
results/figures_serve/s2_phase_cost.png         shot 23
results/figures_extra/e1_roofline.png           shot 24
results/figures_extra/e5_systemc_crossover.png  shot 28
results/qemu/cxl_committed.txt                  shot 29 fallback
```

If `results/figures_serve/` is missing, run `python scripts/plot_serve.py`.

---

## 3. Shot list

### Part one: the problem · 0:00 to 0:40

---

**Shot 1** · 0:00 · 12s

**WHERE** D, top of page, headline card.
**DO** Static, no scroll.
**SAY** DeepSeek-V3 has six hundred and seventy-one billion parameters and uses
about thirty-seven billion of them on any given token. The rest still have to sit
where the GPU can reach them.
**CUT** On the last word.

---

**Shot 2** · 0:12 · 14s

**WHERE** D, scroll slowly to the capacity card.
**SAY** So you buy sixteen H100s to store a model whose arithmetic would fit on
two. Ninety-three to ninety-eight percent of these checkpoints is expert weights,
and they are idle almost all of the time.

---

**Shot 3** · 0:26 · 14s

**WHERE** D, hold on the capacity card.
**SAY** Our question was whether those idle weights can live on a slower, larger
tier instead, and what that costs. The answer depends almost entirely on batch
size, and we can tell you exactly where the line is.

---

### Part two: three measurements that decided the design · 0:40 to 2:10

---

**Shot 4** · 0:40 · 25s

**WHERE** `results/figures_real/r1_coverage.png`, full screen.
**SAY** The obvious design is a cache. Keep the popular experts on the GPU, fetch
the rest on demand. This plot kills it. Horizontal axis is batch size in tokens on
a log scale. Vertical is the fraction of all experts one batch touches at one
layer. One line per workload: prose, math, code, chat. Every line goes to one. By
about a thousand tokens they are all above ninety-seven percent. At serving batch
sizes there is no working set, because the set is everything. The GPU tier stops
being a cache and becomes a fixed partition.

---

**Shot 5** · 1:05 · 18s

**WHERE** `results/figures/f4_policy.png`.
**SAY** The same conclusion from the policy side. Four placement strategies,
static popularity, balanced static, LRU and sampled LFU, against resident fraction.
They lie on top of each other. Hit rate equals the fraction you keep resident,
whatever policy you pick and whatever the routing skew. That is why our runtime
contains no ranking anywhere. Ranking would be code that computes something and
changes nothing.

---

**Shot 6** · 1:23 · 20s

**WHERE** `results/figures_real/r2_overlap.png`.
**SAY** Then, could you at least tune the placement per workload. This is the
overlap between the hot expert sets of two domains. The bright diagonal is a
control: split one workload in half and the halves agree about eighty-five percent
of the time, so the measurement is sound. Now look off the diagonal. Prose against
code is fifteen point two percent, where chance is twenty-five. They do not merely
fail to share experts, they avoid each other.

---

**Shot 7** · 1:43 · 27s

**WHERE** `results/figures_real/r3_penalty.png`.
**SAY** Which produces this. Rows are the workload a placement was profiled on,
columns are what it was then asked to serve. On the diagonal the routing hit rate
is around zero point seven one. Profile on prose, serve code, and it falls to zero
point three eight, against zero point five for placing experts at random. A tuned
placement on the wrong workload is worse than no tuning at all. So: no cache, no
predictor, no ranking, and never profile on a workload you are not going to serve.
Three of our four starting assumptions turned into deletions.

---

### Part three: what we built · 2:10 to 2:55

---

**Shot 8** · 2:10 · 22s

**WHERE** D, the pipeline animation card.
**DO** Let the animation run one full cycle.
**SAY** MEMoE-RT holds the non-resident experts in pinned host memory, laid out
contiguously by layer. While layer L computes, the entire expert set for layer L
plus one is already in flight on a separate copy stream, landing in a ring of
staging buffers whose addresses are fixed at load time. One transfer per layer, not
one per expert. That single decision matters more than it sounds like it should,
and there is a measurement for it later.

---

**Shot 9** · 2:32 · 13s

**WHERE** D, same card.
**DO** Drag the batch slider slowly from lowest to highest, then back to the middle.
**SAY** Watch the cost as the batch grows. Small batch, offload is expensive. Large
batch, close to free. Same configuration, same hardware, same code. The only thing
that changed is the batch.

---

**Shot 10** · 2:45 · 10s

**WHERE** D, the closed-form card, or P at the equation.
**SAY** That crossover has a closed form. It does not depend on expert size,
because a larger expert costs proportionally more to move and does proportionally
more work once it lands. And it does not depend on GPU count, because compute and
per-node bandwidth scale together.

---

### Part four: it runs on real hardware · 2:55 to 4:35

---

**Shot 11** · 2:55 · 8s

**WHERE** R1.
**DO** `nvidia-smi --query-gpu=name,memory.total --format=csv`
**SAY** A Blackwell card, thirty-two gigabytes. OLMoE-1B-7B fits on it
comfortably, which is exactly why it is the right model for measuring what offload
costs. There is a real baseline to compare against.

---

**Shot 12** · 3:03 · 14s

**WHERE** R1.
**DO** `python scripts/bench_runtime.py --check`
**SAY** Correctness first. This runs the tiered model and the untouched reference
on identical input and compares the outputs. If this does not pass, no timing
number underneath it means anything.
**CUT** The moment the pass line prints. Hold two seconds.

---

**Shot 13** · 3:17 · 33s

**WHERE** R1 left, R2 right, memory pane in frame.
**DO** `python scripts/bench_runtime.py --batch 16384 --residency 0.0 --depth 1`
**SAY** Now every expert in every layer is outside GPU memory, streamed in over
PCIe one layer ahead of use. Sixty-four experts, sixteen layers, thirty-eight and a
half gigabytes moved during the run. Twenty-four thousand five hundred and
eighty-three tokens per second, against a fully resident baseline of twenty-six
thousand nine hundred and forty-seven. Both are medians of three trials. That is
ninety-one point two percent of the throughput, in seven point two seven gigabytes
instead of seventeen point seven three. Two point four four times less memory. Time
stalled waiting on a transfer: zero point five six percent.
**CUT** Hold on the final output line two seconds.

---

**Shot 14** · 3:50 · 17s

**WHERE** `results/figures_runtime/rt1_residency.png`.
**SAY** The whole sweep. Horizontal axis is the fraction of experts kept on the
GPU, fully resident on the left, fully offloaded on the right. Two vertical axes,
throughput and peak GPU memory. Memory falls cleanly, which is just arithmetic.
Throughput falls much more slowly. The gap between those two curves is the entire
result: a little speed for a lot of capacity.

---

**Shot 15** · 4:07 · 18s

**WHERE** `results/figures_runtime/rt2_batch_regime.png`.
**SAY** The same system against batch size, with seventy percent of experts
offloaded. At five hundred and twelve tokens we keep thirty-eight percent of
baseline throughput, and offload is a disaster. At sixteen thousand three hundred
and eighty-four we keep ninety-four percent and stall for less than a tenth of a
percent. Identical configuration at both ends. The dotted line is where the closed
form said the crossover would be: it predicted four thousand six hundred and
twenty-six tokens, and the hardware crossed at four thousand six hundred and
forty-five. Half a percent, on a quantity the model was never fitted to.

---

**Shot 16** · 4:25 · 10s

**WHERE** `results/figures_runtime/rt3_depth.png`.
**SAY** And one result that corrected us. Our trace simulator recommended
prefetching four to eight layers ahead. On hardware, throughput is flat from one to
eight while staging memory grows forty-one percent. Depth one is enough once you
move a whole layer at a time. We report the correction, not the original
recommendation.

---

### Part five: the model that does not fit · 4:35 to 5:35

---

**Shot 17** · 4:35 · 8s

**WHERE** B1.
**DO** `nvidia-smi --query-gpu=name,memory.total --format=csv`
**SAY** Different machine. An A10, twenty-three gigabytes, on a slower PCIe
generation.

---

**Shot 18** · 4:43 · 27s

**WHERE** B1.
**DO** `bash scripts/run_deepseek.sh --batch 16384`
**SAY** DeepSeek-V2-Lite is thirty-one point four gigabytes of weights, and
twenty-eight point eight of that is the expert pool. There is no baseline here,
because the model does not fit on this card. It cannot run at all. With the entire
expert pool in host memory it runs in fifteen point one gigabytes at seven thousand
two hundred and forty tokens per second. On the previous machine we saved memory.
On this one we bought a capability that did not exist.
**CUT** Hold on the throughput line two seconds.

---

**Shot 19** · 5:10 · 12s

**WHERE** B1, same output on screen.
**SAY** This is a genuinely different architecture, not a rerun. Different top-k,
shared experts alongside the routed ones, a dense first layer, multi-head latent
attention, and twenty-six MoE layers instead of sixteen. The runtime handles it
through the same tiering path, with no architecture-specific transfer code.

---

**Shot 20** · 5:22 · 13s

**WHERE** `results/figures_runtime/rt5_deepseek_crossover.png`.
**SAY** Same plot as before, different model, different machine, different link
speed. The curve has the same shape and the tipping point lands where the same
formula puts it. The formula was derived from one model on one machine and
transferred without being refitted. That is the part worth pausing on.

---

### Part six: where it belongs in a real server · 5:35 to 6:37

---

**Shot 21** · 5:35 · 10s

**WHERE** D, the serving card.
**SAY** Everything so far was one phase of one request. A real serving request has
two, and they land on opposite sides of that line. So we built a serving loop with
a real KV cache and continuous batching, and measured both.

---

**Shot 22** · 5:45 · 18s

**WHERE** `results/figures_serve/s1_decode_tpot.png`.
**SAY** Time per output token against concurrency. The offloaded line is flat at
three hundred and twenty-five milliseconds while concurrency grows eightfold. The
resident line climbs from fifteen to fifty-two. A flat line is what a
bandwidth-bound cost looks like once it has already been paid: every extra sequence
rides along on the same transfer. The penalty is not a constant, so quoting a
single slowdown factor for offload is meaningless.

---

**Shot 23** · 6:03 · 22s

**WHERE** `results/figures_serve/s2_phase_cost.png`.
**SAY** Split by phase, on the same run. Offload adds seven percent to prefill and
five hundred and twenty-five percent to decode. Nothing changes between those two
bars except the batch size the phase presents. Put the slow tier on a prefill
worker and it is close to free. Put it on a decode worker and it takes most of your
throughput. Split that way we measured full throughput retained with a third of the
prefill-side memory removed. Prefill-decode disaggregation is already where serving
stacks are going, and this says which side the slow tier belongs on.

---

**Shot 24** · 6:25 · 12s

**WHERE** `results/figures_extra/e1_roofline.png`.
**SAY** The same arithmetic says what must not move. Arithmetic intensity against
the balance point of the tier. Expert weights sit above the line, so they can be
streamed. The KV cache sits far below it, because it is read on every single step.
Experts move. KV stays.

---

### Part seven: four checks that share no code · 6:37 to 7:55

---

**Shot 25** · 6:37 · 14s

**WHERE** D, the tier card, or `results/figures/f7_bandwidth.png`.
**SAY** All of this rests on one bandwidth number, so we measured it twice with
different tools. DRAMSim3 sustains sixteen point eight eight gigabytes per second on
a DDR4-3200 channel, against a nameplate of twenty-five point six. gem5,
independently, lands within seven percent. Two simulators with different memory
models agreeing is much harder to argue with than either alone.

---

**Shot 26** · 6:51 · 8s

**WHERE** B1.
**DO** `python scripts/parse_gem5.py`
**SAY** We also used gem5 to check that channel scaling is real rather than
assumed. Four channels delivered three point nine nine nine times one channel, so
the scaled bandwidth figures we quote are not wishful.

---

**Shot 27** · 6:59 · 22s

**WHERE** B3.
**DO** `./memoe_tlm --batch 4626 --residency 0.3`
**SAY** Then a third check, because the closed form was calibrated against the
runtime and those two are not independent witnesses. This is a SystemC TLM model of
the same data path. It shares no code with the runtime or the analytical model, and
it was never told where the crossover is. At a batch of four thousand six hundred
and twenty-six it puts one layer of transfer at thirteen point eight one four
milliseconds and one layer of compute at thirteen point eight one three. That
equality is exactly what the closed form asserts, reached from the other direction
by a discrete-event simulation.
**CUT** Hold on those two numbers for three full seconds.

---

**Shot 28** · 7:21 · 9s

**WHERE** `results/figures_extra/e5_systemc_crossover.png`.
**SAY** Its full sweep also reproduced the depth result independently. One, two,
four and eight layers of lookahead give identical throughput while staging memory
grows from one point one to five point one gigabytes. Our simulator was wrong about
that, the hardware corrected it, and this third model agrees with the hardware.

---

**Shot 29** · 7:30 · 25s

**WHERE** B2, the QEMU guest.
**DO** `cxl list -M`, then `cat /sys/bus/cxl/devices/region0/commit`, then
`dmesg | grep -i "bypassing cpu_cache"`
**SAY** Fourth, the software path. We do not have CXL silicon, so we exercised the
Linux CXL stack in QEMU. The device enumerates, the drivers bind, the decoders
program. Region commit fails on a stock kernel, because it is gated on a cache
invalidation check that returns false inside any virtual machine. We built a six
point eight kernel with the documented bypass, with no root access anywhere on the
host, and it commits two gigabytes. Two things we are not claiming. We committed
that region, we did not push traffic through it. And QEMU has no timing model, so
nothing here is a performance number. Every quantitative figure in this project came
from DRAMSim3, gem5, SystemC or the hardware.
**CUT** Hold on the commit equals one line.

---

### Part eight: close · 7:55 to 8:15

---

**Shot 30** · 7:55 · 10s

**WHERE** R1.
**DO** `bash scripts/reproduce.sh`
**SAY** One command, CPU only, no GPU and no downloads. Thirty-six tests, then
every table and every figure you have seen regenerates from a clean checkout.
**CUT** The moment thirty-six passed appears.

---

**Shot 31** · 8:05 · 20s

**WHERE** D, the recommendations card at the bottom.
**SAY** So here is what we would tell you to build. Expert offload is a prefill
technique. At prefill batch sizes you can put an entire expert pool on a tier fifty
times slower than HBM and lose seven percent. At decode batch sizes the same
configuration takes most of your throughput. The line between them is a formula you
can evaluate before buying any hardware, and it held across two model
architectures, two machines, two link speeds, and a model that fits alongside one
that does not.
**CUT** Hold two seconds. Cut to black. No logo, no music, no sign-off.

---

## 4. If something fails while you are recording

Do not re-record the narration. Cut to the stored file and keep reading.

| Shot | Fails | Show instead |
|---|---|---|
| 12 | `--check` | `results/runtime_bench_all.csv`, headline row |
| 13 | the 16,384 run | `results/runtime_bench.csv`, both rows |
| 18 | `run_deepseek.sh` | `results/runtime_deepseek_all.csv` |
| 26 | `parse_gem5.py` | the gem5 row of the dashboard comparison table |
| 27 | `./memoe_tlm` | `results/tables/15_systemc_sweep.csv`, the batch 4626 row |
| 29 | the QEMU guest | `results/qemu/cxl_committed.txt` |
| 30 | `reproduce.sh` | `results/REPORT.md` |

---

## 5. The five-minute cut

If a length limit appears, keep shots 1, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 17, 18,
20, 22, 23, 27, 30, 31 and drop the rest. That is the problem, the three
measurements, the system, both hardware results, the phase split, the independent
check and the close. It runs about five minutes ten.

Never cut 13, 18, 23 or 27: the hardware result, the model that does not fit, the
deployment rule, and the independent confirmation.

---

## 6. Delivery

- 1080p, H.264, mp4, under 500 MB
- Filename `MEMoE_KingBob_Nebula2026_demo.mp4`
- Upload unlisted to YouTube or Drive with link sharing on, then test the link in a
  private window before emailing
- Put the link in the email body, not only in the report

---

## 7. Reading notes

- Every plot: axes first, then what to look at, then what it means. Never read the
  title out loud.
- Pause a beat after each number. The numbers are the content.
- Land these hardest: ninety-one point two percent of the throughput in two point
  four four times less memory; it cannot run at all; seven percent and five hundred
  and twenty-five percent; and thirteen point eight one four against thirteen point
  eight one three.
- Say the two not-claiming lines in shot 29 flatly. In front of people who build CXL
  controllers, stating your limits is a strength.
- Do not apologise for what is not done. State the limit and move on.
