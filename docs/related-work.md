# Related work and where MEMoE sits

Expert offloading is a crowded area. This section says what the existing systems
do, what MEMoE-RT shares with them, and what is new here.

## The four camps

**Cache and predict.** The largest group assumes expert reuse has temporal
locality and exploits it. Mixtral-offloading keeps an LRU cache of experts on the
GPU and speculatively prefetches one to two layers ahead, on the premise that
consecutive tokens activate overlapping experts [Eliseev & Mazur, arXiv:2312.17238].
MoE-Infinity adds request-level activation tracing; ProMoE trains a predictor;
HOBBIT varies expert precision by importance and compares LRU, sequence-level LFU
and a layer-distance policy; AdapMoE skips experts judged unimportant. The shared
commitment is that *which* experts are needed is worth predicting.

**Move the activation, not the weight.** Fiddler observes that at small batch the
activation is far smaller than an expert's weights, so it copies activations to
the host and runs the expert layer on the CPU rather than moving weights to the
GPU [Kamahori et al., ICLR 2025, arXiv:2402.07033]. Its initial placement is
popularity-ranked from calibration data. Reported gains are on single-batch
latency, long prefill and beam search.

**Throughput-oriented pipelining.** ZeRO-Inference offloads layer weights to host
or NVMe with prefetch, and explicitly targets throughput at large batch rather
than single-request latency; its own finding is that for that regime full offload
beats partial offload, because the larger batch it enables pays for the transfer
[DeepSpeed, 2022]. FlexGen schedules tensor placement across GPU, CPU and disk.
MoE-Lightning adds a CPU-GPU-I/O pipeline with paged weights and a Hierarchical
Roofline Model used to pick a policy, reporting up to 10.3x over prior
offloading systems for Mixtral-8x7B on one T4 [Cao et al., ASPLOS 2025,
arXiv:2411.11217]. MoE-Gen batches per module.

**Static placement in production.** llama.cpp's `--cpu-moe` / `--n-cpu-moe` and
`--override-tensor` pin whole routed-expert tensors to host RAM by layer. There is
no ranking and no prediction. It is the simplest thing that works and it is what
most people actually run.

## What MEMoE-RT shares with them

MEMoE-RT builds on established mechanisms:

- Layer-granular prefetch on a separate stream is what ZeRO-Inference does.
- Static, unranked expert placement is what llama.cpp does.
- "Full offload beats partial offload in the throughput regime" is
  ZeRO-Inference's 2022 result, not ours.
- Having an analytical model that selects a policy is MoE-Lightning's HRM.

What differs is that we arrived at each of these from measurement on captured
routing rather than adopting them, and the measurements say *why* they hold.

## What is new

**A closed-form regime boundary, validated against a runtime.** Existing work
optimises a point in the design space and reports a speedup against a chosen
baseline. We give the batch size at which expert offload stops costing
throughput,

```
B* = (1 - h) · E · F / (BW · k · ε)
```

and show it is independent of expert size and of GPU count. On OLMoE it agrees
with the crossover from MEMoE-RT's measured transfer and compute times to within
0.5%, and with an independent SystemC model exactly. MoE-Lightning's HRM
is the nearest relative, but it selects a micro-batch policy for a given system;
B* is a threshold a reader can evaluate for hardware they do not have.

**Popularity is not worth predicting at serving batch sizes.** This is the claim
that cuts against the largest camp. We measure fetch hit rate equal to resident
fraction exactly, regardless of routing distribution, which makes LRU, LFU and
static placement equivalent. The mechanism is coverage saturation: past a few
hundred tokens a batch touches essentially every expert at every layer, so the
GPU tier stops being a cache and becomes a fixed partition.

This does not contradict the skew those systems report. Their observations are at
batch size 1, where consecutive-token reuse is real and above chance. Ours are at
serving batch sizes. Both can be true, and the reconciliation is the useful part:
prediction machinery earns its complexity in the single-stream regime and stops
earning it as batch grows. llama.cpp's unranked static placement is, on our
measurements, the right design at serving scale. llama.cpp arrived at it by
engineering judgement, and this gives it a reason.

**Profiled placement can hurt across domains.** Prose and code hot sets overlap
15.2% against 25% chance, while each workload overlaps 84.9% with itself. A
placement profiled on prose serves code worse than random. Fiddler and others rank initial placement from
calibration data; our measurements show how far that transfers.

**The tier is characterised, not quoted.** Every system above uses host DRAM over
PCIe. We characterise a CXL-class tier in DRAMSim3 (16.88 GB/s sustained, 179.4 ns
loaded device latency on DDR4-3200) and use measured sustained bandwidth
throughout rather than link rates, with PCIe as the stand-in link. The
capacity and pooling analysis, and B*'s independence from GPU count, are the parts
specific to a disaggregated tier rather than an attached one.

## Comparing published numbers

Published figures from these systems measure different things from ours:

1. **Different baselines.** Our 91.2% throughput retention is against a fully
   resident model on a GPU that fits it. MoE-Lightning's 10.3x and Fiddler's
   speedups are against other offloading systems on GPUs that cannot fit the
   model at all. These are different quantities, so dividing one by the other
   says nothing.
2. **Different regimes.** Most of this work targets batch 1 to 32 on consumer
   hardware. Our measurements run to batch 16,384, and our own result is that the
   regime determines the answer, so a number carried across regimes changes
   meaning.
3. **Different hardware and models.** T4, RTX 3060, Quadro RTX 6000, A30 and our
   RTX PRO 4500 and A10 differ in host-to-device bandwidth, which is the term B*
   is most sensitive to after hit rate.

## References

- Eliseev & Mazur. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* arXiv:2312.17238.
- Kamahori, Gu, Zhu, Kasikci. *Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.* ICLR 2025. arXiv:2402.07033.
- Aminabadi et al. *DeepSpeed Inference / ZeRO-Inference.* SC 2022; DeepSpeed blog, September 2022.
- Sheng et al. *FlexGen: High-Throughput Generative Inference of Large Language Models.* arXiv:2303.06865.
- Cao et al. *MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs.* ASPLOS 2025. arXiv:2411.11217.
- Xue et al. *MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving.* arXiv:2401.14361.
- Tang et al. *HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference.* arXiv:2411.01433.
- Xu et al. *MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching.* arXiv:2503.09716.
- ggml-org/llama.cpp. `--cpu-moe`, `--n-cpu-moe`, `--override-tensor`.
