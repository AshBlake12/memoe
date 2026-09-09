"""
MEMoE-Serve: a serving loop that measures what expert offload costs in a real
request/response setting, rather than in a single batched forward pass.

Our batch sweeps show that expert offload is excellent at prefill batch sizes
and poor at decode batch sizes. That is a statement about arithmetic intensity,
and it has a direct architectural consequence: offload belongs on prefill
workers, not decode workers. Production serving is already moving toward that
split (chunked prefill, prefill/decode disaggregation), so the question is not
whether offload can be bolted onto a monolithic engine, but what it does to
time-to-first-token and time-per-output-token when placed correctly.

This module measures three configurations on the same workload:

  resident     every expert in GPU memory, the conventional deployment
  offload      every expert streamed, one engine serving both phases
  phase_aware  offload during prefill, resident during decode

The third is the disaggregated arrangement, measured by running each phase in
the configuration it would occupy on a real deployment and composing the
results. We do not simulate a network hop between workers; the numbers are
per-phase measurements, and we say so.

What this is not: a production engine. There is no PagedAttention, no
preemption, no prefix caching, and the KV cache is contiguous per sequence
rather than paged. Continuous batching is implemented: sequences join and leave
the running batch between decode steps, and the cache is gathered accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------

@dataclass
class Request:
    rid: int
    prompt_len: int
    max_new: int
    arrival: float = 0.0

    # filled in as it runs
    prefill_done: float = 0.0
    finish: float = 0.0
    generated: int = 0
    decode_times: list = field(default_factory=list)

    @property
    def ttft(self) -> float:
        return self.prefill_done - self.arrival

    @property
    def tpot(self) -> float:
        """Mean time per output token, excluding the first."""
        return (sum(self.decode_times) / len(self.decode_times)
                if self.decode_times else 0.0)


@dataclass
class ServeStats:
    config: str = ""
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ttfts: list = field(default_factory=list)
    tpots: list = field(default_factory=list)
    peak_vram_gb: float = 0.0
    prefill_stall: float = 0.0
    decode_stall: float = 0.0

    def summary(self) -> dict:
        def pct(xs, q):
            if not xs:
                return 0.0
            s = sorted(xs)
            return s[min(len(s) - 1, int(q * len(s)))]
        return {
            "config": self.config,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": round(self.wall, 3),
            "output_tokens_per_second": round(self.output_tokens / self.wall, 1)
            if self.wall else 0.0,
            "ttft_mean_ms": round(1000 * sum(self.ttfts) / len(self.ttfts), 1)
            if self.ttfts else 0.0,
            "ttft_p95_ms": round(1000 * pct(self.ttfts, 0.95), 1),
            "tpot_mean_ms": round(1000 * sum(self.tpots) / len(self.tpots), 2)
            if self.tpots else 0.0,
            "tpot_p95_ms": round(1000 * pct(self.tpots, 0.95), 2),
            "prefill_seconds": round(self.prefill_seconds, 3),
            "decode_seconds": round(self.decode_seconds, 3),
            "prefill_stall_seconds": round(self.prefill_stall, 4),
            "decode_stall_seconds": round(self.decode_stall, 4),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
        }


# --------------------------------------------------------------------------
# cache helpers
# --------------------------------------------------------------------------

def _layers(cache):
    """Return the per-layer (k, v) list from whatever cache object we got."""
    if cache is None:
        return []
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)


def _rebuild(cache, keep: torch.Tensor):
    """Drop finished sequences from the batch dimension of every layer."""
    from transformers import DynamicCache
    out = DynamicCache()
    for i, (k, v) in enumerate(_layers(cache)):
        out.update(k.index_select(0, keep), v.index_select(0, keep), i)
    return out


def _left_pad_to(cache, target_len: int):
    """
    Pad every layer's cache on the left so a batch of sequences at different
    positions can be decoded together. Padded positions are masked out, so the
    values themselves do not matter.
    """
    from transformers import DynamicCache
    out = DynamicCache()
    for i, (k, v) in enumerate(_layers(cache)):
        pad = target_len - k.shape[2]
        if pad > 0:
            zk = torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3],
                             dtype=k.dtype, device=k.device)
            zv = torch.zeros_like(zk)
            k = torch.cat([zk, k], dim=2)
            v = torch.cat([zv, v], dim=2)
        out.update(k, v, i)
    return out


def _concat_batch(a, b):
    """Join two caches along the batch dimension. Lengths must already match."""
    from transformers import DynamicCache
    if a is None:
        return b
    out = DynamicCache()
    la, lb = _layers(a), _layers(b)
    for i, ((ka, va), (kb, vb)) in enumerate(zip(la, lb)):
        out.update(torch.cat([ka, kb], dim=0), torch.cat([va, vb], dim=0), i)
    return out


def _cache_len(cache) -> int:
    ls = _layers(cache)
    return ls[0][0].shape[2] if ls else 0


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

class ServeEngine:
    """
    Continuous batching over a fixed number of slots. Sequences are admitted
    when a slot frees, prefilled, then folded into the running decode batch.
    """

    def __init__(self, model, tier=None, device="cuda", max_running: int = 32,
                 max_prefill_tokens: int = 8192, vocab: int | None = None):
        self.model = model
        self.tier = tier
        self.device = torch.device(device)
        self.max_running = max_running
        self.max_prefill_tokens = max_prefill_tokens
        self.vocab = vocab or model.config.vocab_size
        self.g = torch.Generator(device="cpu").manual_seed(0)

    def _ids(self, shape) -> torch.Tensor:
        return torch.randint(0, self.vocab, shape, generator=self.g).to(self.device)

    def _stall(self) -> float:
        if self.tier is None:
            return 0.0
        torch.cuda.synchronize()
        self.tier.collect_stalls()
        return self.tier.stats.stall_seconds

    @torch.no_grad()
    def _prefill(self, reqs: list[Request]):
        """
        Prefill a cohort together. Sequences are right-aligned by left padding,
        which is also the alignment decode needs, so no reshuffling later.
        """
        maxlen = max(r.prompt_len for r in reqs)
        n = len(reqs)
        ids = self._ids((n, maxlen))
        mask = torch.zeros(n, maxlen, dtype=torch.long, device=self.device)
        for i, r in enumerate(reqs):
            mask[i, maxlen - r.prompt_len:] = 1

        if self.tier is not None:
            self.tier.warmup()
        out = self.model(input_ids=ids, attention_mask=mask, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return out.past_key_values, mask, nxt

    @torch.no_grad()
    def _decode_step(self, tokens, cache, mask):
        if self.tier is not None:
            self.tier.warmup()
        out = self.model(input_ids=tokens, attention_mask=mask,
                         past_key_values=cache, use_cache=True)
        return out.logits[:, -1, :].argmax(-1, keepdim=True), out.past_key_values

    @torch.no_grad()
    def run(self, requests: list[Request], config: str) -> ServeStats:
        st = ServeStats(config=config)
        st.requests = len(requests)
        st.prompt_tokens = sum(r.prompt_len for r in requests)

        pending = list(requests)
        running: list[Request] = []
        cache = None
        mask = None
        tokens = None

        torch.cuda.reset_peak_memory_stats(self.device)
        if self.tier is not None:
            self.tier.reset_stats()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        last_stall = 0.0

        while pending or running:
            # ---- admit ----
            if pending and len(running) < self.max_running:
                cohort, budget = [], 0
                while pending and len(running) + len(cohort) < self.max_running:
                    if budget + pending[0].prompt_len > self.max_prefill_tokens and cohort:
                        break
                    r = pending.pop(0)
                    cohort.append(r)
                    budget += r.prompt_len

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                new_cache, new_mask, new_tok = self._prefill(cohort)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                st.prefill_seconds += dt

                s = self._stall()
                st.prefill_stall += s - last_stall
                last_stall = s

                now = time.perf_counter()
                for r in cohort:
                    r.prefill_done = now
                    r.generated = 1
                    st.ttfts.append(now - t_start)

                # fold into the running batch, aligning cache lengths
                if cache is None:
                    cache, mask, tokens = new_cache, new_mask, new_tok
                else:
                    target = max(_cache_len(cache), _cache_len(new_cache))
                    cache = _left_pad_to(cache, target)
                    new_cache = _left_pad_to(new_cache, target)
                    cache = _concat_batch(cache, new_cache)
                    mask = torch.cat([
                        torch.nn.functional.pad(mask, (target - mask.shape[1], 0)),
                        torch.nn.functional.pad(new_mask,
                                                (target - new_mask.shape[1], 0)),
                    ], dim=0)
                    tokens = torch.cat([tokens, new_tok], dim=0)
                running.extend(cohort)
                continue

            # ---- decode one step for everyone running ----
            mask = torch.cat([mask, torch.ones(mask.shape[0], 1,
                                               dtype=mask.dtype,
                                               device=self.device)], dim=1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokens, cache = self._decode_step(tokens, cache, mask)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            st.decode_seconds += dt
            st.output_tokens += len(running)

            s = self._stall()
            st.decode_stall += s - last_stall
            last_stall = s

            for r in running:
                r.generated += 1
                r.decode_times.append(dt)

            # ---- retire ----
            keep = [i for i, r in enumerate(running) if r.generated < r.max_new]
            if len(keep) != len(running):
                done = [r for i, r in enumerate(running) if i not in set(keep)]
                now = time.perf_counter()
                for r in done:
                    r.finish = now
                    st.tpots.append(r.tpot)
                if not keep:
                    cache, mask, tokens, running = None, None, None, []
                else:
                    idx = torch.tensor(keep, device=self.device)
                    cache = _rebuild(cache, idx)
                    mask = mask.index_select(0, idx)
                    tokens = tokens.index_select(0, idx)
                    running = [running[i] for i in keep]

        torch.cuda.synchronize()
        st.wall = time.perf_counter() - t_start
        st.peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        return st


# --------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------

def make_workload(n: int, prompt_len: int, max_new: int) -> list[Request]:
    return [Request(rid=i, prompt_len=prompt_len, max_new=max_new)
            for i in range(n)]
