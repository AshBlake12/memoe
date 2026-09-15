#!/usr/bin/env python3
"""Build a small, randomly initialised MoE checkpoint for trying the runtime.

    uv run --no-project --python 3.11 --with "transformers>=5.0.0" --with "torch>=2.5.0" \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        python scripts/make_tiny_moe.py --arch qwen3_moe --out /tmp/tiny-qwen3moe
    bash scripts/run_runtime.sh --ckpt /tmp/tiny-qwen3moe --check

A few megabytes, no download, same module layout as the real architecture. Use it
to confirm a new machine, launcher environment or transformers version works
end to end before pulling a large checkpoint. Throughput numbers from a model
this small are dominated by kernel overhead, so use it for --check, not timing.

transformers 5.x stores Qwen3-MoE and Mixtral experts as fused tensors;
transformers 4.x stores Mixtral experts as a ModuleList. Building the same
--arch under both covers both of MEMoE-RT's code paths.
"""

from __future__ import annotations

import argparse

import torch
import transformers


def build(arch: str):
    common = dict(vocab_size=4096, hidden_size=512, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=2)
    if arch == "qwen3_moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        cfg = Qwen3MoeConfig(intermediate_size=1024, moe_intermediate_size=256,
                             head_dim=64, num_experts=32, num_experts_per_tok=4,
                             decoder_sparse_step=1, mlp_only_layers=[], **common)
        return Qwen3MoeForCausalLM(cfg)
    if arch == "mixtral":
        from transformers import MixtralConfig, MixtralForCausalLM
        cfg = MixtralConfig(intermediate_size=1024, num_local_experts=8,
                            num_experts_per_tok=2, **common)
        return MixtralForCausalLM(cfg)
    raise SystemExit(f"unknown --arch {arch}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["qwen3_moe", "mixtral"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    model = build(a.arch).to(torch.bfloat16)
    model.save_pretrained(a.out)

    experts = next(m.experts for m in model.modules() if hasattr(m, "experts"))
    layout = ("ModuleList" if isinstance(experts, torch.nn.ModuleList)
              else "fused tensors")
    params = sum(t.numel() for t in model.parameters())
    print(f"wrote {a.out}: {a.arch}, {params / 1e6:.1f}M parameters, "
          f"experts as {layout} (transformers {transformers.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
