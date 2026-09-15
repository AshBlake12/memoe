#!/usr/bin/env bash
# OLMoE runtime benchmark on bhaskar. The project venv is Python 3.14, for which
# no cu121 torch exists, so this uses the same throwaway 3.11 + cu121 env that
# run_deepseek.sh uses. transformers is >=4.45 because that is where OLMoE landed.
#
#   scripts/run_runtime.sh --check
#   scripts/run_runtime.sh --batch 16384 --residency 1.0 0.0 --depth 1
set -eu
cd "$(dirname "$0")/.."
# Override for other models or GPUs, e.g. a Blackwell card needs CUDA 12.8 wheels:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/run_runtime.sh --check
#   TRANSFORMERS="transformers==4.44.2" TORCH="torch==2.4.1" bash scripts/run_runtime.sh --ckpt ...
PY="${PY:-3.11}"
TRANSFORMERS="${TRANSFORMERS:-transformers>=5.0.0}"
TORCH="${TORCH:-torch>=2.5.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
PYTHONPATH=. uv run --no-project --python "$PY" \
  --with "$TRANSFORMERS" --with accelerate --with numpy \
  --with pandas --with pyyaml \
  --with "$TORCH" --extra-index-url "$TORCH_INDEX" \
  python scripts/bench_runtime.py "$@"
