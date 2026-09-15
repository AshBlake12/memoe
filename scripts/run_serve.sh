#!/usr/bin/env bash
# Serving loop on bhaskar. Same environment as scripts/run_runtime.sh.
#
#   scripts/run_serve.sh --requests 16 --gen 32
set -eu
cd "$(dirname "$0")/.."
# Override for other models or GPUs, e.g. a Blackwell card needs CUDA 12.8 wheels:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/run_serve.sh --check
#   TRANSFORMERS="transformers==4.44.2" TORCH="torch==2.4.1" bash scripts/run_serve.sh --ckpt ...
PY="${PY:-3.11}"
TRANSFORMERS="${TRANSFORMERS:-transformers>=5.0.0}"
TORCH="${TORCH:-torch>=2.5.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
PYTHONPATH=. uv run --no-project --python "$PY" \
  --with "$TRANSFORMERS" --with accelerate --with numpy \
  --with pandas --with pyyaml \
  --with "$TORCH" --extra-index-url "$TORCH_INDEX" \
  python scripts/bench_serve.py "$@"
