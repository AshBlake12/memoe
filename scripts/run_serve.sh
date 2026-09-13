#!/usr/bin/env bash
# Serving loop on bhaskar. Same environment as scripts/run_runtime.sh.
#
#   scripts/run_serve.sh --requests 16 --gen 32
set -eu
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run --no-project --python 3.11 \
  --with "transformers>=5.0.0" --with accelerate --with numpy \
  --with pandas --with pyyaml \
  --with "torch>=2.5.0" --extra-index-url https://download.pytorch.org/whl/cu121 \
  python scripts/bench_serve.py "$@"
