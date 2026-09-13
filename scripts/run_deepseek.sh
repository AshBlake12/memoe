#!/usr/bin/env bash
# DeepSeek-V2-Lite wants its own environment. transformers is pinned at 4.44.2
# because the checkpoint's remote code doesn't survive later versions, and torch
# is pinned to the cu121 build matching the driver on bhaskar.
#
#   scripts/run_deepseek.sh --batch 6144
set -eu
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run --no-project --python 3.11 \
  --with "transformers==4.44.2" --with accelerate --with numpy --with pandas --with pyyaml \
  --with "torch==2.4.1" --extra-index-url https://download.pytorch.org/whl/cu121 \
  python scripts/bench_deepseek.py "$@"
