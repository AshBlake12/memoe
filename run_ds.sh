#!/usr/bin/env bash
cd "$(dirname "$0")"
PYTHONPATH=. uv run --no-project --python 3.11 \
  --with "transformers==4.44.2" --with accelerate --with numpy \
  --with "torch==2.4.1" --extra-index-url https://download.pytorch.org/whl/cu121 \
  python scripts/bench_deepseek.py "$@"
