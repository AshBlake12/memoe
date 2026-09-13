#!/usr/bin/env bash
# OLMoE runtime benchmark on bhaskar. The project venv is Python 3.14, for which
# no cu121 torch exists, so this uses the same throwaway 3.11 + cu121 env that
# run_deepseek.sh uses. transformers is >=4.45 because that is where OLMoE landed.
#
#   scripts/run_runtime.sh --check
#   scripts/run_runtime.sh --batch 16384 --residency 1.0 0.0 --depth 1
set -eu
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run --no-project --python 3.11 \
  --with "transformers>=5.0.0" --with accelerate --with numpy \
  --with pandas --with pyyaml \
  --with "torch>=2.5.0" --extra-index-url https://download.pytorch.org/whl/cu121 \
  python scripts/bench_runtime.py "$@"
