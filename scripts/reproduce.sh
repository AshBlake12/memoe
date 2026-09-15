#!/usr/bin/env bash
# rebuilds every table, figure and report under results/ from a clean checkout.
#
#   pip install -r requirements.txt
#   scripts/reproduce.sh
#
# cpu only. no gpu, no checkpoint, no network. about 8 minutes.
#
# the runtime benchmarks are deliberately not in here, since they need a gpu and
# a model on disk. those live in scripts/bench_runtime.py.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== tests =="
python -m pytest tests -q

echo; echo "== analysis + simulation (tables/, figures/, REPORT.md) =="
python scripts/run_all.py

echo; echo "== closed-form extras (tables_extra/, figures_extra/) =="
python scripts/run_extras.py

echo; echo "== captured-trace results (tables_real/, figures_real/) =="
python scripts/run_real.py

echo; echo "== offloadable frontier =="
python scripts/frontier.py

echo; echo "== dashboard =="
python scripts/build_dashboard.py

echo; echo "Done. See results/REPORT.md and results/dashboard.html"
