#!/usr/bin/env bash
# One command per component on bhaskar (A10). Setup is in docs/SETUP.md.
#
#   scripts/run_a10.sh <component>
#
# Components:
#   gpu            the card and link
#   tests          38 tests
#   olmoe-check    OLMoE: tiered output vs untouched reference
#   olmoe          OLMoE: fully resident vs fully offloaded at batch 16384
#   deepseek-check DeepSeek-V2-Lite: 25% vs 0% resident agree
#   deepseek       DeepSeek-V2-Lite: whole expert pool off-GPU at batch 16384
#   dramsim        DRAMSim3: CXL and HBM tiers
#   systemc        SystemC TLM: transfer equals compute at B*
#   gem5           gem5: 1/2/4/8 channel scaling
#   qemu           instructions for the CXL guest
#
# Outputs go to /tmp/memoe-a10 so the committed results/ stay untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=/tmp/memoe-a10
mkdir -p "$OUT"

case "${1:-}" in
  gpu)
    nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
    ;;
  tests)
    uv run python -m pytest tests -q
    ;;
  olmoe-check)
    bash scripts/run_runtime.sh --check
    ;;
  olmoe)
    bash scripts/run_runtime.sh --batch 16384 --residency 1.0 0.0 --depth 1 --reps 1 --out-dir "$OUT/olmoe"
    ;;
  deepseek-check)
    bash scripts/run_deepseek.sh --check
    ;;
  deepseek)
    bash scripts/run_deepseek.sh --batch 16384 --residency 0.0 --reps 1 --out "$OUT/deepseek.csv"
    ;;
  dramsim)
    D="${DRAMSIM:-$HOME/DRAMsim3}"
    for t in cxl:DDR4_8Gb_x8_3200 hbm:HBM2_8Gb_x128; do
      mkdir -p "$OUT/dramsim/${t%%:*}"
      "$D/build/dramsim3main" "$D/configs/${t#*:}.ini" -c 3000000 \
        -t results/dramsim/expert_fetch.trace -o "$OUT/dramsim/${t%%:*}" > /dev/null
    done
    uv run python scripts/parse_dramsim.py "$OUT/dramsim" "$D" | grep -v "^->"
    ;;
  systemc)
    export SYSTEMC_HOME="${SYSTEMC_HOME:-$HOME/local/systemc}"
    make -s -C systemc
    echo "batch,residency,depth,tier_bw_gbs,wall_s,compute_s,stall_s,stall_fraction,tokens_per_second,staging_gb,b_star"
    ./systemc/memoe_tlm --batch 4626 --residency 0.3 | tail -1
    ;;
  gem5)
    bash gem5/run_channels.sh
    ;;
  qemu)
    cat <<'EOF'
Terminal 1:  bash ~/cxlvm/run-cxl.sh          (wait for the login prompt)
Terminal 2:  ssh -p 2222 ubuntu@localhost      (password memoe), then:
  uname -r
  sudo cxl list -M
  sudo cxl create-region -m -t ram -d decoder0.0 -w 1 mem0
  cat /sys/bus/cxl/devices/region0/commit
  sudo dmesg | grep -i "bypassing cpu_cache"
EOF
    ;;
  *)
    sed -n 2,18p "$0"; exit 1
    ;;
esac
