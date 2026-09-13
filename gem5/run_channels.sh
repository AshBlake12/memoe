#!/usr/bin/env bash
# gem5 channel scaling: 1, 2, 4, 8 interleaved DDR4-2400 channels.
#
#   bash gem5/run_channels.sh
#
# Needs build/NULL/gem5.opt from gem5 v22.1 (docs/SETUP.md). Writes
# results/gem5/ch<N>/stats.txt and prints sustained bandwidth per channel count.
set -euo pipefail
cd "$(dirname "$0")/.."
GEM5="${GEM5_HOME:-$HOME/local/src/gem5}"
OUT="${OUT:-results/gem5}"

printf "%-9s %12s %8s\n" channels "GB/s" ratio
base=""
for n in 1 2 4 8; do
  d="$OUT/ch$n"
  # inner shell reaps gem5, so its "Aborted" job message goes to /dev/null too
  if ! bash -c '"$@"; exit $?' _ "$GEM5/build/NULL/gem5.opt" -re --outdir="$d" \
      gem5/channel_scaling.py --gem5-configs "$GEM5/configs" --channels "$n" \
      > /dev/null 2>&1; then
    printf "%-9s %s\n" "$n" "bus saturated: $(grep -o 'panic: .*' "$d/simerr" | head -1)"
    continue
  fi
  gbs=$(awk '/^simSeconds/ {s=$2}
             /mem_ctrls[0-9]*\.dram\.bytesRead::total/ {b+=$2}
             END {printf "%.2f", b/s/1e9}' "$d/stats.txt")
  base=${base:-$gbs}
  printf "%-9s %12s %8s\n" "$n" "$gbs" "$(awk -v a="$gbs" -v b="$base" 'BEGIN{printf "%.3f", a/b}')"
done
