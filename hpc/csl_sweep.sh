#!/usr/bin/env bash
# Cross-arch decode thread-scaling sweep (compute-vs-bandwidth probe).
# Same model on Skylake (no VNNI) vs Cascade Lake (VNNI) tells us whether VNNI flips
# decode from compute-bound to bandwidth-bound (early saturation + higher BW utilization).
# Run inside: nix shell nixpkgs#numactl --command bash csl_sweep.sh /path/model.gguf
set -u
M="${1:?model path required}"
B="${B:-/home/zrg/llama.cpp/build-native/bin/llama-bench}"
echo "# arch=$(hostname) model=$(basename "$M")"
echo "# socket0-local thread sweep (decode tg, n=32, r=2)"
for th in 6 12 18 24; do
  echo "## threads=$th"
  numactl --cpunodebind=0 --membind=0 -- "$B" -m "$M" -p 0 -n 32 -r 2 -t "$th" 2>/dev/null | grep -E '\| *tg'
done
