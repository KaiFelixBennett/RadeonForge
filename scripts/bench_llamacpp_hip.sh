#!/usr/bin/env bash
# RadeonForge — proper llama.cpp HIP decode/prefill bench for RDNA4 (gfx1201, R9700).
# Uses llama-bench (the canonical tool) — builds the target if missing (the repo's
# build script only built server/cli/quantize). Reads GGUFs from WSL ext4 (NOT /mnt/e:
# mmap over drvfs is pathologically slow). Reports pp512 (prefill) + tg128 (decode)
# tok/s with flash-attn OFF vs ON, plus model params/size/backend (= GPU proof).
#
# Usage (inside WSL2 Ubuntu, as root):  bash bench_llamacpp_hip.sh
set -u
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
LLAMA=/root/llama.cpp
BENCH=$LLAMA/build-hip/bin/llama-bench

pkill -9 -x llama-cli 2>/dev/null; pkill -9 -x llama-bench 2>/dev/null; sleep 1

if [ ! -x "$BENCH" ]; then
  echo "== building llama-bench (one target, config is cached) =="
  cmake --build "$LLAMA/build-hip" -j"$(nproc)" --config Release --target llama-bench 2>&1 | tail -6
fi
[ -x "$BENCH" ] || { echo "FATAL: llama-bench build failed"; exit 1; }
echo

GEMMA=/root/router-pilot/12b-routing-Q4_K_M.gguf          # gemma-4-12B Q4_K_M (ext4)
MINI=/root/bench-models/ministral-3-14b-Q4_K_M.gguf     # Ministral-3-14B Q4_K_M (ext4)

for m in "$GEMMA" "$MINI"; do
  [ -f "$m" ] || { echo "MISSING: $m"; continue; }
  echo "############################################################"
  echo "# $m"
  echo "############################################################"
  HSA_ENABLE_DXG_DETECTION=1 "$BENCH" -m "$m" -ngl 99 -fa 0,1 -p 512 -n 128
  echo
done
echo DONE
