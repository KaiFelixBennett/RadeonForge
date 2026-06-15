#!/usr/bin/env bash
# RadeonForge — build llama.cpp with the HIP/ROCm backend for AMD RDNA4 (gfx1201).
# Produces GPU-accelerated llama-server / llama-cli / llama-quantize for SERVING.
# A plain CPU `cmake -B build` is fine for convert/quantize but slow for inference
# AND does not build llama-server by default — use this for real serving.
# Verified 2026-06-15 on R9700 (gfx1201), ROCm 7.2, WSL2: ~117 tok/s on E2B Q4_K_M.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
GFX="${GFX:-gfx1201}"          # RDNA4: R9700 / RX 9070 (XT). RDNA3 = gfx1100.
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm

[ -d "$LLAMA_CPP" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"

echo "== configure (GGML_HIP=ON, $GFX) =="
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B "$LLAMA_CPP/build-hip" -S "$LLAMA_CPP" \
    -DGGML_HIP=ON -DAMDGPU_TARGETS="$GFX" -DGPU_TARGETS="$GFX" \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release

echo "== build (llama-server, llama-cli, llama-quantize) =="
cmake --build "$LLAMA_CPP/build-hip" -j --config Release \
  --target llama-server llama-cli llama-quantize

echo "== built -> $LLAMA_CPP/build-hip/bin/ =="
ls -lh "$LLAMA_CPP/build-hip/bin/" | grep -E "server|cli|quantize" || true
