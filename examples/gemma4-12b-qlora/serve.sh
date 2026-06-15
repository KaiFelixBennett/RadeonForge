#!/usr/bin/env bash
# RadeonForge — serve a Gemma-4 GGUF on an AMD RDNA4 GPU (gfx1201) with llama.cpp.
#
# WHY THE FLAGS MATTER (verified 2026-06-15 on R9700/gfx1201):
# llama.cpp's --jinja engine (minja) MIS-RENDERS Gemma-4's embedded <|turn>
# chat template. Symptoms on /v1/chat/completions WITHOUT the fix below:
#   * model drifts into a "Thinking Process:" monologue (reasoning_content), or
#   * with --reasoning-budget 0 it echoes literal <|turn>model<|turn>... forever.
# FIX: pass the known-good template via --chat-template-file (bundled next to
# this script). Then /v1/chat/completions returns clean output (finish_reason
# "stop", correct content). Fallback needing no template: POST token-id prompts
# to /completion, formatting client-side with apply_chat_template (see RUNBOOK).
set -euo pipefail

MODEL="${MODEL:-gemma4-12b-Q4_K_M.gguf}"
LLAMA_BIN="${LLAMA_BIN:-./llama.cpp/build-hip/bin/llama-server}"   # built by scripts/build_llamacpp_hip.sh
TEMPLATE="${TEMPLATE:-$(dirname "$0")/gemma4-chat-template.jinja}"
PORT="${PORT:-8099}"
NGL="${NGL:-99}"          # offload all layers to GPU
CTX="${CTX:-4096}"

export HSA_ENABLE_DXG_DETECTION=1                       # WSL2 + RDNA4
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

echo "serving $MODEL on :$PORT (ngl=$NGL) with $TEMPLATE"
exec "$LLAMA_BIN" -m "$MODEL" \
  --jinja --chat-template-file "$TEMPLATE" --reasoning-budget 0 \
  --host 127.0.0.1 --port "$PORT" -ngl "$NGL" -c "$CTX"
