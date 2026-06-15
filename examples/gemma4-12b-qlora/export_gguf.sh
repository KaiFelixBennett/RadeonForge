#!/usr/bin/env bash
# RadeonForge — merge the trained LoRA into the base model and export GGUF (Q4_K_M).
# convert_hf_to_gguf.py does NOT read PEFT adapters, so we merge first.
set -euo pipefail

BASE="${BASE:-google/gemma-4-12B-it}"
ADAPTER="${ADAPTER:-gemma4-12b-sft}"
MERGED="${MERGED:-gemma4-12b-merged}"
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
OUT="${OUT:-gemma4-12b-Q4_K_M.gguf}"

echo "== 1/3 merge LoRA ($ADAPTER) into base ($BASE) -> $MERGED =="
python - <<PY
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained("$BASE", dtype=torch.bfloat16, device_map="cpu")
merged = PeftModel.from_pretrained(base, "$ADAPTER").merge_and_unload()
merged.save_pretrained("$MERGED", safe_serialization=True)
# Copy tokenizer files (the GGUF converter needs tokenizer.json/_config; see llama.cpp#19152)
AutoTokenizer.from_pretrained("$BASE").save_pretrained("$MERGED")
print("merged ->", "$MERGED")
PY

echo "== 2/3 ensure llama.cpp (current checkout, post 2026-06-03 for gemma4) =="
if [ ! -d "$LLAMA_CPP" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
  cmake -B "$LLAMA_CPP/build" -S "$LLAMA_CPP" && cmake --build "$LLAMA_CPP/build" -j --config Release
  pip install -r "$LLAMA_CPP/requirements.txt"
fi

echo "== 3/3 convert -> f16 GGUF, then quantize -> $OUT (Q4_K_M) =="
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outtype f16 --outfile gemma4-12b-f16.gguf
"$LLAMA_CPP/build/bin/llama-quantize" gemma4-12b-f16.gguf "$OUT" Q4_K_M

echo "== done: $OUT =="
echo "Note: gemma-4-12B is multimodal (gemma4_unified); this exports the TEXT tower only."
echo
echo "Next — SERVE on the GPU (RDNA4): build the HIP backend, then use serve.sh."
echo "  scripts/build_llamacpp_hip.sh        # builds llama-server for gfx1201"
echo "  MODEL=$OUT examples/gemma4-12b-qlora/serve.sh"
echo "IMPORTANT: serve with --chat-template-file gemma4-chat-template.jinja — llama.cpp's"
echo "--jinja mis-renders Gemma-4's embedded template (see docs/gemma4-notes.md 'Serving')."
