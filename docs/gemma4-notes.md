# Gemma-4 notes (model id, chat template, GGUF export)

Verified 2026-06-15 against the Hugging Face API and the raw model files.

## Model id (get the casing right)
- Instruct: **`google/gemma-4-12B-it`** — note the **capital B**. (created 2026-05-23, Apache 2.0, ~1.08M downloads)
- Base: `google/gemma-4-12B`
- Other real Gemma-4 ids: `gemma-4-E2B-it`, `gemma-4-E4B-it`, `gemma-4-31B-it`, `gemma-4-26B-A4B-it` (MoE), plus `-qat-q4_0-gguf` / `-it-assistant` (drafter) variants.

`gemma-4` is real and current (NOT gemma-3). It is the **`gemma4_unified`** encoder-free multimodal arch. For a text chatbot you load/convert the **text tower**; vision/audio needs a separate mmproj (see export caveats).

## Chat template — DO NOT hand-roll
Gemma-4 uses **new** turn tokens: `<|turn>` (start) / `<turn|>` (end) — **not** Gemma-2/3's `<start_of_turn>`/`<end_of_turn>`. Roles are `user` / `assistant` / `system` (Gemma-3 used `model`). The `assistant` role renders as `<|turn>model\n` in the template.

**Always** use `tokenizer.apply_chat_template` (TRL/Unsloth do this for you). One-line sanity check before any long run:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
print(tok.apply_chat_template(
    [{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}], tokenize=False))
# You MUST see <|turn> markers, NOT <start_of_turn>.
```

### Loss masking (assistant-only)
- **TRL:** `SFTConfig(assistant_only_loss=True)` — relies on `{% generation %}` markers; TRL ≥0.20 auto-patches the template if missing. Don't hand-roll the mask.
- **Unsloth:** `train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")`.

## Export LoRA → GGUF (Q4_K_M)
`convert_hf_to_gguf.py` does **not** read PEFT adapters — **merge first**.

```bash
# 1. merge adapter into fp16 weights
python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", dtype=torch.bfloat16, device_map="cpu")
merged = PeftModel.from_pretrained(base, "gemma4-12b-sft").merge_and_unload()
merged.save_pretrained("gemma4-12b-merged", safe_serialization=True)
AutoTokenizer.from_pretrained("google/gemma-4-12B-it").save_pretrained("gemma4-12b-merged")
PY

# 2. convert + quantize (use a CURRENT llama.cpp checkout, post 2026-06-03)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --config Release
pip install -r requirements.txt
python convert_hf_to_gguf.py ../gemma4-12b-merged --outtype f16 --outfile gemma4-12b-f16.gguf
./build/bin/llama-quantize gemma4-12b-f16.gguf gemma4-12b-Q4_K_M.gguf Q4_K_M

# 3. test (recommended Gemma-4 sampling)
./build/bin/llama-cli -m gemma4-12b-Q4_K_M.gguf -p "Hallo" --temp 1.0 --top-p 0.95 --top-k 64
```

One-shot alternative via Unsloth (bundles a known-good llama.cpp):
```python
model.save_pretrained_gguf("gemma4-12b-gguf", tokenizer, quantization_method="q4_k_m")
```

## Export caveats
- **Multimodal:** `gemma-4-12B` is `gemma4_unified`. `convert_hf_to_gguf.py` converts the **text** tower; for multimodal GGUF prefer the official prebuilt GGUFs (`ggml-org/gemma-4-12B-it-GGUF`, `google/gemma-4-12B-it-qat-q4_0-gguf`).
- **Converter bug** [llama.cpp#19152]: Gemma conversion has historically expected a SentencePiece `tokenizer.model`. Ensure your merged dir contains `tokenizer.json` + `tokenizer_config.json` (copied from base, step 1). If it still fails, fall back to Unsloth's `save_pretrained_gguf` or the official GGUFs.
- **Drafter not supported:** `gemma-4-…-it-assistant` (`Gemma4AssistantForCausalLM`) is not yet handled by the converter — irrelevant for the standard `-it` model.
- **Early GGUF template bug:** Gemma-4 GGUFs pulled before ~May 2026 shipped a broken embedded chat template (model echoes literal tokens / hallucinates user turns). Re-pull the latest official GGUF or pass an updated `--chat-template-file`.

Sources: HF API `google/gemma-4-12B-it` · raw `config.json` / `tokenizer_config.json` / `chat_template.jinja` · [HF Gemma-4 blog](https://huggingface.co/blog/gemma4) · [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [transformers gemma4 doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gemma4.md)
