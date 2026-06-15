#!/usr/bin/env python
"""
Post-training check — did the fine-tune actually learn the task?

Loads the base model in 4-bit + the trained LoRA adapter and runs a small
HELD-OUT set of queries, printing the model's raw JSON and checking the
predicted target_level. This is how you verify a routing tune worked.

Usage (in WSL, training venv active):
  HSA_ENABLE_DXG_DETECTION=1 python examples/gemma4-12b-qlora/test_routing.py \
      --base google/gemma-4-E2B-it --adapter /root/aria-pilot/e2b-task-a
"""
import argparse, os, torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.enable_flash_sdp(False)          # gfx1201: use math SDPA
torch.backends.cuda.enable_mem_efficient_sdp(False)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Must MATCH the system prompt the dataset was built with (train == inference).
SYS = (
    "You are a HiRAG query-complexity router. Return JSON only (no prose, no markdown): "
    '{"query_type": "overview|specific|deep_dive|comparison|search", '
    '"target_level": 0|1|2, "confidence": 0.0-1.0, "reasoning": "...", '
    '"keywords": ["..."], "requires_cross_doc": true|false}. '
    "target_level 0=document summaries (only for listing which documents exist), "
    "1=chapter/section or a document summary, 2=detail chunks (default)."
)
# Held-out queries (NOT identical to training rows) + expected target_level.
TESTS = [
    ("Welche Dokumente kennst du?", 0),
    ("Liste alle Berichte auf", 0),
    ("List all documents you have", 0),
    ("Fasse Kapitel 3 zusammen", 1),
    ("Gib mir eine Zusammenfassung des Annual Reports", 1),
    ("Erkläre das Saga-Pattern im Detail", 2),
    ("Was kostet der Sonntagsbrunch?", 2),
    ("Vergleiche die Empfehlungen aus beiden Büchern", 0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base HF model id, e.g. google/gemma-4-E2B-it")
    ap.add_argument("--adapter", required=True, help="path to the trained LoRA adapter dir")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    ok = 0
    for q, exp in TESTS:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": f"Analyze this query:\n{q}"}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        n = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=False)
        txt = tok.decode(out[0][n:], skip_special_tokens=True).strip()
        hit = (f'"target_level": {exp}' in txt) or (f'"target_level":{exp}' in txt)
        ok += hit
        print(f"\nQ: {q}\n   expect L{exp}  {'OK' if hit else 'MISS'}\n   -> {txt[:220]}")
    print(f"\n=== {ok}/{len(TESTS)} target_level correct ===")


if __name__ == "__main__":
    main()
