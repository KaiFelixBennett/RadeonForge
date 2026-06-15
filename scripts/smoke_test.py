#!/usr/bin/env python
"""
RadeonForge smoke test — the durable reproducibility guarantee.

Runs a tiny 4-bit QLoRA training loop on a small model and FAILS LOUDLY if the
environment hits the known RDNA4/gfx1201 silent-corruption traps:

  * loss collapses to 0.0 / grad becomes NaN  (the paged-optimizer / bad-backend bug)
  * loss does not decrease at all              (broken autograd / attention path)

It deliberately uses a RAW PyTorch loop (no HF Trainer) because the Trainer can
trip HIP edge cases where a raw loop succeeds — so this isolates the GPU + bnb +
optimizer + LoRA path itself.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --model HuggingFaceTB/SmolLM2-135M-Instruct --steps 60
    python scripts/smoke_test.py --no-4bit          # plain bf16 LoRA fallback path
"""
import argparse, os, sys, math

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
# RDNA4 workaround: flash / mem-efficient SDPA crash in training -> force math SDPA.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                    help="small, ungated model for a fast environment check")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--no-4bit", action="store_true", help="skip bitsandbytes; bf16 LoRA path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() is False — no ROCm GPU. See docs/track-a-wsl2-rocm.md")
        return 2
    dev = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(dev) \
        if args.no_4bit else AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, **load_kwargs)

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()

    # Tiny fixed dataset (must be learnable in ~50 steps).
    texts = [
        "RadeonForge proves that fine-tuning works on AMD RDNA4.",
        "The capital of France is Paris.",
        "Two plus two equals four.",
        "Gemma is a family of open models from Google.",
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = enc.input_ids.to(dev)
    attn = enc.attention_mask.to(dev)
    labels = input_ids.clone()
    labels[attn == 0] = -100

    # adamw_torch — NEVER a paged optimizer on gfx1201 (silently corrupts).
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)

    losses = []
    for step in range(args.steps):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = loss.item()
        losses.append(lv)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:>3} | loss {lv:.4f}")
        if not math.isfinite(lv):
            print(f"\n❌ FAIL: loss became {lv} at step {step} — NaN/Inf.")
            print("   Classic gfx1201 trap. Check: NOT using a paged optimizer; bitsandbytes ROCm backend OK; math SDPA active.")
            return 1

    first = sum(losses[:3]) / min(3, len(losses))
    last = sum(losses[-3:]) / min(3, len(losses))
    print(f"\n  avg loss: first≈{first:.4f}  →  last≈{last:.4f}")

    if last >= first - 0.05:
        print("❌ FAIL: loss did not decrease. Autograd/attention/optimizer path is broken on this stack.")
        return 1
    if last <= 1e-4:
        print("❌ FAIL: loss collapsed to ~0 — the silent paged-optimizer/backend corruption signature.")
        return 1

    print("✅ PASS: loss is finite and decreasing. This environment can train. You're good to run the real job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
