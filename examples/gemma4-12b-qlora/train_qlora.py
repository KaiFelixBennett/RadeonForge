#!/usr/bin/env python
"""
RadeonForge — Gemma-4-12B QLoRA trainer (Track A: WSL2 + ROCm, AMD gfx1201/RDNA4).

The two RDNA4 workarounds are baked in:
  (1) optim="adamw_torch"  — never a paged optimizer (silently corrupts on gfx1201).
  (2) math SDPA            — flash/mem-efficient SDPA crash during training on RDNA4.

The Gemma-4 chat template (<|turn>/<turn|>, role "assistant"->"model") is handled
automatically by SFTTrainer via the model's own tokenizer — do NOT hand-roll prompts.

Run:
    python examples/gemma4-12b-qlora/train_qlora.py --config examples/gemma4-12b-qlora/config.yaml
Then export to GGUF:
    bash examples/gemma4-12b-qlora/export_gguf.sh
"""
import argparse, os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import yaml
import torch
# (2) RDNA4: force math SDPA.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    lora, tr = cfg["lora"], cfg["train"]

    assert "paged" not in tr["optim"], "Refusing to run: paged optimizers silently corrupt training on gfx1201."

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if tr.get("load_in_4bit", True) else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=quant,
        dtype=torch.bfloat16,
        attn_implementation=tr.get("attn_implementation", "sdpa"),   # (2)
        device_map={"": 0},
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.05),
        bias="none", target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )

    # Conversational dataset: each row {"messages":[{"role":..,"content":..}, ...]}.
    # SFTTrainer applies the model's chat template + assistant-only loss for us.
    ds = load_dataset("json", data_files=cfg["dataset"], split="train")

    args = SFTConfig(
        output_dir=cfg["output_dir"],
        optim=tr["optim"],                                  # (1) adamw_torch
        assistant_only_loss=True,                           # mask everything but assistant spans
        max_length=tr.get("max_length", 2048),
        per_device_train_batch_size=tr.get("per_device_batch_size", 1),
        gradient_accumulation_steps=tr.get("grad_accum", 8),
        learning_rate=float(tr["learning_rate"]),
        num_train_epochs=tr.get("epochs", 1),
        bf16=tr.get("bf16", True),
        gradient_checkpointing=tr.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        seed=tr.get("seed", 42),
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=args, peft_config=peft_cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"\n✅ LoRA adapter saved to: {cfg['output_dir']}")
    print("   Next: bash examples/gemma4-12b-qlora/export_gguf.sh")


if __name__ == "__main__":
    main()
