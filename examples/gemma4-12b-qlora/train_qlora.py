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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


class ProgressCallback(TrainerCallback):
    """Opt-in: writes live status (step/epoch/loss) as JSON for progress_dashboard.py.
    Active only when cfg['progress_file'] is set. Pure stdlib, also works across
    /mnt/e from inside WSL."""
    def __init__(self, path: str):
        import os
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.total = None
        self._ptok = None; self._ptime = None; self._tps = []   # tokens/s history
        self._loss = []                                          # loss history (for the live loss chart)

    def _write(self, state, status):
        import json, os, time
        loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), None)
        ntok = next((h["num_tokens"] for h in reversed(state.log_history) if "num_tokens" in h), None)
        now = time.time()
        tps = None
        if ntok is not None and self._ptok is not None and self._ptime and now > self._ptime:
            d = ntok - self._ptok
            if d > 0:
                tps = round(d / (now - self._ptime))
                self._tps.append(tps); self._tps = self._tps[-80:]
        if ntok is not None:
            self._ptok = ntok; self._ptime = now
        if loss is not None and (not self._loss or self._loss[-1] != loss):
            self._loss.append(round(float(loss), 4)); self._loss = self._loss[-200:]
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"name": os.path.splitext(os.path.basename(self.path))[0],
                       "status": status, "step": state.global_step,
                       "total_steps": self.total or state.max_steps,
                       "epoch": state.epoch, "epochs": round(state.num_train_epochs)
                       if getattr(state, "num_train_epochs", None) else None,
                       "loss": loss, "tps": tps, "tps_series": self._tps,
                       "loss_series": self._loss,
                       "updated": time.time()}, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def on_train_begin(self, args, state, control, **kw):
        self.total = state.max_steps
        self._write(state, "running")

    def on_log(self, args, state, control, **kw):
        self._write(state, "running")

    def on_train_end(self, args, state, control, **kw):
        self._write(state, "done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    lora, tr = cfg["lora"], cfg["train"]

    assert "paged" not in tr["optim"], "Refusing to run: paged optimizers silently corrupt training on gfx1201."

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], **cfg.get("tokenizer_kwargs", {}))

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if tr.get("load_in_4bit", True) else None

    # Multimodal archs expose the LM via a different AutoModel: gemma-4 "unified" works
    # under AutoModelForCausalLM, but Mistral3/Ministral-3 (vision_encoder) needs
    # AutoModelForImageTextToText. We still do TEXT-ONLY SFT (processing_class=tokenizer).
    # Set `model_class: image_text_to_text` in the config for such models.
    if cfg.get("model_class") == "image_text_to_text":
        from transformers import AutoModelForImageTextToText as _ModelCls
    else:
        _ModelCls = AutoModelForCausalLM
    model = _ModelCls.from_pretrained(
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

    # "Capture everything": write a snapshot of config + dataset size next to the run (Windows-readable
    # if progress_file lives on /mnt/...). Makes every future run self-documenting.
    try:
        import json as _json, time as _time
        _md = os.path.dirname(cfg.get("progress_file") or "") or cfg["output_dir"]
        os.makedirs(_md, exist_ok=True)
        _name = os.path.splitext(os.path.basename(cfg.get("progress_file") or cfg["output_dir"]))[0]
        with open(os.path.join(_md, _name + ".meta.json"), "w", encoding="utf-8") as _f:
            _json.dump({"config": cfg, "dataset_rows": len(ds), "started": _time.time()},
                       _f, ensure_ascii=False, default=str, indent=2)
    except Exception as _e:
        print("run_meta could not be written:", _e)

    # completion_only_loss (loss ONLY on the answer): only set it if this TRL version supports it
    # AND the data is in prompt/completion format. For prompt/completion it's the default anyway.
    import inspect
    _extra = {}
    if "completion_only_loss" in inspect.signature(SFTConfig.__init__).parameters \
            and tr.get("completion_only_loss") is not None:
        _extra["completion_only_loss"] = tr["completion_only_loss"]
    # pad_to_multiple_of: forces a CONSTANT sequence shape per batch (e.g. 4096) so the
    # Triton-AMD flash kernel on gfx1201 compiles only ONCE (otherwise a recompile stall per length).
    if "pad_to_multiple_of" in inspect.signature(SFTConfig.__init__).parameters \
            and tr.get("pad_to_multiple_of") is not None:
        _extra["pad_to_multiple_of"] = tr["pad_to_multiple_of"]

    args = SFTConfig(
        output_dir=cfg["output_dir"],
        optim=tr["optim"],                                  # (1) adamw_torch
        assistant_only_loss=tr.get("assistant_only_loss", False),  # True needs a {% generation %} chat template; gemma-4's lacks it
        max_length=tr.get("max_length", 2048),
        per_device_train_batch_size=tr.get("per_device_batch_size", 1),
        gradient_accumulation_steps=tr.get("grad_accum", 8),
        learning_rate=float(tr["learning_rate"]),
        num_train_epochs=tr.get("epochs", 1),
        bf16=tr.get("bf16", True),
        gradient_checkpointing=tr.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tr.get("logging_steps", 5),   # =1 -> dashboard tile ticks ~every step
        save_strategy=tr.get("save_strategy", "no"),   # "steps" -> reboot-sichere Checkpoints
        save_steps=tr.get("save_steps", 500),
        save_total_limit=tr.get("save_total_limit", 3),
        seed=tr.get("seed", 42),
        report_to="none",
        **_extra,
    )

    # Pass the tokenizer explicitly so text-only SFT works on multimodal archs
    # (e.g. gemma-4 "unified") without pulling in the image Processor.
    callbacks = [ProgressCallback(cfg["progress_file"])] if cfg.get("progress_file") else []
    trainer = SFTTrainer(model=model, args=args, peft_config=peft_cfg, train_dataset=ds,
                         processing_class=tok, callbacks=callbacks)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"\n✅ LoRA adapter saved to: {cfg['output_dir']}")
    print("   Next: bash examples/gemma4-12b-qlora/export_gguf.sh")


if __name__ == "__main__":
    main()
