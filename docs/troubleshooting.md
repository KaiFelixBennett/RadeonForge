# Troubleshooting — gfx1201 / RDNA4 fine-tuning

The gotcha → fix table. Most of these are **silent** failures (no error, just garbage). Run [`scripts/smoke_test.py`](../scripts/smoke_test.py) to catch them fast.

| Symptom | Cause | Fix |
|---|---|---|
| **Loss → 0.0, grad_norm → NaN after step 1**, no error | Paged optimizer broken on gfx1201 (HIP paging unimplemented) | `optim="adamw_torch"` — **never** `paged_adamw_*` / `adamw_bnb_8bit` |
| `hipErrorLaunchFailure` mid-training | flash / mem-efficient SDPA crash on RDNA4 | Disable both SDP backends + `attn_implementation="sdpa"` (math). See snippet below |
| `flash-attn` build fails ("20 errors generated") | No CK FlashAttention for Wave32 RDNA4 | Don't build flash-attn; use math SDPA. (Triton flag `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is inference-only-ish, unstable for training) |
| `import bitsandbytes` ok but NF4 train corrupts / no ROCm backend | PyPI bnb lacks working ROCm backend | Install CI preview wheel `1.33.7.preview` or compile `-DBNB_ROCM_ARCH="gfx1201"`; set `BNB_BACKEND=rocm`, `AMDGPU_TARGETS="gfx1201"` |
| `torch.cuda.is_available()` is False | CUDA wheel installed, or GPU not seen in WSL | Install the ROCm torch wheels; `export HSA_ENABLE_DXG_DETECTION=1`; check `rocminfo \| grep gfx1201` |
| `rocminfo` → `librocdxg.so: cannot open` / no gfx1201 in WSL | `librocdxg` package missing (it is **not** from the driver) | install `rocdxg-roct` .deb (→ `/opt/rocm/lib`); `export HSA_ENABLE_DXG_DETECTION=1`; ensure an Adrenalin WSL2 driver provides `/dev/dxg` |
| `modprobe amdgpu` fails in WSL | You tried to install the kernel module in WSL | Don't — host owns the GPU. Install with `--no-dkms` |
| `amdgpu-install: unknown usecase wsl` | `wsl` usecase removed in amdgpu-install 30.x | Use `--usecase=rocm,hip --no-dkms` |
| HF `Trainer`/`SFTTrainer` HIP launch failure (raw loop works) | Trainer internals trip a HIP edge case | Reduce batch, set `PYTORCH_ALLOC_CONF=expandable_segments:True`; as last resort a minimal raw PyTorch loop |
| Random training crashes | Compute-wave-save-restore on RDNA4 | WSL kernel cmdline `amdgpu.cwsr_enable=0` |
| OOM on 12B QLoRA in 32 GB | Activations | `gradient_checkpointing=True`, batch 1 + grad-accum, shorter `max_length`, leave host VRAM headroom |
| PEFT error on tied weights | `modules_to_save` on lm_head+embed_tokens | Use `target_modules="all-linear"`, drop `modules_to_save` |
| GGUF model echoes literal `<|turn>` / hallucinates user turns | Early/broken embedded chat template | Re-pull latest official GGUF or pass `--chat-template-file` |
| Convert step fails on missing `tokenizer.model` | llama.cpp Gemma converter quirk (#19152) | Ensure merged dir has `tokenizer.json`+`tokenizer_config.json`; use current llama.cpp; or Unsloth `save_pretrained_gguf` |

## The two snippets you'll always need

```python
# Math SDPA (avoid the RDNA4 flash/mem-efficient crash)
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
# ... from_pretrained(..., attn_implementation="sdpa")
```

```bash
export HSA_ENABLE_DXG_DETECTION=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# host-side WSL kernel cmdline if you get random crashes: amdgpu.cwsr_enable=0
```

## Field reports these fixes come from (2026)
- Level1Techs — R9700 gfx1201 ROCm 7.2.1 QLoRA bug writeup (paged-optimizer corruption, SDPA crash → math-SDPA, Trainer issues, `amdgpu.cwsr_enable=0`).
- LLaMA-Factory issue #10511 — R9700 gfx1201 training (bnb compile-from-source, flash-attn build failure, `BNB_BACKEND`/`AMDGPU_TARGETS`).
- apollo-mg RDNA4 gfx1201 ROCm gist — Wave32 vs Wave64 FlashAttention rationale, Triton enable flag.
- bitsandbytes docs/releases — gfx1201 wheel targets, CI preview wheel, source-compile flags.

> These are community sources; the durable check is the smoke test, not any single report. Re-verify on your stack.
