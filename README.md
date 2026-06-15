# RadeonForge

**Fine-tune LLMs on AMD Radeon (RDNA4 / RDNA3) under Windows & WSL2 — reproducibly.**
QLoRA via ROCm (WSL2) · LoRA via Vulkan (native Windows) · worked Gemma-4-12B example · a smoke test that proves the loss actually falls.

> Status: **working draft, community-maintained.** Verified on an **AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB)** as of **June 2026**. Versions in this repo are *pinned and dated* — see [VERSIONS.md](VERSIONS.md) and re-verify before relying on them.

---

## Why this exists

Fine-tuning an LLM on an **AMD consumer/workstation GPU under Windows** is, as of 2026, still a yak-shave: official AMD tutorials assume datacenter Instinct (MI300X) on Linux; consumer **RDNA** is "supported" at the runtime layer but the *training* stack (bitsandbytes, FlashAttention, paged optimizers) breaks in undocumented, model- and arch-specific ways. The knowledge that makes it work is scattered across forum posts, GitHub issues and one-off blogs.

RadeonForge packages a **single, pinned, tested path** for the exact intersection that is genuinely unmet:

> **discrete RDNA4 (RX 9070 / Radeon AI PRO R9700, `gfx1201`) + Windows/WSL2 + reproducible QLoRA/LoRA + a Gemma-4 example + a smoke test.**

It is deliberately *not* "yet another ROCm wrapper". The general case (Linux/Instinct, or APUs) is already covered by [Unsloth](https://unsloth.ai), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning). The value here is the **Windows/WSL2 + discrete-RDNA4 glue** and the **honest, dated workarounds**.

## The honest hardware/OS matrix (June 2026)

| Platform | Inference | **Training (fine-tune)** |
|---|---|---|
| **Native Windows + ROCm** | ✅ preview | ❌ **No ML training** (AMD: *"No ML training support"*) |
| **Native Windows + Vulkan** | ✅ (often fastest on RDNA4) | ❌ **not a real path** — upstream Vulkan has no backward-pass kernel; QVAC fork immature (see below) |
| **WSL2 + ROCm** | ✅ | ✅ **QLoRA/LoRA** — install the `rocdxg-roct` package (`librocdxg`); any Adrenalin WSL2 driver |
| **Native Linux + ROCm** | ✅ | ✅ **most stable — recommended long-term home** |

**Takeaway:** **train on ROCm, not Vulkan.** On Windows today → **WSL2 + ROCm** (install the **`rocdxg-roct`** package for the `librocdxg` bridge — any recent Adrenalin WSL2 driver; no downgrade). For a serious long-term rig → **native Linux + ROCm**. Vulkan is the *inference* path (great on RDNA4) — it **cannot train** (no backward-pass kernels upstream). Full reasoning + evidence: **[docs/paths-and-stability.md](docs/paths-and-stability.md)**.

## Which path — training is ROCm, not Vulkan

> **Updated 2026-06-15 after researching real-world evidence** ([docs/paths-and-stability.md](docs/paths-and-stability.md)). The earlier "Vulkan as a 2nd training track" idea does **not** hold: upstream llama.cpp/ggml has **no Vulkan backward-pass (`OUT_PROD`) kernel**, and the QVAC Vulkan-LoRA fork has ~zero independent adoption + a stale fine-tune branch. **Vulkan = inference only.** The real choice is *where* you run ROCm:

| | **WSL2 + ROCm** (start here on Windows) | **Native Linux + ROCm** (long-term home) |
|---|---|---|
| OS | Windows + WSL2 (Ubuntu 24.04) | native Ubuntu 24.04 |
| Training | ✅ QLoRA / LoRA / SFT / DPO | ✅ same, most reliable |
| Stability | ⚠️ works; WSL/driver updates can break it | ✅ best (most real RDNA4 success evidence) |
| Prerequisite | install `rocdxg-roct` .deb → `librocdxg` (any Adrenalin WSL2 driver; **not** a downgrade) | none |
| Guide | [docs/track-a-wsl2-rocm.md](docs/track-a-wsl2-rocm.md) | same guide, native steps |

**Recommendation:** start on **WSL2 + ROCm** (you're likely 90 % there), move to **native Linux + ROCm** for serious/long-term runs. Use **Vulkan for inference only** ([docs/track-b-vulkan-qvac.md](docs/track-b-vulkan-qvac.md) is an experimental note, not a training recommendation).

## Quickstart (Track A, WSL2)

```bash
# 1. Set up the stack (one-time) — see docs/track-a-wsl2-rocm.md for the full pinned recipe
bash scripts/doctor.sh            # checks driver, ROCm, torch, bitsandbytes-ROCm, the GPU is seen

# 2. Prove training actually works on THIS box before committing hours (uses a tiny model)
python scripts/smoke_test.py      # 50 steps; FAILS loudly if loss→0/NaN (the gfx1201 paged-optimizer trap)

# 3. Run the worked example
python examples/gemma4-12b-qlora/train_qlora.py --config examples/gemma4-12b-qlora/config.yaml

# 4. Merge LoRA → GGUF (Q4_K_M) for local llama.cpp serving
bash examples/gemma4-12b-qlora/export_gguf.sh
```

## The two load-bearing gfx1201 workarounds (baked into every config here)

1. **Never use paged optimizers.** `paged_adamw_*` / `adamw_bnb_8bit` **silently corrupt** training on gfx1201 (loss→0, grad_norm→NaN after step 1). Use `optim="adamw_torch"`. *(NF4 4-bit weight quant itself is safe — only the **paged** optimizer is broken.)*
2. **Use math SDPA attention.** FlashAttention does not compile for gfx1201; AOTriton flash/mem-efficient SDPA crash during training. Disable both SDP backends and load with `attn_implementation="sdpa"`. (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is the documented Triton flag but is unstable for *training* on gfx1201 — prefer math SDPA.)

See [docs/troubleshooting.md](docs/troubleshooting.md) for the full table.

## Philosophy: the smoke test is the real reproducibility guarantee

Pinned versions rot fast on this hardware. RadeonForge's durable promise is **not** a version list — it is [`scripts/smoke_test.py`](scripts/smoke_test.py): a 50-step run that **fails loudly** if your environment hits the known silent-corruption traps. Run it first, every time, before a long job.

## Repo layout

```
RadeonForge/
├─ README.md                    ← you are here
├─ VERSIONS.md                  ← pinned, dated version matrix (perishable — re-verify)
├─ docs/
│  ├─ track-a-wsl2-rocm.md      ← full Track A setup (verified commands)
│  ├─ track-b-vulkan-qvac.md    ← full Track B setup
│  ├─ gemma4-notes.md           ← model id, <|turn> chat template, GGUF export
│  └─ troubleshooting.md        ← the gfx1201 gotcha → fix table
├─ scripts/
│  ├─ doctor.sh                 ← environment health check
│  └─ smoke_test.py             ← "does the loss actually fall?" sanity run
└─ examples/gemma4-12b-qlora/
   ├─ config.yaml               ← pinned versions + hyperparams
   ├─ train_qlora.py            ← Track A QLoRA trainer (workarounds baked in)
   ├─ export_gguf.sh            ← merge LoRA → convert → quantize Q4_K_M
   └─ sample_data.jsonl         ← tiny HF-chat-schema example dataset
```

## Tested on

- **GPU:** AMD Radeon AI PRO R9700 (RDNA4, `gfx1201`, 32 GB)
- **OS:** Windows 11 + WSL2 (Ubuntu 24.04) — Track A · native Windows — Track B
- **Date of verification:** June 2026 (see [VERSIONS.md](VERSIONS.md))

> Also expected to work on RX 9070 / 9070 XT (gfx1201) and, with arch flags changed, RDNA3 (gfx1100, e.g. RX 7900 XTX). Not yet tested there — PRs welcome.

## Contributing

This is meant to be handed to the community. If you get it working on another card / ROCm version, please open a PR updating [VERSIONS.md](VERSIONS.md) **with the date and your exact stack**. A version entry without a date is worse than none.

## Credits & sources

Built on the shoulders of: AMD ROCm, PyTorch-ROCm, [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [Hugging Face TRL/PEFT](https://github.com/huggingface/trl), [llama.cpp](https://github.com/ggml-org/llama.cpp), and the [QVAC/Tether Vulkan-finetune fork](https://github.com/tetherto/qvac-fabric-llm.cpp). Field reports that made the gfx1201 workarounds possible are cited in [docs/troubleshooting.md](docs/troubleshooting.md).

## License

[MIT](LICENSE).
