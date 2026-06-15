# Which path is actually stable? (training on AMD, 2026)

> Researched 2026-06-15 against real-world evidence (GitHub activity, AMD docs, practitioner reports). This page exists because the obvious-sounding "just train on native Windows / Vulkan" idea does **not** hold up — and the data says where to actually invest.

## TL;DR

- **Train on ROCm, not Vulkan.**
- **Most stable long-term:** **native Linux + ROCm.**
- **Pragmatic on a Windows box today:** **WSL2 + ROCm** (install the **`rocdxg-roct`** package for the `librocdxg` bridge — any recent Adrenalin WSL2 driver works; **no downgrade**).
- **Vulkan = inference only** (and on RDNA4 often the *fastest* inference) — it **cannot train**.
- **Native-Windows ROCm training:** not yet — AMD's own docs say *"No ML training support."* Coming, but ~12 months early as of mid-2026.

## Stability ranking (for fine-tuning)

| Path | Train? | Stability | Notes |
|---|---|---|---|
| **Native Linux + ROCm** | ✅ | 🟢 best | most real RDNA4 success evidence; recommended long-term home |
| **WSL2 + ROCm** | ✅ | 🟡 works, fragile | install the `rocdxg-roct` package (`librocdxg`); WSL/driver updates can break it |
| Native Windows + ROCm | ❌ | — | AMD: *"No ML training support"* / "no backward pass". Inference-only preview |
| Vulkan (QVAC / llama.cpp) | ❌ | 🔴 not viable | no upstream backward-pass kernel; QVAC immature (see below). **Inference only.** |
| Native Windows + DirectML | ❌ | 🔴 dead | `torch-directml` in maintenance mode |

## Why Vulkan training is out (not just "unverified")

- **Upstream llama.cpp/ggml has no Vulkan `OUT_PROD` kernel** — that op *is* the matmul backward/gradient. No backward pass = no training. ggml's training code is currently broken on *all* backends (issue [#18805](https://github.com/ggml-org/llama.cpp/issues/18805)) and the maintainers are steering away from training contributions. llama.cpp is an **inference** project.
- **The QVAC/Tether Vulkan-LoRA fork** (the one path that adds a Vulkan backward pass) after ~6 months public: **108 stars, 0 community-filed issues, 0 independent AMD/Windows/Vulkan success reports**, fine-tune branch **~4.5 months stale**, **LoRA-only**, and **Gemma-4 unsupported** (an outsider patched it — and tested on NVIDIA/CUDA, not AMD/Vulkan). Single-vendor bus factor; abandonment risk moderate-to-high for the fine-tune use case.

→ Keep Vulkan for **inference** (LM Studio / llama.cpp Vulkan — excellent on RDNA4). Do **not** build a training workflow on it.

## The validation that matters for us (R9700 / gfx1201)

A fully documented **end-to-end QLoRA success on the exact card** — Level1Techs, 2026-06-03: **Qwen3-8B QLoRA on Radeon AI PRO R9700 (gfx1201), native Ubuntu 24.04.4, ROCm 7.2.1, PyTorch 2.9.1, bitsandbytes 0.49.2, PEFT 0.19.1, TRL** — ran cleanly (189 steps) **after defeating exactly the 5 gfx1201 pitfalls this repo already bakes in**:
1. paged optimizer corrupts state → use `adamw_torch`
2. flash/mem-efficient SDPA crash → force **math SDPA**
3. HF `Trainer`/`SFTTrainer` HIP launch failures → fall back to a **raw PyTorch loop**
4. chat-template label-masking quirks → use the model's own template
5. PEFT `modules_to_save` breaks tied embeddings → `target_modules="all-linear"`
plus `amdgpu.cwsr_enable=0`.

That report is independent confirmation that **RadeonForge's recipe is the right one** — and that **native Linux** is the most proven route.

## The `librocdxg` gotcha — a missing PACKAGE, not a driver problem

WSL+ROCm reaches the GPU via `/dev/dxg`, bridged by **`librocdxg.so`**. Important correction: this lib is **NOT shipped by the Windows AMD driver** and does **not** belong in `/usr/lib/wsl/lib`. It is a **separate open-source package** ([ROCm/librocdxg](https://github.com/ROCm/librocdxg)) that installs into **`/opt/rocm/lib`**, where the ROCm runtime `dlopen()`s it. `amdgpu-install --usecase=wsl` does **not** pull it, and there is **no apt repo** — install the prebuilt `.deb` (or build from source).

**Symptom (package missing):**
```
LoadLib(librocdxg.so) failed: librocdxg.so: cannot open shared object file
dlsym failed: .../libhsa-runtime64.so.1: undefined symbol: hsaKmtOpenKFD
```
**Fix (no driver change, no downgrade):**
```bash
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
sudo apt install ./rocdxg-roct_1.2.0_amd64.deb   # -> /opt/rocm/lib/librocdxg.so
export HSA_ENABLE_DXG_DETECTION=1                 # legacy, pre-7.13 ROCm
rocminfo | grep -i gfx                            # should now list gfx1201
```
You only need the Adrenalin **WSL2 driver present** (for `libdxcore.so` + `/dev/dxg`) — **any** recent version; the librocdxg matrix is intentionally driver-version-agnostic. **Verified 2026-06-15:** R9700 + Adrenalin **26.5.2** + ROCm 7.2.0 → installing `rocdxg-roct 1.2.0` made `rocminfo` list `gfx1201`. No downgrade.

## Sources
AMD ROCm Radeon limitations (7.2.1, "No ML training support"); Phoronix "AMD Improves GPU Support Under WSL With Production ROCDXG" (2026-03-30); ggml-org/llama.cpp `docs/ops.md` (no Vulkan OUT_PROD) + issue #18805; github.com/tetherto/qvac-fabric-llm.cpp (GitHub API, 2026-06-15); Level1Techs R9700 gfx1201 QLoRA writeup (2026-06-03); LukeLamb/rdna4-ready; promptinjection.net AMD Strix Halo fine-tune guide (2026-05-11). Versions perishable — re-verify.
