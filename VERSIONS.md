# Pinned version matrix

> ⚠️ **Versions on RDNA4 rot fast.** Everything below is dated. Treat it as "known-good as of this date", not "the only thing that works". Always run [`scripts/smoke_test.py`](scripts/smoke_test.py) after changing any version. A version claim without a date is worthless — keep the dates when you update.

## Track A — WSL2 + ROCm (verified 2026-06-15, AMD Radeon AI PRO R9700 / gfx1201)

| Component | Pinned version | Source |
|---|---|---|
| GPU / arch | Radeon AI PRO R9700 · RDNA4 · `gfx1201` · 32 GB | — |
| Windows host driver | **AMD Adrenalin 26.2.2 for WSL2** | AMD R9700 driver page |
| WSL distro | Ubuntu 24.04 (noble); 22.04 jammy also OK | — |
| ROCm | **7.2.x** (stable; latest 7.2.4, 2026-05-29) | `repo.radeon.com/amdgpu-install/7.2/` |
| amdgpu-install deb | `amdgpu-install_7.2.70200-1_all.deb` | repo.radeon.com |
| PyTorch | `torch 2.9.1+rocm7.2.0` (+ torchvision 0.24.0, torchaudio 2.9.0, triton 3.5.1) | `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` |
| Python | 3.12 | — |
| transformers | **5.8.0** (≥5.5.0 required for `gemma4`) | PyPI |
| peft | 0.19.1 | PyPI |
| accelerate | 1.11.0 (or 1.6.0 if you hit issues) | PyPI |
| trl | ≥0.20.0 (for `assistant_only_loss`) | PyPI |
| bitsandbytes | CI preview `1.33.7.preview` **or** source-build `-DBNB_ROCM_ARCH="gfx1201"` (stable tag 0.49.2) | bnb releases |

### Required env / kernel settings (Track A)
```bash
export HSA_ENABLE_DXG_DETECTION=1            # WSL GPU discovery via /dev/dxg
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Windows-side WSL kernel cmdline (reported needed to avoid training crashes):
#   amdgpu.cwsr_enable=0
```

### Non-obvious facts that bit people (re-verify before pinning)
- The **public** pip index `repo.amd.com/rocm/whl/gfx120X-all/` currently carries **only ROCm 7.13 *preview* wheels** — for stable 7.2 torch you must download from `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`.
- `amdgpu-install --usecase=wsl` was reportedly removed in amdgpu-install 30.x; if it errors use `--usecase=rocm,hip --no-dkms`. Never use DKMS in WSL.
- **bitsandbytes** from plain PyPI may ship without a working ROCm backend → use the preview wheel or compile. Verify with a 2-step NF4 optimizer run.

## Track B — Vulkan LoRA (QVAC fork) (verified 2026-06-15)

| Component | Pinned version | Source |
|---|---|---|
| Repo | `tetherto/qvac-fabric-llm.cpp` | github.com/tetherto/qvac-fabric-llm.cpp |
| Release tag | **b7349** (2026-03-31); master ahead (commits to 2026-05-28) | GitHub Releases |
| Vulkan SDK | 1.3.283.0 (pinned in `docs/build.md`; newer 1.4.x also works) | vulkan.lunarg.com |
| Build flag | `-DGGML_VULKAN=ON` | — |
| Verified LoRA models | Qwen3 (0.6/1.7/4B), **Gemma-3 (1/4B)** | QVAC HF blog |

> ⚠️ **Gemma-4 on Track B is unverified.** QVAC's tested set is Gemma-3/Qwen-3. Gemma-4 is a new "unified" arch (June 2026) — test on a tiny model first, or use Track A for Gemma-4.

## Gemma-4-12B (model facts, verified 2026-06-15)

| Fact | Value | Source |
|---|---|---|
| HF id (instruct) | `google/gemma-4-12B-it` (capital **B**) | HF API |
| HF id (base) | `google/gemma-4-12B` | HF API |
| License | Apache 2.0 | HF |
| Arch | `gemma4_unified` (encoder-free multimodal) | config.json |
| Chat-template turn tokens | `<|turn>` / `<turn|>` (**not** `<start_of_turn>`); roles user/assistant/system | tokenizer_config.json + chat_template.jinja |
| transformers min | ≥ 5.5.0 | transformers gemma4 doc |
| Official prebuilt GGUF | `ggml-org/gemma-4-12B-it-GGUF`, `unsloth/gemma-4-12b-it-GGUF`, `google/gemma-4-12B-it-qat-q4_0-gguf` | HF |
| Recommended sampling | `--temp 1.0 --top-p 0.95 --top-k 64` | model card |

See [docs/gemma4-notes.md](docs/gemma4-notes.md) for the export caveats (text-tower vs multimodal, converter bugs).
