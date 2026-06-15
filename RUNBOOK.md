# RUNBOOK — fine-tune an LLM on AMD RDNA4 (R9700/gfx1201) in WSL2, end to end

A complete, **reproducible** walkthrough of what we actually ran to QLoRA-fine-tune
**gemma-4** on a **Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB)** under **WSL2** —
from a blank Ubuntu to a model that produces correct task output.

> Verified 2026-06-15. Worked example: a HiRAG "query router" (query → routing JSON).
> Result: **8/8** held-out queries classified correctly. New to fine-tuning? Read
> [docs/how-finetuning-works.md](docs/how-finetuning-works.md) first — it explains *why* each step exists.

Conventions: commands run **inside WSL2 Ubuntu as your user** unless noted. The venv is `~/.venvs/rdna4-train`.

---

## 0. The mental model (one paragraph)
Fine-tuning = show a base model thousands of `(input → desired output)` examples so it
learns the behaviour. We use **QLoRA**: load the base model in **4-bit** (small VRAM) and
train tiny **LoRA adapter** matrices on top (the base stays frozen). Steps: **(1)** working
GPU stack → **(2)** a dataset of examples → **(3)** train the adapter → **(4)** test on
held-out examples → **(5)** merge + export to GGUF to serve.

---

## 1. Hardware / OS
- GPU: **AMD Radeon AI PRO R9700** (RDNA4, `gfx1201`, 32 GB)
- Windows 11 + **WSL2 Ubuntu 24.04**, Adrenalin WSL2 driver installed (any recent version — provides `/dev/dxg`)
- **No NVIDIA GPU, no driver downgrade.** Training on **native Windows is not possible** (ROCm = no training there) → WSL2 (or native Linux).

## 2. Environment setup (the part everyone gets stuck on)

### 2a. ROCm 7.2 inside WSL (no kernel module — host owns the GPU)
```bash
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt update && sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=wsl,rocm,hip --no-dkms     # if "wsl" errors: --usecase=rocm,hip --no-dkms
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc && source ~/.bashrc
```

### 2b. ⚠️ The `librocdxg` gotcha — a MISSING PACKAGE, not a driver problem
`amdgpu-install --usecase=wsl` does **not** install the WSL↔GPU bridge `librocdxg`. Without it
`rocminfo` fails (`librocdxg.so: cannot open` / `undefined symbol: hsaKmtOpenKFD`). It is a
separate open-source package (NOT from the Windows driver, NOT on apt):
```bash
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
sudo apt install ./rocdxg-roct_1.2.0_amd64.deb     # -> /opt/rocm/lib/librocdxg.so
rocminfo | grep -i gfx                              # MUST list gfx1201
```

### 2c. Python venv + PyTorch-ROCm 7.2 (⚠️ needs `--find-links` for ROCm-Triton)
```bash
sudo apt install -y python3.12 python3.12-venv python3-pip git
python3.12 -m venv ~/.venvs/rdna4-train && source ~/.venvs/rdna4-train/bin/activate
pip install --upgrade pip wheel
# torch pulls a ROCm-specific triton that is NOT on PyPI -> point pip at the AMD wheel dir:
pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.9.1+rocm7.2.0  True  AMD Radeon AI PRO R9700
```

### 2d. Training libraries
```bash
pip install "transformers>=5.5.0" "peft" "accelerate" "trl" "datasets" "pillow" "pyyaml"
```

### 2e. ⚠️ bitsandbytes (4-bit) — install with `--no-deps` or it KILLS your ROCm torch
`--force-reinstall` on bnb pulls CUDA-torch and overwrites your ROCm build. Always:
```bash
pip install --no-deps "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"
python -c "import torch,bitsandbytes as bnb,bitsandbytes.functional as F; x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16); F.quantize_4bit(x); print('4-bit OK', bnb.__version__)"
```

## 3. Verify the environment BEFORE wasting hours
```bash
bash scripts/doctor.sh                 # checks driver/ROCm/torch/bnb/HF
python scripts/smoke_test.py           # 50-step 4-bit QLoRA on a tiny model; FAILS loudly if loss→0/NaN
python scripts/smoke_test.py --no-4bit # same in bf16-LoRA (no bitsandbytes)
```
A green smoke test = the GPU can train. (We saw loss 5.78 → 0.15.)

## 4. Build the dataset
Each training row is a chat with the desired output as the assistant turn:
```json
{"messages":[{"role":"system","content":"<task instruction>"},
             {"role":"user","content":"Analyze this query:\nWelche Dokumente kennst du?"},
             {"role":"assistant","content":"{\"query_type\":\"overview\",\"target_level\":0, ...}"}]}
```
Our ARIA routing dataset is generated + reviewed by scripts in the **custom-rag** repo:
- `finetune/build_task_a_routing.py` — builds the JSONL (gold seeds + cleaned real queries; noise filter + balance).
- `finetune/review_task_a.py` — renders a human-reviewable Markdown to eyeball the labels **before** training.
> Lesson: **review the labels first.** Our first pass was 88 % one class + had mislabels + chatbot noise; after cleaning it was balanced and correct.

## 5. Train (QLoRA)
Config: [`examples/gemma4-12b-qlora/config.yaml`](examples/gemma4-12b-qlora/config.yaml) (or the pilot `pilot_e2b_routing.yaml`). Trainer: [`examples/gemma4-12b-qlora/train_qlora.py`](examples/gemma4-12b-qlora/train_qlora.py).
```bash
HSA_ENABLE_DXG_DETECTION=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python examples/gemma4-12b-qlora/train_qlora.py --config <your-config>.yaml
```
The 3 baked-in gfx1201 workarounds (do not remove):
1. `optim="adamw_torch"` — **never** a paged optimizer (silently corrupts on gfx1201).
2. **math SDPA** (`enable_flash_sdp(False)` + `attn_implementation="sdpa"`) — flash-attn crashes.
3. `processing_class=tokenizer` — text-only SFT on gemma-4's multimodal "unified" arch (else it needs the image Processor).
Note: `assistant_only_loss` is **off** — gemma-4's chat template lacks `{% generation %}` markers, so we train on the full (short) sequence. Fine for short outputs.

Our pilot run (gemma-4-E2B-it, 158 samples, 3 epochs, ~92 s):
```
epoch 0.5: loss 2.86 → 1.0: 0.90 → 2.0: 0.28 → 3.0: 0.21   (mean_token_accuracy 0.62 → 0.96)
✅ LoRA adapter saved → /root/aria-pilot/e2b-task-a
```

## 6. Test that it WORKED (the important part)
A falling loss is necessary but not sufficient — test on **held-out** queries:
```bash
HSA_ENABLE_DXG_DETECTION=1 python examples/gemma4-12b-qlora/test_routing.py \
    --base google/gemma-4-E2B-it --adapter /root/aria-pilot/e2b-task-a
```
Our result: **8/8 target_level correct**, valid JSON, from the *short* prompt — e.g.
`"Welche Dokumente kennst du?" → {"query_type":"overview","target_level":0,...}`.

## 7. Merge → GGUF → serve (deployment)  *(next step — see export_gguf.sh)*
```bash
bash examples/gemma4-12b-qlora/export_gguf.sh    # merge LoRA → fp16 → convert → quantize Q4_K_M
```
Then serve the GGUF with llama.cpp and confirm it still emits the routing JSON.

## Gotchas recap
| Symptom | Fix |
|---|---|
| `librocdxg.so: cannot open` | install `rocdxg-roct` .deb (§2b) |
| `No matching distribution ... triton==…+rocm…` | add `--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` |
| `torch.cuda.is_available()` False after a pip install | something pulled CUDA-torch → reinstall ROCm wheel; install bnb with `--no-deps` |
| `Gemma4Processor requires the PIL library` | `pip install pillow` and/or pass `processing_class=tokenizer` |
| `chat template … missing {% generation %}` | set `assistant_only_loss=False` |
| loss → 0.0 / grad_norm NaN at step 1 | you used a **paged** optimizer → `optim="adamw_torch"` |
