# Track A — WSL2 + ROCm + QLoRA

Full, pinned setup for **4-bit QLoRA / LoRA** fine-tuning on **AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB)** under **WSL2 (Ubuntu)**. Verified 2026-06-15. See [VERSIONS.md](../VERSIONS.md) for the version table; re-verify before relying.

> Native Windows + ROCm cannot train (AMD: *"No ML training support"*). Training = WSL2 or native Linux. If WSL2 blocks hard, native Ubuntu on the same box is the fallback (perf within noise).

## 0) Windows host (once, outside WSL)

1. Install **any recent AMD Adrenalin WSL2 driver** (R9700 driver page) — it provides `libdxcore.so` + `/dev/dxg`. **Driver version is NOT the blocker** (26.5.x is fine; no downgrade). The WSL↔GPU bridge `librocdxg` is a **separate package** (`amdgpu-install --usecase=wsl` does NOT include it) — install it inside WSL:
   ```bash
   wget https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
   sudo apt install ./rocdxg-roct_1.2.0_amd64.deb     # -> /opt/rocm/lib/librocdxg.so
   ```
   Symptom if missing: `rocminfo` → `librocdxg.so: cannot open`. Details: [paths-and-stability.md](paths-and-stability.md).
2. In PowerShell:
   ```powershell
   wsl --install -d Ubuntu-24.04
   wsl --update
   wsl --shutdown
   ```
3. *(Recommended, if you hit mid-training crashes)* add the WSL kernel param `amdgpu.cwsr_enable=0` via `.wslconfig`.

## 1) Install ROCm 7.2.x inside WSL

Do **not** install the amdgpu kernel module in WSL — the Windows host owns the GPU via `/dev/dxg`.

```bash
# Ubuntu 24.04 (noble); for 22.04 use .../ubuntu/jammy/...
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=wsl,rocm,hip --no-dkms
#   ^ if "--usecase=wsl" errors (removed in amdgpu-install 30.x), use: --usecase=rocm,hip --no-dkms

sudo usermod -aG render,video $USER
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
source ~/.bashrc

# Verify (MUST show gfx1201):
rocminfo | grep -i gfx
rocm-smi
```

## 2) Python venv + PyTorch (ROCm 7.2, stable)

The public `repo.amd.com/rocm/whl/gfx120X-all/` index currently carries **only 7.13 preview** wheels. For **stable 7.2**, download the manylinux wheels:

```bash
sudo apt install -y python3.12 python3.12-venv python3-pip build-essential cmake git
python3.12 -m venv ~/.venvs/rdna4-train && source ~/.venvs/rdna4-train/bin/activate
pip install --upgrade pip wheel

cd /tmp
BASE=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
wget $BASE/torch-2.9.1%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
wget $BASE/torchvision-0.24.0%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
wget $BASE/torchaudio-2.9.0%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
wget $BASE/triton-3.5.1%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
pip install ./torch-2.9.1+rocm7.2.0-cp312-cp312-linux_x86_64.whl \
            ./torchvision-0.24.0+rocm7.2.0-cp312-cp312-linux_x86_64.whl \
            ./torchaudio-2.9.0+rocm7.2.0-cp312-cp312-linux_x86_64.whl \
            ./triton-3.5.1+rocm7.2.0-cp312-cp312-linux_x86_64.whl

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.9.1+rocm7.2.0 True AMD Radeon AI PRO R9700
```

> **Preview alternative** (ROCm 7.13, bleeding edge, less battle-tested):
> ```bash
> pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ \
>   "torch==2.11.0+rocm7.13.0" "torchvision==0.26.0+rocm7.13.0" "torchaudio==2.11.0+rocm7.13.0"
> ```

## 3) Fine-tuning libraries

```bash
pip install "transformers==5.8.0" "peft==0.19.1" "accelerate==1.11.0" "trl>=0.20.0" "datasets"
# transformers>=5.5.0 is REQUIRED to load gemma4. If accelerate 1.11.0 misbehaves, pin 1.6.0.
```

### bitsandbytes (the weak link) — pick one

```bash
# OPTION A — prebuilt CI/preview wheel that includes the ROCm backend (fastest):
pip install --force-reinstall \
  https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl

# OPTION B — compile for gfx1201 (guaranteed):
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1201" -DCMAKE_HIP_ARCHITECTURES="gfx1201" -S .
make -j$(nproc) && pip install .
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

**Immediately validate** bitsandbytes works (not just imports):
```bash
python ../RadeonForge/scripts/smoke_test.py   # or: python scripts/smoke_test.py
```

## 4) The two gfx1201 workarounds (already baked into the example)

```python
# (a) NO paged optimizers — they silently corrupt training on gfx1201.
TrainingArguments(optim="adamw_torch", ...)        # NOT paged_adamw_* / adamw_bnb_8bit

# (b) Math SDPA attention — flash/mem-efficient SDPA crash during training.
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="sdpa")
```

Plus:
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```
And in PEFT use `target_modules="all-linear"`; **avoid** `modules_to_save` on tied-weight models (lm_head+embed_tokens).

## 5) Train + export

```bash
python examples/gemma4-12b-qlora/train_qlora.py --config examples/gemma4-12b-qlora/config.yaml
bash   examples/gemma4-12b-qlora/export_gguf.sh
```

VRAM: a 12B in 4-bit QLoRA fits comfortably in 32 GB with `gradient_checkpointing=True` + `bf16`. WSL shares VRAM accounting with the Windows host — leave headroom.

## Known gotchas
See [troubleshooting.md](troubleshooting.md). The big ones: HF `Trainer` can throw HIP launch failures where a raw PyTorch loop succeeds; `amdgpu.cwsr_enable=0` may be mandatory; the bitsandbytes ROCm backend is the most common silent failure.
