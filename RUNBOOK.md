# RUNBOOK — fine-tune an LLM on AMD RDNA4 (R9700/gfx1201) in WSL2, end to end

A complete, **reproducible** walkthrough of what we actually ran to QLoRA-fine-tune
**gemma-4** on a **Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB)** under **WSL2** —
from a blank Ubuntu to a model that produces correct task output.

> Verified 2026-06-15. Worked example: a "query router" (query → routing JSON).
> Result: **8/8** held-out queries correct in transformers, then **merged → GGUF (Q4_K_M) →
> served on the GPU, 4/4 correct, ~117 tok/s**. New to fine-tuning? Read
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
The routing dataset for this example was generated + reviewed by a small data-prep step (kept in your own data repo — not shipped here). The pattern that mattered:
- a **builder** script that emits the JSONL (gold seeds + cleaned real queries; noise filter + class balance).
- a **reviewer** script that renders a human-readable Markdown so you can eyeball the labels **before** training.
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
✅ LoRA adapter saved → /root/router-pilot/e2b-task-a
```

### 5a. Watch progress live — the reusable dashboard (optional)
Tool: [`scripts/progress_dashboard.py`](scripts/progress_dashboard.py) — one file, stdlib only, offline
(no CDN). Serves an HTML page + JSON API at `http://127.0.0.1:8765/`, auto-refreshing every 3 s.
Two generic sources:
- **Training runs** ← `<runs-dir>/*.json`, written by the `ProgressCallback` in `train_qlora.py`.
  Enable it by setting **`progress_file: <runs-dir>/<name>.json`** in your training config — the
  callback then writes live `step / total_steps / epoch / loss / status` (works across `/mnt/...` from WSL).
- **Datasets** ← either a `--tracks` manifest (`{title, tracks:[{label,file,target,group}]}`) **or**
  auto-discovery of all `*.jsonl` in `--data-dir` (line count = progress, for data-generation runs).
```bash
python scripts/progress_dashboard.py --runs-dir /root/router-pilot/_runs --open   # just training runs
python scripts/progress_dashboard.py --data-dir ./data --tracks ./dashboard.tracks.json --open
```
Reuse in any project: copy nothing — point `--data-dir`/`--runs-dir`/`--tracks` at that project.
(Pattern: drop a thin wrapper in your project that `runpy`s this engine with your data dir +
tracks manifest pre-filled — see [docs/reuse-your-own-model.md](docs/reuse-your-own-model.md).)

**One-command on any machine (no install — pure stdlib, Python 3.8+):**
```bash
cp scripts/dashboard.tracks.example.json my.tracks.json   # edit title/subtitle/pipeline/tracks
python scripts/progress_dashboard.py --data-dir ./out --tracks my.tracks.json --open
```
**Models are configurable, not baked in:** the engine knows nothing about any specific model.
- *Student* (what you fine-tune) = `model_id` in your training YAML.
- *Teacher* (if you distill) = your data-gen script's choice (e.g. an Ollama tag) — the dashboard
  doesn't run it.
- *Display* = the manifest `subtitle`, e.g. `"Teacher: <x> · Student: <y>"` — shown under the title.
So no project-specific names live in RadeonForge; they only live in *your* manifest.

## 6. Test that it WORKED (the important part)
A falling loss is necessary but not sufficient — test on **held-out** queries:
```bash
HSA_ENABLE_DXG_DETECTION=1 python examples/gemma4-12b-qlora/test_routing.py \
    --base google/gemma-4-E2B-it --adapter /root/router-pilot/e2b-task-a
```
Our result: **8/8 target_level correct**, valid JSON, from the *short* prompt — e.g.
`"Welche Dokumente kennst du?" → {"query_type":"overview","target_level":0,...}`.

**Prove it changed something — base vs fine-tuned A/B.** A score means little without a control: run the *same* held-out queries through the **untouched base** model and the fine-tuned one, side by side:
```bash
HSA_ENABLE_DXG_DETECTION=1 python examples/gemma4-12b-qlora/ab_test.py \
    --base google/gemma-4-E2B-it --finetuned /root/router-pilot/e2b-merged \
    --out-md examples/gemma4-12b-qlora/PILOT_RESULTS.md \
    --out-json examples/gemma4-12b-qlora/pilot_ab_results.json
```
This writes the full evidence (every output from both models + metrics) to [`PILOT_RESULTS.md`](examples/gemma4-12b-qlora/PILOT_RESULTS.md) / `pilot_ab_results.json`. Our pilot: base **79 %** → tuned **100 %** target_level accuracy, **−19 %** output length. The base's misses were all *domain-policy* cases (it routes "compare two contracts" to L2 detail; our convention is L0 summaries) — exactly the knowledge the tune injects.

## 7. Merge → GGUF → serve (deployment) — verified end to end

### 7a. Merge the LoRA adapter into the base (fp16)
`convert_hf_to_gguf.py` cannot read PEFT adapters, so merge first (CPU is fine):
```bash
python - <<'PY'
import torch; from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE="google/gemma-4-E2B-it"; ADAPTER="/root/router-pilot/e2b-task-a"; OUT="/root/router-pilot/e2b-merged"
base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)   # converter needs tokenizer.json/_config
print("merged ->", OUT)
PY
```
**Sanity-check the merge in transformers before converting** — re-run `test_routing.py` style inference on the merged dir. If it's clean here but broken after serving, the bug is the *template*, not your weights.

### 7b. Build llama.cpp with the HIP backend (for GPU serving)
The plain CPU `cmake -B build` does **not** build `llama-server` and is slow for inference. Build the GPU backend once:
```bash
bash scripts/build_llamacpp_hip.sh         # -> llama.cpp/build-hip/bin/{llama-server,llama-cli,llama-quantize}
```

### 7c. Convert + quantize → GGUF (Q4_K_M)
```bash
python llama.cpp/convert_hf_to_gguf.py /root/router-pilot/e2b-merged --outtype f16 \
  --outfile /root/router-pilot/e2b-f16.gguf
llama.cpp/build-hip/bin/llama-quantize /root/router-pilot/e2b-f16.gguf \
  /root/router-pilot/e2b-Q4_K_M.gguf Q4_K_M
# E2B: 8.7 GB f16 -> 3.2 GB Q4_K_M
```

### 7d. Serve & verify ⚠️ THE CHAT-TEMPLATE GOTCHA
llama.cpp's `--jinja` engine **mis-renders Gemma-4's `<|turn>` template** — the model either drifts into a "Thinking Process:" monologue (empty `content`) or, with `--reasoning-budget 0`, echoes `<|turn>model<|turn>...`. The merged weights are fine (you proved that in 7a); only the server's template engine is broken. **Two verified fixes:**

```bash
# FIX A — OpenAI endpoint with the bundled, known-good template:
MODEL=/root/router-pilot/e2b-Q4_K_M.gguf bash examples/gemma4-12b-qlora/serve.sh
# then:
curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"system","content":"<SYS>"},
              {"role":"user","content":"Analyze this query:\nWelche Dokumente kennst du?"}],
  "max_tokens":160,"temperature":0}'
# -> {"...":"...","content":"{\"query_type\":\"overview\",\"target_level\":0,...}"}  finish_reason:"stop"
```
```python
# FIX B — template-free: format client-side, POST token ids to /completion:
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)["input_ids"]
requests.post("http://127.0.0.1:8099/completion",
              json={"prompt":[int(i) for i in ids], "n_predict":160, "temperature":0})
```
Our pilot result: **4/4 held-out routing queries correct on the GPU** (R9700), `finish_reason:"stop"`, ~117 tok/s.

> For the 12B: identical steps with `BASE=google/gemma-4-12B-it` and your 12B adapter; `serve.sh` defaults to a 12B filename. Same template fix applies.

## Gotchas recap
| Symptom | Fix |
|---|---|
| `librocdxg.so: cannot open` | install `rocdxg-roct` .deb (§2b) |
| `No matching distribution ... triton==…+rocm…` | add `--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` |
| `torch.cuda.is_available()` False after a pip install | something pulled CUDA-torch → reinstall ROCm wheel; install bnb with `--no-deps` |
| `Gemma4Processor requires the PIL library` | `pip install pillow` and/or pass `processing_class=tokenizer` |
| `chat template … missing {% generation %}` | set `assistant_only_loss=False` |
| loss → 0.0 / grad_norm NaN at step 1 | you used a **paged** optimizer → `optim="adamw_torch"` |
| served model "thinks" / echoes `<|turn>` | llama.cpp `--jinja` mis-renders Gemma-4 template → `--chat-template-file gemma4-chat-template.jinja` (§7d) |
