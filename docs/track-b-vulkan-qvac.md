# Track B — Vulkan LoRA (QVAC fork), native Windows, no ROCm

> 🛑 **Not a recommended training path (reviewed 2026-06-15).** Vulkan is **inference-only** in practice: upstream llama.cpp/ggml has **no Vulkan backward-pass (`OUT_PROD`) kernel**, and the QVAC fork that adds one has **~zero independent adoption**, a **fine-tune branch ~4.5 months stale**, is **LoRA-only**, and had **no Gemma-4 support** (an outsider patched it — tested on NVIDIA, not AMD/Vulkan). **Train on ROCm (Track A / native Linux); use Vulkan only for inference.** Kept here as an experimental reference. See [paths-and-stability.md](paths-and-stability.md).

LoRA fine-tuning over the **Vulkan** API using the [QVAC/Tether llama.cpp fork](https://github.com/tetherto/qvac-fabric-llm.cpp). Runs on **native Windows** (or any OS), vendor-agnostic, works **directly on GGUF**. Verified 2026-06-15 (tag `b7349`).

## When to use Track B
- You want to stay on **native Windows** (no WSL2/ROCm).
- You already live in **llama.cpp / GGUF**.
- LoRA (not 4-bit QLoRA) is enough.

> ⚠️ **Gemma-4 caveat:** QVAC's verified set is **Gemma-3 / Qwen-3** (and TinyLlama, LLaMA-3.2-1B). **Gemma-4** is a brand-new "unified" arch (June 2026) and is **not yet confirmed** here — test on a tiny model first. For Gemma-4-12B specifically, prefer **Track A**.

## 0) Prerequisites (Windows)
- Visual Studio (Desktop C++ workload) + CMake
- **Vulkan SDK 1.3.283.0** (pinned in the fork; current 1.4.x also works) — https://vulkan.lunarg.com/sdk/home#windows
- AMD Adrenalin driver (provides the Vulkan ICD — **no ROCm/HIP needed**)

## 1) Build with Vulkan
```sh
git clone https://github.com/tetherto/qvac-fabric-llm.cpp
cd qvac-fabric-llm.cpp
git checkout b7349                 # pin for reproducibility (master is ahead, has newer fixes)

cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j

.\build\bin\llama-cli --list-devices   # confirm the AMD GPU is listed
```

## 2) LoRA fine-tune
Dataset = JSONL, HF chat schema, one object per line:
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

```sh
.\build\bin\llama-finetune-lora ^
  -m base_model.gguf ^
  -f dataset.jsonl ^
  -ngl 999 -c 512 -b 512 -ub 512 ^
  --lora-rank 16 --lora-alpha 32 ^
  --lora-modules "attn_q,attn_k,attn_v,attn_o" ^
  --assistant-loss-only ^
  --learning-rate 1e-5 --lr-scheduler cosine --warmup-ratio 0.03 ^
  --checkpoint-save-steps 100 --checkpoint-save-dir .\checkpoints ^
  --output-adapter adapter.gguf ^
  -fa off
```

- `-fa off` is **mandatory** on Vulkan.
- `--lora-modules` options: `attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down,embed,output,all`.
- `--assistant-loss-only` masks system/user tokens.
- `--auto-resume` / `--resume-from PATH` resume from `checkpoints/` (holds adapter + optimizer.gguf + metadata).

## 3) Use / merge / export
```sh
# Attach at inference time:
.\build\bin\llama-cli -m base_model.gguf --lora adapter.gguf -ngl 999

# Merge adapter INTO the base → standalone GGUF:
.\build\bin\llama-export-lora -m base_model.gguf --lora adapter.gguf -o merged.gguf
```

## Caveats
- **LoRA-only.** No documented 4-bit QLoRA (no NF4 base + adapter). Int4/int8 training paths exist but are experimental — use an **f16** base for stability.
- **OUT_PROD is fp16 on Vulkan** (the op behind the LoRA backward pass). fp16 accumulation can hurt stability → if you see NaNs/poor convergence, use an f16 base, modest batch sizes, f32 optimizer state.
- **Memory:** optimizer state is stored too (`optimizer.gguf`). LLaMA-3.2-1B fine-tune "works with 24 GB"; for bigger models lower `-c/-b/-ub` or partial offload `-ngl 99`.
- **Docs drift:** master is ahead of `b7349`; re-check `examples/training/README.md` + `docs/build.md` on your exact checkout, and confirm the JSONL key is `messages`.

Sources: [repo](https://github.com/tetherto/qvac-fabric-llm.cpp) · [build.md](https://github.com/tetherto/qvac-fabric-llm.cpp/blob/master/docs/build.md) · [training README](https://github.com/tetherto/qvac-fabric-llm.cpp/blob/master/examples/training/README.md) · [HF engineering blog](https://huggingface.co/blog/qvac/fabric-llm-finetune)
