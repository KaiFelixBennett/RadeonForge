<div align="center">

<img src="assets/hero.png" alt="RadeonForge — fine-tune your own LLM on a Radeon GPU" width="100%">

<br>

[![GPU](https://img.shields.io/badge/GPU-RDNA4%20·%20gfx1201-ED1C24?style=flat-square&logo=amd&logoColor=white)](VERSIONS.md)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-22d3ee?style=flat-square)](docs/track-a-wsl2-rocm.md)
[![OS](https://img.shields.io/badge/OS-Windows%20WSL2%20·%20Linux-3b9cff?style=flat-square)](docs/paths-and-stability.md)
[![Python](https://img.shields.io/badge/Python-3.12-8b5cf6?style=flat-square&logo=python&logoColor=white)](RUNBOOK.md)
[![License: MIT](https://img.shields.io/github/license/KaiFelixBennett/RadeonForge?style=flat-square&color=10b981)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-10b981?style=flat-square)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/KaiFelixBennett/RadeonForge?style=flat-square&color=f59e0b)](https://github.com/KaiFelixBennett/RadeonForge/stargazers)

### Fine-tune your own LLM on an AMD Radeon GPU — the easy, tested way.

**No NVIDIA. No PhD. No guesswork.** A copy-paste path from a blank machine to a model *you trained yourself*, on the GPU you already own — plus a **reusable live training dashboard** you can drop into any project.

📖 [**Step-by-step (RUNBOOK)**](RUNBOOK.md) · 🔁 [**Use your own model**](docs/reuse-your-own-model.md) · 🧠 [**Never trained before? Start here**](docs/how-finetuning-works.md) · 🩺 [**If something breaks**](docs/troubleshooting.md)

</div>

---

## 🛠️ Train your own model — 5 commands

Fine-tuning has a reputation for being hard. On AMD it has a reputation for being *impossible*.
Neither is true anymore. Here's the whole path — each step is **one command**, and step 2
proves your GPU can train **before** you spend real time on it.

<div align="center">

<img src="assets/steps.png" alt="Your first fine-tune in 5 steps: setup, smoke, train, export, chat" width="100%">

</div>

```bash
git clone https://github.com/KaiFelixBennett/RadeonForge && cd RadeonForge
```

| # | Command | What it does | You'll see |
|---|---|---|---|
| **1** | `make setup` | Installs the ROCm + Python stack and **handles the known AMD traps for you** (the librocdxg bridge, the bitsandbytes & Triton install order). Run once, inside WSL2. | a green environment check |
| **2** | `make smoke` | A 50-step test run on a tiny model. This is the honest part: it **fails loudly** if your box *can't* train, so you never waste an hour. | `loss 5.78 → 0.15  ✓ this box can train` |
| **3** | `make train` | QLoRA fine-tunes the example model on the example data (or swap in your own). Watch it **live** in the dashboard. | the loss falling, in real time |
| **4** | `bash examples/gemma4-12b-qlora/export_gguf.sh` | Merges your adapter and packs it into a small **GGUF (Q4_K_M)** so it runs locally. | `23.8 GB → 6.9 GB` |
| **5** | `bash examples/gemma4-12b-qlora/serve.sh` | Serves **your** model on your own Radeon GPU, OpenAI-compatible. | a model you can chat with, ~117 tok/s |

<details>
<summary><b>▶ Watch the whole run (animated)</b></summary>

<div align="center"><img src="assets/terminal-cast.svg" alt="doctor → smoke → train → export → serve" width="84%"></div>
</details>

> **New to all of this?** [docs/how-finetuning-works.md](docs/how-finetuning-works.md) explains
> *why* each step exists, in plain language — no prior fine-tuning experience assumed. The full
> walkthrough from a blank Ubuntu is the [**RUNBOOK**](RUNBOOK.md).

No `make`? Every step is a single command — see the [Makefile](Makefile). On Windows the
training steps run inside WSL2; the dashboard runs anywhere.

---

## 📊 You trained this — and here's the proof

A falling loss is necessary but not enough, so RadeonForge always does a **controlled A/B**:
the *same* held-out prompts through the untouched base model and the one you just trained.
Here's the worked example (a query-router), measured on **one** AMD Radeon AI PRO R9700:

<div align="center">

<img src="assets/before-after.png" alt="Base vs fine-tuned: routing 79→100%, real training loss curve, ~117 tok/s" width="100%">

</div>

| | Before (base) | After (you trained it) |
|---|---|---|
| **Routing accuracy** (14 held-out) | 79 % | **100 %** |
| Query-type accuracy | 64 % | **100 %** |
| Avg output length | 88 tok | **71 tok (−19 %)** |
| Served speed (Q4_K_M, R9700) | — | **~117 tok/s** |

Every number is regenerated from raw A/B JSON by [`make charts`](examples/gemma4-12b-qlora/make_charts.py)
— nothing is hand-typed. It also **scales to the 12B** on the same card (~11 min to train,
6.9 GB Q4_K_M, ~53 tok/s, 86 % → 93 %). Full evidence: [**PILOT_REPORT.md**](PILOT_REPORT.md).

---

## 🎁 You also get a reusable live dashboard

Training in a black box is stressful. RadeonForge ships a **high-quality live dashboard** — one
Python file, **zero dependencies, works offline** — that you can drop into *any* training
project. It shows your run as it happens: live loss & tokens/s, a base-vs-fine-tuned **radar
fingerprint**, a pass/fail eval grid, GPU telemetry, and a data-driven roadmap.

<div align="center">

<img src="assets/dashboard/dashboard.gif" alt="Live training dashboard — loss + tokens/s, animated" width="90%">

</div>

**See it right now, with no GPU and no install** (it ships with a tiny simulator so you can
explore the whole UI in ~10 seconds):

```bash
make demo                       # → opens http://127.0.0.1:8765/gl
# no make? →  python scripts/demo_dashboard.py --open
# on Windows, just double-click  start_dashboard.bat
```

<div align="center">
<img src="assets/dashboard/panel-radar.png" alt="Base vs fine-tuned radar fingerprint" width="49%">
<img src="assets/dashboard/panel-results.png" alt="Before/after results chart" width="49%">
</div>

Point it at your own runs in one line — schemas + the "drop it into your project" recipe are in
[scripts/DASHBOARD.md](scripts/DASHBOARD.md) and [docs/reuse-your-own-model.md](docs/reuse-your-own-model.md).

<details>
<summary><b>The full <code>/gl</code> dashboard (one screenshot)</b></summary>

<div align="center"><img src="assets/dashboard/dashboard-gl.png" alt="Full RadeonForge dashboard" width="78%"></div>
</details>

---

## 🤔 Why this exists

Fine-tuning an LLM on an **AMD consumer/workstation GPU under Windows** is, as of 2026, still a
yak-shave. Official AMD tutorials assume datacenter Instinct (MI300X) on Linux; consumer
**RDNA** is "supported" at the runtime layer, but the *training* stack (bitsandbytes,
FlashAttention, paged optimizers) breaks in undocumented, arch-specific ways. The knowledge that
makes it work is scattered across forum posts, GitHub issues, and one-off blogs — which is
exactly why so many Radeon owners believe training "doesn't work on AMD." It does.

RadeonForge packages **one pinned, tested path** for the intersection that is genuinely unmet:

> **discrete RDNA4 (RX 9070 / Radeon AI PRO R9700, `gfx1201`) + Windows/WSL2 + reproducible QLoRA/LoRA + a worked Gemma-4 example + a smoke test.**

It's deliberately *not* "yet another ROCm wrapper" — the general case (Linux/Instinct, or APUs)
is already covered by [Unsloth](https://unsloth.ai),
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and
[kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning).
The value here is the **Windows/WSL2 + discrete-RDNA4 glue** and the **honest, dated workarounds**.

---

## 🧭 Will it work on my card? (the honest matrix, June 2026)

We'd rather be trusted than impressive, so here's the unvarnished version:

| Platform | Inference | **Training (fine-tune)** |
|---|---|---|
| **Native Windows + ROCm** | ✅ preview | ❌ **No ML training** (AMD: *"No ML training support"*) |
| **Native Windows + Vulkan** | ✅ (often fastest on RDNA4) | ❌ no backward-pass kernel upstream |
| **WSL2 + ROCm** | ✅ | ✅ **QLoRA/LoRA** — start here on Windows |
| **Native Linux + ROCm** | ✅ | ✅ **most stable — long-term home** |

- **Verified:** AMD Radeon AI PRO R9700 (RDNA4, `gfx1201`) — the numbers above are from this card.
- **Expected to work** (same `gfx1201`): RX 9070 / 9070 XT.
- **Expected with an arch-flag change**: RDNA3 (`gfx1100`, e.g. RX 7900 XTX) — *not yet tested*.

**Train on ROCm, not Vulkan.** On Windows → **WSL2 + ROCm** (you're likely 90 % there); for a
serious rig → **native Linux + ROCm**. Got it working on another card? A
[hardware report](../../issues/new/choose) is the single most useful thing you can contribute.
Full reasoning: [docs/paths-and-stability.md](docs/paths-and-stability.md).

---

## 🪤 The two AMD traps we already solved for you

Most "AMD can't train" horror stories come down to two silent failures on `gfx1201`. Both are
**baked into every config here**, and `make smoke` catches them in 60 seconds:

1. **Never use paged optimizers.** `paged_adamw_*` / `adamw_bnb_8bit` **silently corrupt**
   training (loss→0, grad_norm→NaN after step 1). We use `optim="adamw_torch"`. *(4-bit NF4 weight
   quant itself is fine — only the **paged** optimizer is broken.)*
2. **Use math SDPA attention.** FlashAttention doesn't compile for gfx1201, and AOTriton SDPA
   crashes during training. We disable both SDP backends and load with `attn_implementation="sdpa"`.

The full symptom → fix table is in [docs/troubleshooting.md](docs/troubleshooting.md).

---

## 🔁 Make it yours (your model, your data)

The whole stack is **project-agnostic**. To retarget it you change **3 things** — the dataset (a
chat JSONL), the config (`model_id` + paths), and the dashboard manifest — and keep the gfx1201
workarounds. ➡️ [**docs/reuse-your-own-model.md**](docs/reuse-your-own-model.md) is the 10-minute
path from *clone* to *my model, my data, my live dashboard*.

<details>
<summary><b>📈 More results — the policy insight, the efficiency frontier, throughput</b></summary>

<div align="center">
<img src="charts/02_accuracy_before_after.png" width="49%">
<img src="charts/03_policy_comparison.png" width="49%">
<img src="charts/10_the_inversion.png" width="49%">
<img src="charts/12_tokens_per_sec.png" width="49%">
</div>

All 13 charts are generated from the raw A/B JSON (`make charts`), so they can't drift from the
measured numbers. See [charts/](charts/) and [PILOT_REPORT.md](PILOT_REPORT.md).
</details>

---

## 🗂️ What's in the box

```
RadeonForge/
├─ Makefile · setup.sh        ← one-command setup / demo / smoke / train / charts
├─ RUNBOOK.md                 ← the full step-by-step (blank Ubuntu → served model)
├─ PILOT_REPORT.md            ← every measured result, in one place (the evidence file)
├─ VERSIONS.md                ← pinned, dated version matrix (perishable — re-verify)
├─ docs/                      ← setup, gemma-4 notes, troubleshooting, "use your own model"
├─ scripts/
│  ├─ smoke_test.py           ← "does the loss actually fall?" — the real guarantee
│  ├─ progress_dashboard.py   ← the reusable live dashboard engine (stdlib only)
│  └─ demo_dashboard.py       ← the no-GPU live demo (seeds data + simulates a run)
├─ examples/gemma4-12b-qlora/ ← the worked example: trainer, config, eval, export, charts
└─ charts/ · assets/          ← generated result charts · README visuals (regen: make assets)
```

---

## 🧪 Tested on · 🤝 Contributing · 📄 License

- **Tested on:** AMD Radeon AI PRO R9700 (RDNA4, `gfx1201`, 32 GB) · Windows 11 + WSL2 (Ubuntu
  24.04) · ROCm 7.2 · PyTorch 2.9.1+rocm7.2.0 · June 2026 — see [VERSIONS.md](VERSIONS.md).
- **Contributing:** this is meant to be handed to the community. The most valuable contribution
  is a **dated** report of a stack that works (or doesn't) on your card — *an entry without a date
  is worse than none.* See [CONTRIBUTING.md](CONTRIBUTING.md), and be honest about what *didn't*
  change after fine-tuning; that's what makes the wins credible.
- **Built on:** AMD ROCm, PyTorch-ROCm,
  [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes),
  [TRL/PEFT](https://github.com/huggingface/trl), [llama.cpp](https://github.com/ggml-org/llama.cpp).
- **License:** [MIT](LICENSE) — use it, fork it, ship it.

<div align="center"><br><sub>Fine-tuning on the GPU you already own. If RadeonForge made it click for you, a ⭐ helps the next Radeon owner find it.</sub></div>
