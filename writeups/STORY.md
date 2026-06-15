# I fine-tuned Gemma-4 on an AMD Radeon under Windows/WSL2 — every silent trap, and a repo so you don't repeat my week

> A field report, not a tutorial-shaped ad. Real numbers, every dead end, and a
> repo ([RadeonForge](https://github.com/KaiFelixBennett/RadeonForge)) that takes
> you from a blank Ubuntu to a fine-tuned model **served on the GPU**.

## TL;DR
- **Hardware almost nobody documents for training:** AMD **Radeon AI PRO R9700** (RDNA4, `gfx1201`, 32 GB), **Windows 11 + WSL2**, ROCm 7.2. No NVIDIA. No driver downgrade.
- **What I built:** a QLoRA fine-tune of **Gemma-4** for a real task (a RAG query-complexity *router*: free-text question → routing JSON).
- **Did it actually work?** Yes — and I prove it with a base-vs-fine-tuned **A/B control**, not vibes: **79 % → 100 %** routing accuracy, **−19 %** output length, served at **~117 tok/s** on the Radeon. And the *same pipeline* runs the **12B** end-to-end on the same card (see "It scales").
- **The actual value:** the ~10 **silent** failures on this path and their fixes, distilled into a `doctor.sh` + a 50-step `smoke_test.py` so you catch them in 60 seconds instead of 6 hours.

This isn't "AMD is now as easy as NVIDIA." It's "AMD RDNA4 training is *possible and reproducible* — here is the exact minefield map."

---

## Why this exists
Every "fine-tune your own LLM" guide assumes CUDA. If you own a Radeon, you hit a wall of cryptic errors that each cost hours, most of which **fail silently** (no crash — just garbage results). I wanted the guide I couldn't find. So I kept a brutally honest log of everything that broke.

## The setup that almost broke me (the part worth reading)
Each of these is a real dead end I hit on `gfx1201` / RDNA4. Most produce **no error** — that's what makes them expensive.

1. **`rocminfo` fails: `librocdxg.so: cannot open`.** I nearly downgraded my GPU driver chasing this (and saw forum posts telling me to). Wrong. `librocdxg` (the WSL↔GPU bridge) is a **separate open-source package** that `amdgpu-install` doesn't pull — not a driver feature. One `.deb` and it works. *Lesson: when "they removed a feature in a newer version" feels wrong, it usually is.*
2. **Loss → 0.0, grad_norm → NaN after step 1.** No error. The cause: **paged optimizers silently corrupt** on gfx1201. Fix: `optim="adamw_torch"`, never `paged_adamw_*`.
3. **`flash-attn` won't build** (no CK FlashAttention for Wave32 RDNA4). Fix: math SDPA (`enable_flash_sdp(False)` + `attn_implementation="sdpa"`).
4. **`bitsandbytes` install nukes your GPU stack.** `pip install --force-reinstall bitsandbytes` quietly pulls **CUDA** torch over your ROCm torch → `torch.cuda.is_available()` flips to `False`. Fix: install bnb with `--no-deps`.
5. **`triton==…+rocm…` not on PyPI** → torch install fails. Fix: `--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`.
6. **Gemma-4 is multimodal.** `Gemma4Processor requires the PIL library` and the trainer wants an image processor. Fix: `pip install pillow` + pass `processing_class=tokenizer` to train the text tower only.
7. **`assistant_only_loss` blows up** — Gemma-4's chat template lacks `{% generation %}` markers. Fix: train on the full (short) sequence.
8. **E2B and 12B are *different architectures*.** `gemma-4-E2B-it` is `model_type: gemma4_audio`; `gemma-4-12B-it` is `gemma4_unified`. My transformers knew the first, not the second (`KeyError: 'gemma4_unified'`). *Same model family ≠ same setup* — the 12B needs a newer transformers.

I turned this list into [`scripts/doctor.sh`](../scripts/doctor.sh) (env health) + [`scripts/smoke_test.py`](../scripts/smoke_test.py) (a 50-step 4-bit run that **fails loudly** if loss→0/NaN). Green smoke test = the GPU can actually train. That one check would have saved me most of the week.

## Proving it worked — A/B, not vibes
A falling loss proves nothing about the *task*. So I ran the **same held-out queries through the untouched base model and the fine-tuned one** ([`ab_test.py`](../examples/gemma4-12b-qlora/ab_test.py) writes the report + raw JSON automatically):

| Metric | Base (no training) | Fine-tuned |
|---|---|---|
| routing accuracy (`target_level`) | **79 %** | **100 %** |
| `query_type` accuracy | 64 % | 100 % |
| valid-JSON rate | 100 % | 100 % |
| avg output length | 88 tok | **71 tok** |

The interesting part isn't the number — it's **why** the base missed. All three errors were *domain-policy* cases: the base routed "compare two contracts" to detail-level retrieval (generic, reasonable), but **our** RAG convention is to start from document summaries. The base model wasn't dumb; it didn't know our policy. **That's what fine-tuning injects — and now I can measure it.**

## A harder eval — where it gets honest (and interesting)
My first eval was too easy: many queries (detail look-ups) are ones where the base model and our routing policy already *agree*. So I built a deliberately hard set (21 queries) stressing the cases where they *diverge* — document comparisons, listings, whole-document summaries.

| | Base accuracy | Fine-tuned | "compare two docs" cases | output length |
|---|---|---|---|---|
| **E2B** | 62 % | 81 % | **0/5 → 5/5** | −18 % |
| **12B** | 76 % | 81 % | 0/5 → 1/5 | −41 % |

Three findings I didn't expect, all useful:
1. **Every base model gets all 5 document-comparisons wrong** — both E2B and 12B route "compare these two contracts" to detail retrieval, because generically it *sounds* like deep work. Our convention is to start from summaries. The single clearest "domain policy the base can't guess."
2. **The smaller model is *easier* to steer.** E2B (weaker priors) learned the comparison rule completely (5/5). The 12B (stronger priors) fixed only 1/5 from the same 158 examples — yet its `query_type` accuracy hit 100 %, i.e. it *correctly labels* them "comparison" but won't override its depth instinct. Counterintuitive but real: to teach a policy that contradicts a model's instinct, the bigger model needs more data.
3. **The eval earns its keep.** The E2B fine-tune fixed comparisons but *regressed* on whole-document summaries — a query type my 158-sample set barely covered. Not a failure; the eval telling me exactly what data to add next.

Takeaway for anyone fine-tuning: build your eval around where the base and your target behaviour *disagree*, not where they happen to agree — otherwise you're measuring luck. And don't assume bigger = easier to steer. Evidence: [`PILOT_RESULTS_HARD_E2B.md`](../examples/gemma4-12b-qlora/PILOT_RESULTS_HARD_E2B.md) · [`PILOT_RESULTS_HARD_12B.md`](../examples/gemma4-12b-qlora/PILOT_RESULTS_HARD_12B.md).

## The serving trap that ate an afternoon
Training worked; serving the GGUF looked broken. Through llama.cpp's OpenAI endpoint the model either drifted into a "Thinking Process:" monologue (empty output) or, with reasoning disabled, echoed `<|turn>model<|turn>…` forever.

The weights were **fine** — I confirmed clean JSON from the merged model in transformers. The culprit: **llama.cpp's `--jinja` engine mis-renders Gemma-4's `<|turn>` chat template.** Fix: ship a 12-line known-good template via `--chat-template-file` (+ `--reasoning-budget 0`). Bundled as [`gemma4-chat-template.jinja`](../examples/gemma4-12b-qlora/gemma4-chat-template.jinja). *Lesson: when serving looks broken, prove the weights in transformers first — it isolates a template bug from a model bug in minutes.*

## What's in the repo (so you can reuse it)
[**RadeonForge**](https://github.com/KaiFelixBennett/RadeonForge) — MIT.
- [`RUNBOOK.md`](../RUNBOOK.md) — blank Ubuntu → trained → **served on the GPU**, exact commands + every gotcha inline.
- `scripts/doctor.sh`, `scripts/smoke_test.py` — catch the silent failures in 60 s.
- `examples/gemma4-12b-qlora/` — `train_qlora.py` (the 3 baked-in gfx1201 workarounds), `ab_test.py` (one-command proof), `serve.sh` + the chat-template fix, `build_llamacpp_hip.sh` (GPU serving on gfx1201).
- [`docs/troubleshooting.md`](../docs/troubleshooting.md) — the gotcha→fix table, sourced from real RDNA4 field reports.
- [`PILOT_REPORT.md`](../PILOT_REPORT.md) — every measured result in one place.

## Honest caveats
- This is a **pilot/validation** on a *narrow* task (routing), deliberately. The point was to de-risk the **mechanics** on RDNA4, not to claim a frontier result.
- Numbers are from one run on one box; the durable check is the smoke test + A/B, both reproducible — re-verify on your stack.
- The RDNA4 fixes draw on community field reports (Level1Techs, LLaMA-Factory issues, bitsandbytes docs); credited in the repo.

## It scales: same pipeline, 12B, same single card
I re-ran the *entire* pipeline on **gemma-4-12B-it** on the same R9700 — no code changes beyond the config file:
- **Trained** in ~11 min on the GPU (QLoRA, loss 3.3 → 0.34).
- **Merged → GGUF → Q4_K_M = 6.9 GB**, **served at ~53 tok/s** on the Radeon — clean routing JSON through the OpenAI endpoint (with the template fix), server ready in 12 s.
- **A/B (base vs fine-tuned):** routing accuracy **86 % → 93 %**, `query_type` **71 % → 93 %**, output length **−42 %** (124 → 72 tokens).

Two honest, useful lessons fell out of the 12B run:
1. **A trap nobody warns you about:** the 12B is a *different architecture type* than the E2B (`gemma4_unified` vs `gemma4_audio`) and needed a newer `transformers` (`KeyError: 'gemma4_unified'`). Same model family ≠ same setup.
2. **Bigger base = stronger prior = harder to bend.** The 12B base already scored 86 % and "knows" that comparing two contracts deserves detailed retrieval. On those rare *policy-override* cases, 158 examples flipped one of two; the smaller E2B (weaker priors) flipped both to 100 %. The fix isn't magic — it's more data. But note the efficiency win is *bigger* on the 12B (**−42 %** tokens): even when the base is already accurate, the fine-tune makes it dramatically more concise and consistent. (Evidence: [`PILOT_RESULTS_12B.md`](../examples/gemma4-12b-qlora/PILOT_RESULTS_12B.md).)

The point stands: **train → quantize → serve a 12B end-to-end on one consumer-class AMD card.**

If this saves you even one of those dead ends, the repo did its job. Issues/PRs welcome.
