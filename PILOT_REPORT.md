# Pilot report — ARIA HiRAG router on AMD RDNA4 (single source of truth)

**The evidence file.** Every measured result from the end-to-end pilot, in one place.
Model: **`gemma-4-E2B-it`** (risk-de-risking run before the 12B). Hardware:
**AMD Radeon AI PRO R9700** (RDNA4, gfx1201, 32 GB), **WSL2** + ROCm 7.2 +
PyTorch 2.9.1+rocm7.2.0. Date: **2026-06-15**. Task: HiRAG query-complexity
routing (free-text query → routing JSON with `target_level` 0/1/2).

> How to read this: each stage has the command, the artifact it produced, and the
> measured number. The headline proof is **Stage 6** (base vs fine-tuned A/B).

---

## Verdict (TL;DR)

| | Before training (base) | After training (fine-tuned) |
|---|---|---|
| **target_level accuracy** (14 held-out) | **79 %** (11/14) | **100 %** (14/14) |
| query_type accuracy | 64 % | 100 % |
| valid-JSON rate | 100 % | 100 % |
| avg output length | 88 tok | **71 tok (−19 %)** |
| serving speed (R9700, Q4_K_M) | — | **~117 tok/s** |

The full pipeline — **dataset → QLoRA → merge → GGUF → GPU serving** — is proven on
RDNA4. The base model's misses were all *domain-policy* cases; the tune injects our
routing convention. Detail: [`examples/gemma4-12b-qlora/PILOT_RESULTS.md`](examples/gemma4-12b-qlora/PILOT_RESULTS.md).

**It also scales to the 12B** on the same card (Stage 7): trained in ~11 min, served as a
6.9 GB Q4_K_M at ~53 tok/s; A/B **86 % → 93 %** accuracy, output **−42 %**.

---

## Stage 1 — GPU stack works
`scripts/doctor.sh` green; `scripts/smoke_test.py` (50-step 4-bit QLoRA on a tiny model):
**loss 5.78 → 0.15**. → the R9700 can actually train in 4-bit. Setup: [RUNBOOK §2–3](RUNBOOK.md).

## Stage 2 — Dataset (reviewed before training)
Task-A routing, **158 samples**, hand-reviewed + balanced:

| class / level | count |
|---|---|
| search / L2 | 77 |
| specific / L1 | 27 |
| overview / L0 | 26 |
| comparison / L0 | 16 |
| deep_dive / L2 | 12 |

> First pass was 88 % one class + mislabels + chatbot noise → cleaned to the above.
> Generators live in `custom-rag/finetune/{build_task_a_routing.py,review_task_a.py}`.

## Stage 3 — Training (QLoRA)
`gemma-4-E2B-it`, LoRA r16/α32, 3 epochs, batch 4, lr 2e-4, ~92 s on the R9700.

```
epoch 0.5: loss 2.86  →  1.0: 0.90  →  2.0: 0.28  →  3.0: 0.21
mean_token_accuracy: 0.62 → 0.96
```
Adapter → `/root/aria-pilot/e2b-task-a`. Trainer: [`examples/gemma4-12b-qlora/train_qlora.py`](examples/gemma4-12b-qlora/train_qlora.py).

## Stage 4 — Held-out check (transformers, base + adapter)
`test_routing.py`, 8 held-out queries: **8/8 target_level correct**, valid JSON, from the *short* prompt.

## Stage 5 — Merge → GGUF → serve on the GPU
- Merge adapter → fp16; convert → GGUF **f16 8.7 GB**; quantize → **Q4_K_M 3.2 GB** (exit 0).
- Built llama.cpp with the **HIP backend** for gfx1201 (`scripts/build_llamacpp_hip.sh`).
- Served on the R9700 (`-ngl 99`): server ready in **3–5 s**, **~117 tok/s**, model shown as `ROCm0: AMD Radeon AI PRO R9700`.
- `/completion` with token-id prompts: **4/4 correct**. `/v1/chat/completions` (+ bundled template): clean JSON, `finish_reason:"stop"`.

> ⚠️ **Serving caveat (carried into the 12B run):** llama.cpp's `--jinja` mis-renders
> Gemma-4's `<|turn>` template → model "thinks" (empty content) or echoes `<|turn>`.
> Weights are fine (proven in transformers). Fix: `--chat-template-file
> gemma4-chat-template.jinja --reasoning-budget 0`, or token-ids to `/completion`.
> See [docs/gemma4-notes.md](docs/gemma4-notes.md) → "Serving".

## Stage 6 — PROOF: base vs fine-tuned A/B (the control experiment)
Same 14 held-out queries through the **untouched base** and the **fine-tuned** model
(`ab_test.py`, greedy decode, identical prompts):

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| target_level accuracy | 79 % (11/14) | **100 % (14/14)** | +21 pp |
| query_type accuracy | 64 % | **100 %** | +36 pp |
| valid-JSON rate | 100 % | 100 % | — |
| avg output length | 88 tok | **71 tok** | −19 % |

The 3 base misses, all *domain-policy* (base uses generic reasoning; tune learned our convention):

| Query | Base → | Correct (tuned) → |
|---|---|---|
| „Liste alle verfügbaren Berichte auf" | L1 ✗ | **L0** ✓ |
| „Vergleiche die Aussagen aus beiden Verträgen" | L2 ✗ | **L0** ✓ |
| „Worin unterscheiden sich Bericht A und Bericht B?" | L1 ✗ | **L0** ✓ |

Full per-query outputs from both models: [`PILOT_RESULTS.md`](examples/gemma4-12b-qlora/PILOT_RESULTS.md)
· raw data: [`pilot_ab_results.json`](examples/gemma4-12b-qlora/pilot_ab_results.json).

## Stage 7 — Scaling to the 12B (same pipeline, same card)
Re-ran the *whole* pipeline on `gemma-4-12B-it` on the same R9700, config-only change.

| Step | Result |
|---|---|
| Training (QLoRA, GPU) | loss 3.32 → 0.34, ~11 min (30 steps @ ~23 s) |
| Merge → GGUF f16 → Q4_K_M | 23.8 GB → **6.9 GB** |
| Serving (R9700, `-ngl 99`, +template fix) | clean JSON, `finish_reason:stop`, **~53 tok/s**, ready in 12 s |
| A/B `target_level` accuracy | 86 % → **93 %** |
| A/B `query_type` accuracy | 71 % → **93 %** |
| A/B output length | 124 → **72 tok (−42 %)** |

Two findings worth keeping:
- **Arch gotcha:** the 12B is `model_type: gemma4_unified` (vs the E2B's `gemma4_audio`) — needed `transformers` 5.12 (`KeyError: 'gemma4_unified'` on 5.8). Same family ≠ same setup.
- **Bigger base, stronger prior:** the 12B base already hit 86 % and resisted one *policy-override* case (routing "compare two contracts" to detail). 158 samples flipped one of two comparison cases; the smaller E2B flipped both. More data would close it — but the **efficiency** gain is larger on the 12B (−42 % tokens). Detail: [`PILOT_RESULTS_12B.md`](examples/gemma4-12b-qlora/PILOT_RESULTS_12B.md).

---

## Artifacts (where everything lives)
| What | Path |
|---|---|
| Reproducible walkthrough | [`RUNBOOK.md`](RUNBOOK.md) |
| A/B evidence (generated) | `examples/gemma4-12b-qlora/PILOT_RESULTS.md` + `pilot_ab_results.json` |
| A/B generator | `examples/gemma4-12b-qlora/ab_test.py` |
| Trainer / config | `examples/gemma4-12b-qlora/{train_qlora.py,config.yaml}` |
| Serving (GPU) | `scripts/build_llamacpp_hip.sh` · `examples/gemma4-12b-qlora/{serve.sh,gemma4-chat-template.jinja}` |
| Pilot model files (WSL) | `/root/aria-pilot/` (adapter, merged, `e2b-routing-Q4_K_M.gguf`) |

## Next (real 12B run)
Same scripts, `BASE=google/gemma-4-12B-it`. Open: full dataset (Task A–F), D3 data
basis, frozen eval set. Plan: `custom-rag/FINETUNING_PLAN_ARIA.md`.
