# How fine-tuning works (and what we did) — a from-scratch explainer

Written so you can **understand it, reproduce it, and explain it to others.** No prior ML
background assumed. The concrete example throughout is our "query router".

---

## 1. Three ways to change what an LLM does
1. **Prompting** — you tell the model what to do in the input. Fast, but unreliable for strict formats and it costs tokens every call.
2. **RAG** (retrieval) — you *inject facts* into the prompt at runtime. Great for knowledge that changes; doesn't change the model's *behaviour*.
3. **Fine-tuning** — you *change the model's weights* by training on examples, so a behaviour becomes baked in. Best for **stable skills/formats** (e.g. "always answer as strict JSON"), not for facts.

We fine-tune **behaviour** (routing → JSON), and keep facts in RAG. That split is the core design rule.

## 2. What "training" actually is (the loop)
A language model predicts the next token. Training shows it `(input, desired output)` pairs and nudges the weights so its predictions match the desired output:

```
for each batch of examples:
    prediction = model(input)            # forward pass
    loss       = how wrong is it?         # cross-entropy vs the desired tokens
    gradients  = d(loss)/d(weights)       # backward pass (this needs the GPU's OUT_PROD kernel)
    optimizer.step()                      # nudge weights to reduce loss
```
- **loss** ↓ over time = the model is learning. (Ours: 2.86 → 0.21.)
- **mean_token_accuracy** ↑ = fraction of next-tokens it now predicts correctly. (Ours: 0.62 → 0.96.)
- **grad_norm** = size of the update; should stay finite & shrink. (NaN = something broke.)

> This is why training is harder than inference: inference only does the *forward* pass;
> training also needs the *backward* pass. (That's exactly why Vulkan can't train on AMD —
> it has no backward-pass kernel — and why we use ROCm.)

## 3. LoRA — train tiny adapters, not the whole model
A 12B model has 12 billion weights; updating all of them needs huge VRAM. **LoRA** (Low-Rank Adaptation) instead:
- **Freezes** the entire base model.
- Adds small `A·B` matrices (rank `r`, e.g. 16) next to the big weight matrices and trains **only those** — a tiny fraction of the parameters.
- Result is a small "**adapter**" file you stack onto the base model.

Cheap, fast, and you can keep many adapters for one base model.

## 4. QLoRA — LoRA on a 4-bit base
**QLoRA** = load the frozen base in **4-bit** (NF4 quantization → ~¼ the VRAM) and train the LoRA adapter in bf16 on top. This is what lets a 12B model fine-tune comfortably in 32 GB.
- 4-bit **weight** quantization on gfx1201 works fine.
- (Caveat we hit: 4-bit **paged optimizers** are broken on gfx1201 → we use `adamw_torch`.)

bf16-LoRA (no 4-bit) also works but the base then uses more VRAM.

## 5. The chat template — why format matters
Models are trained with special "turn" tokens. Gemma-4 uses `<|turn>user … <turn|>` /
`<|turn>model …` (NOT the old `<start_of_turn>`). You must format training data with the
**model's own** template (`tokenizer.apply_chat_template`) so that **training format == serving
format**. Get this wrong and the tuned model misbehaves at inference. We never hand-roll the
format — the tokenizer does it.

A related knob: **loss masking**. Ideally you compute loss only on the *assistant* tokens
(`assistant_only_loss`), so the model isn't "rewarded" for reproducing the prompt. That needs
`{% generation %}` markers in the template; gemma-4's lacks them, so for our **short**-prompt
routing task we train on the full sequence — fine because the output is what dominates.

## 6. What WE did, concretely (the routing example)
**Goal:** turn "which retrieval level does this query need?" from a 5 KB prompt + a 35B helper
into a small model that emits the answer as JSON from a tiny prompt.

1. **Dataset** (`build_task_a_routing.py`): each row = `system (short instruction)` + `user ("Analyze this query:\n…")` + `assistant (the routing JSON)`. We mixed **hand-labelled "gold" seeds** (balanced across the classes) with **real user queries** (de-noised + auto-labelled + mislabel-corrected). 158 rows.
2. **Review** (`review_task_a.py`): we read the labels first and fixed problems (skew, mislabels, chatbot noise). **Garbage in → garbage out.**
3. **Train** (`train_qlora.py`): QLoRA on gemma-4, 3 epochs, ~92 s. Loss fell to 0.21.
4. **Test** (`test_routing.py`): on **held-out** queries the tuned model got **8/8** target_levels right, as valid JSON, from the short prompt → the 5 KB prompt is no longer needed.

## 7. How you KNOW it worked (don't trust the loss alone)
- **Necessary:** loss goes down + accuracy goes up during training.
- **Sufficient:** the model produces **correct outputs on queries it never saw in training** (held-out test). A model can memorise the training set and still be useless — the held-out check is the real proof.
- For generative tasks, also eyeball that the **format** is right (valid JSON, no markdown fence, no `<think>`).

## 8. Glossary
- **Base model** — the pretrained LLM (e.g. gemma-4) before our training.
- **SFT** — Supervised Fine-Tuning: training on `(input → desired output)` pairs (what we did).
- **LoRA / adapter** — small trainable matrices added to a frozen base; the saved result.
- **QLoRA** — LoRA with the base loaded in 4-bit.
- **Epoch** — one full pass over the dataset (we did 3).
- **Loss / cross-entropy** — how wrong the predictions are; we minimise it.
- **Merge** — fold the LoRA adapter back into full weights (needed before GGUF export).
- **GGUF** — llama.cpp's model file format for fast local serving.
- **DPO** — a later technique that learns from *preferred vs rejected* answer pairs (Phase 2; reduces hallucination beyond SFT).

## 9. Where the code lives
| What | File |
|---|---|
| Build the dataset | your own data-prep step (builder → JSONL) |
| Review the dataset | your own data-prep step (renders the labels for a human check) |
| Train (QLoRA) | `RadeonForge/examples/gemma4-12b-qlora/train_qlora.py` |
| Config | `RadeonForge/examples/gemma4-12b-qlora/config.yaml` |
| Test after training | `RadeonForge/examples/gemma4-12b-qlora/test_routing.py` |
| Env health / smoke | `RadeonForge/scripts/doctor.sh`, `scripts/smoke_test.py` |
| Full reproduce | `RadeonForge/RUNBOOK.md` |
