# Training data 101 — teacher, student, and how to feed the model

New to fine-tuning? This page explains the words and shows you how to make a dataset in a few
minutes — no prior experience assumed. The companion page
[how-finetuning-works.md](how-finetuning-works.md) explains the *training* itself.

---

## The two words you need: **student** and **teacher**

- **Student** = the model **you train** (e.g. `gemma-4`). Fine-tuning nudges *its* weights so it
  behaves the way you want. This is what runs on your Radeon at the end.
- **Teacher** = an *optional* stronger model that **writes or labels examples for you** (a
  helper, not the thing you ship). Using a strong model to create training data for a smaller
  one is called **knowledge distillation**. The teacher can be:
  - **local** — a model on your own GPU via [Ollama](https://ollama.com) (free, private), or
  - **cloud** — an API like Claude or OpenAI (higher quality, costs money + needs a key).

You don't *have* to use a teacher. If you already have good `(input → output)` pairs, just
[convert](#3-convert--you-already-have-data) them. The teacher only saves you the work of
writing examples by hand.

> Plain version: **the student learns by example.** You (optionally with a teacher's help)
> assemble a pile of "when the user says X, the ideal answer is Y" examples, you *review* them,
> and training makes the student imitate them.

---

## What the data must look like

The student learns from a **chat-JSONL** file: one JSON object **per line**, each a short
conversation whose **last assistant turn is the behaviour you want to teach**:

```json
{"messages":[
  {"role":"system","content":"You are a concise, friendly assistant."},
  {"role":"user","content":"How do I reverse a list in Python?"},
  {"role":"assistant","content":"Use slicing: my_list[::-1] returns a reversed copy."}
]}
```

Rules of thumb (these matter more than people think):

| Rule | Why |
|---|---|
| **Quality > quantity** | 150 clean, correct examples beat 5,000 noisy ones. A wrong-but-confident dataset trains a wrong-but-confident model. |
| **One behaviour, many phrasings** | Vary the wording/topics so the model learns the *behaviour*, not specific sentences. |
| **Review before you train** | Always eyeball the labels first — our tool opens a review page for exactly this. |
| **Hold out a test set** | Keep ~10–20 examples *out* of training to honestly measure if it worked (the README's A/B). |
| **The model applies its own chat template** | Give plain `messages`; do **not** hand-format `<|turn|>`/special tokens — the trainer does that. |

Background reading we found useful: Weights & Biases'
[Preparing a Dataset for Instruction Tuning](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2),
Hugging Face's [fine-tuning cookbook](https://huggingface.co/learn/cookbook), and DigitalOcean's
[How to Create Data for Fine-Tuning LLMs](https://www.digitalocean.com/community/tutorials/how-to-create-llm-finetuning-dataset).

---

## The tool: `scripts/make_dataset.py`

One stdlib script, **three ways to get to a chat-JSONL**, each followed by a **review page**
(open in a browser, keep/drop/edit, download the cleaned set). Try it instantly with the
**offline mock teacher** (no model, no key) — then switch `--provider`:

```bash
make data        # = generate a sample set with the offline teacher + open the review page
```

Pick your teacher with `--provider`: `ollama` (local), `anthropic`, `openai`, or `mock` (offline test).

### 1. Generate — *"describe your task, I'll draft the data"*
You have an idea but no data yet. Describe the task; the teacher writes N diverse examples.

```bash
# local teacher (free): pull a model first ->  ollama pull llama3.1
python scripts/make_dataset.py generate \
  --task "answer beginner Python questions clearly" \
  --system "You are a concise Python tutor." \
  --n 60 --provider ollama --model llama3.1 --out data/python_sft.jsonl
```

### 2. Label — *"I have the questions, write the answers"*
You already have inputs (one per line, or a `.jsonl` with a `user` field); the teacher writes
the ideal output for each.

```bash
python scripts/make_dataset.py label --inputs my_questions.txt \
  --system "You are a helpful support agent." \
  --provider anthropic --model claude-opus-4-8 --out data/support_sft.jsonl
```
*(Cloud providers read `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` from your environment.)*

### 3. Convert — *"you already have data"*
A CSV / JSONL / JSON of pairs → chat-JSONL. **No teacher needed.** Columns are auto-detected
(`question/answer`, `instruction/response`, `prompt/completion`, …) or set them explicitly.

```bash
python scripts/make_dataset.py convert --in pairs.csv \
  --user-col question --assistant-col answer \
  --system "You are a helpful assistant." --out data/sft.jsonl
```

### Always: review before you train
Every mode writes `<name>.review.html` and opens it. Keep/drop/edit each example, then
**Download cleaned JSONL** — everything runs in your browser, your data never leaves the page.
Re-open it anytime:

```bash
python scripts/make_dataset.py review --in data/sft.jsonl
```

---

## Then train on it

Point your config's `dataset:` at the cleaned file and run the normal flow:

```yaml
# examples/gemma4-12b-qlora/config.yaml  (copy + edit)
dataset: data/sft.clean.jsonl
```
```bash
make train        # QLoRA on your data — watch it live in the dashboard
```

Full reuse walkthrough: [reuse-your-own-model.md](reuse-your-own-model.md). How the training step
actually works: [how-finetuning-works.md](how-finetuning-works.md).
