# Bring your own model — a tiny, end-to-end example

The smallest possible "I trained my own LLM" loop, on one AMD Radeon (RDNA4/gfx1201):
a ~0.8 B model, fine-tuned on data **you** generated. ~2 minutes.

```bash
# 1) make some data with a teacher (local Ollama = free; or --provider anthropic/openai)
python scripts/make_dataset.py generate --example --provider ollama --model llama3.1:8b \
    --n 20 --out data/example.jsonl
#    ^ opens a review page — keep/drop/edit, then it's ready

# 2) train YOUR model on it (QLoRA, 4-bit, the gfx1201 workarounds baked in)
python examples/gemma4-12b-qlora/train_qlora.py --config examples/byo-instruct/config.yaml
#    watch it live:  make demo   (the same dashboard engine, pointed at data/_runs)

# 3) the LoRA adapter lands in ./byo-instruct-adapter — merge + serve like the main example
```

Why `Qwen/Qwen3.5-0.8B`? It's the very model the [charts](../../charts/) crown as **"best &
smallest"** (a fine-tuned 0.8 B router beat the 6.9 B model — 95% vs 87%), it's tiny and trains in
a couple of minutes — a perfect first run. Swap `model_id` in [config.yaml](config.yaml) for any
model you like (gemma-4, Llama, Mistral, a bigger Qwen …); the recipe is the same. Full reuse
guide: [docs/reuse-your-own-model.md](../../docs/reuse-your-own-model.md).
