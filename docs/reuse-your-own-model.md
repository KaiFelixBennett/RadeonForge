# Reuse RadeonForge for your own model & dataset

Everything here is built to be **lifted into your own project**. The worked example is a
query-router, but nothing is hard-coded to it. This is the 10-minute path from "clone" to
"my model, my data, my dashboard."

> TL;DR — change **3 things**: the dataset (a chat JSONL), the config (`model_id` +
> paths), and the dashboard manifest (title + tracks). The gfx1201 workarounds stay baked in.

---

## 1. Your dataset — one chat per line (JSONL)

Each row is a conversation whose **last assistant turn is the behaviour you want to teach**:

```json
{"messages":[
  {"role":"system","content":"<your task instruction>"},
  {"role":"user","content":"<an input>"},
  {"role":"assistant","content":"<the desired output>"}
]}
```

`SFTTrainer` applies the model's own chat template for you — **do not** hand-roll prompt
strings. Review the labels *before* training (a wrong-but-confident dataset trains a
wrong-but-confident model).

## 2. Your config — copy and edit 4 lines

Copy [`examples/gemma4-12b-qlora/config.yaml`](../examples/gemma4-12b-qlora/config.yaml) and change:

```yaml
model_id: <your-base-model>          # e.g. google/gemma-4-12B-it, Qwen/..., mistralai/...
dataset: /path/to/your_train.jsonl
output_dir: /path/to/your-adapter
progress_file: /path/to/_runs/your-run.json   # <- lights up the live dashboard
```

Then train with the **same trainer** (workarounds already baked in):

```bash
python examples/gemma4-12b-qlora/train_qlora.py --config your_config.yaml
```

Multimodal base (vision encoder, e.g. Mistral-3/Ministral-3)? set `model_class: image_text_to_text`
in the config — text-only SFT still works via `processing_class=tokenizer`.

## 3. Your dashboard — copy the manifest, point it at your data

The dashboard (`scripts/progress_dashboard.py`) is **project-agnostic** — it only reads
files. Copy [`scripts/dashboard.tracks.example.json`](../scripts/dashboard.tracks.example.json),
edit `title` / `subtitle` / `flow` / `tracks`, then:

```bash
python scripts/progress_dashboard.py \
  --data-dir ./your_out --tracks ./your.tracks.json --open
```

What lights up, and from where (all plain files, read live each request):

| Panel | Source file | Who writes it |
|---|---|---|
| Live loss + tokens/s + HUD | `<runs-dir>/<run>.json` | the `ProgressCallback` in the trainer (set `progress_file`) |
| Scorecard table + radar | `<data-dir>/scorecard.json` | your eval script |
| Before/after line chart | `<data-dir>/dashboard_results.json` | your eval script |
| Live eval pass/fail grid | `<data-dir>/_eval/<tag>.json` | your eval script |
| Flow + roadmap status | the `--tracks` manifest (`done_if` / `active_if`) | derived live from the data above |
| Dataset progress | `*.jsonl` line counts vs `target` | your data-prep step |

Schemas for `scorecard.json` / `dashboard_results.json` / the manifest are in
[scripts/DASHBOARD.md](../scripts/DASHBOARD.md). Want to see them populated before wiring
your own? Run `python scripts/demo_dashboard.py --seed-only` and read the files it writes
into `examples/dashboard-demo/` — that's a complete, working set you can copy and adapt.

### Optional: a one-line wrapper in your project

```python
# your_project/dashboard.py — pre-fill the paths so teammates just run `python dashboard.py`
import runpy, sys
sys.argv = ["progress_dashboard.py", "--data-dir", "./out",
            "--tracks", "./your.tracks.json", "--open"]
runpy.run_path("/path/to/RadeonForge/scripts/progress_dashboard.py", run_name="__main__")
```

## 4. Keep these (they're why it trains at all on gfx1201)

The trainer bakes in the two load-bearing RDNA4 workarounds — **don't remove them**:

1. `optim="adamw_torch"` — never a paged optimizer (silent loss→0 / grad NaN on gfx1201).
2. math SDPA (`enable_flash_sdp(False)` + `attn_implementation="sdpa"`) — flash-attn crashes.

See [docs/troubleshooting.md](troubleshooting.md) for the full trap → fix table, and
[VERSIONS.md](../VERSIONS.md) for the pinned, dated stack.

## 5. Serve it

Merge → GGUF → llama.cpp (HIP) is in [`examples/gemma4-12b-qlora/export_gguf.sh`](../examples/gemma4-12b-qlora/export_gguf.sh)
and [`serve.sh`](../examples/gemma4-12b-qlora/serve.sh). For architectures llama.cpp can't
run yet, `scripts/serve_transformers.py` exposes an OpenAI-compatible endpoint from the
merged HF model.
