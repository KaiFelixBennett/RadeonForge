# Charts — pilot results, visualised

All generated from the **raw A/B result JSONs** by
[`../examples/gemma4-12b-qlora/make_charts.py`](../examples/gemma4-12b-qlora/make_charts.py)
(reproducible, no hand-entered numbers). Regenerate:

```bash
python examples/gemma4-12b-qlora/make_charts.py
```

| File | What it shows | Best used for |
|---|---|---|
| `01_training_loss.png` | E2B + 12B QLoRA loss curves, converge in 2–11 min on one R9700 | "it really trains on AMD" |
| `02_accuracy_before_after.png` | Routing accuracy base → fine-tuned (hard set): E2B 62→81 %, 12B 76→81 % | the core before/after proof |
| `03_policy_comparison.png` | Comparison queries learned (of 5): base 0, E2B **5/5**, 12B 1/5 | the headline insight: small model learns a contradictory policy better |
| `04_speed_vs_accuracy.png` | Same 81 % accuracy, E2B at ~117 tok/s & 3.2 GB vs 12B ~53 tok/s & 6.9 GB | "small = more usable" |
| `05_output_length.png` | Output tokens base vs fine-tuned (−18 % to −42 %) | efficiency / cost |

## Ideas for more charts (data is already in the JSONs)
- **Per-category accuracy** (overview / comparison / specific / deep_dive / search) base vs fine-tuned — shows *where* it helped vs regressed (whole-doc summaries).
- **Easy vs hard eval gap** — why a hard eval matters (the gap widens on the hard set).
- **Training time / cost** — minutes + (rough) kWh per run, "fine-tune for the price of a coffee."
- **Accuracy vs training-set size** — would need extra runs; motivates the full Task A–F dataset.
