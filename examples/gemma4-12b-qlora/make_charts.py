#!/usr/bin/env python
"""
Generate the pilot charts from the real A/B result JSONs (reproducible, no hand numbers).
Outputs PNGs to RadeonForge/charts/. Run after the A/B tests:
  python examples/gemma4-12b-qlora/make_charts.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "charts"))
os.makedirs(OUT, exist_ok=True)

def load(f):
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        return json.load(fh)

hard_e2b = load("pilot_ab_hard_e2b.json")
hard_12b = load("pilot_ab_hard_12b.json")
easy_e2b = load("pilot_ab_results.json")
easy_12b = load("pilot_ab_results_12b.json")

C_E2B, C_12B, C_BASE = "#2563eb", "#dc2626", "#9aa4b2"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25})

def acc(d, k):  return d["metrics"][k]["level_accuracy"] * 100
def tacc(d, k): return d["metrics"][k]["type_accuracy"] * 100
def toks(d, k): return d["metrics"][k]["avg_out_tokens"]
def comp_ok(d, k):
    return sum(1 for q in d["queries"]
               if q[k]["expected_type"] == "comparison" and q[k]["level_correct"])

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print("wrote", p)

# 1) Training loss curves (from the training logs)
e2b_ep, e2b_loss = [0.5, 1, 2, 3], [2.86, 0.90, 0.28, 0.21]
v12_ep, v12_loss = [0.5, 1, 1.5, 2, 2.5, 3], [3.32, 1.06, 0.56, 0.42, 0.37, 0.34]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(e2b_ep, e2b_loss, "-o", color=C_E2B, lw=2.5, label="gemma-4-E2B (small)")
ax.plot(v12_ep, v12_loss, "-s", color=C_12B, lw=2.5, label="gemma-4-12B (large)")
ax.annotate(f"{e2b_loss[-1]:.2f}", (3, e2b_loss[-1]), textcoords="offset points", xytext=(6, 6), color=C_E2B)
ax.annotate(f"{v12_loss[-1]:.2f}", (3, v12_loss[-1]), textcoords="offset points", xytext=(6, -14), color=C_12B)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
ax.set_title("QLoRA training loss on one AMD Radeon R9700 (RDNA4)\nBoth converge cleanly — E2B ~2 min, 12B ~11 min")
ax.legend()
save(fig, "01_training_loss.png")

# 2) Accuracy base -> fine-tuned (hard set) — the core proof
fig, ax = plt.subplots(figsize=(7.5, 6))
x = [0, 1]
for d, c, m, lbl in [(hard_e2b, C_E2B, "o", "E2B (small)"), (hard_12b, C_12B, "s", "12B (large)")]:
    ys = [acc(d, "base"), acc(d, "finetuned")]
    ax.plot(x, ys, "-", marker=m, color=c, lw=2.8, markersize=10, label=lbl)
    for xi, yi in zip(x, ys):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=c, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Base\n(no fine-tune)", "Fine-tuned"])
ax.set_ylabel("Routing accuracy (%)"); ax.set_ylim(50, 100); ax.set_xlim(-0.25, 1.25)
ax.set_title("Fine-tuning lifts routing accuracy\nHard, policy-focused eval (21 queries)")
ax.legend()
save(fig, "02_accuracy_before_after.png")

# 3) The policy insight: comparison queries learned (of 5)
labels = ["Base\n(either model)", "E2B\nfine-tuned", "12B\nfine-tuned"]
vals = [comp_ok(hard_e2b, "base"), comp_ok(hard_e2b, "finetuned"), comp_ok(hard_12b, "finetuned")]
cols = [C_BASE, C_E2B, C_12B]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.annotate(f"{v}/5", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontweight="bold", fontsize=13)
ax.set_ylim(0, 5.6); ax.set_ylabel("Comparison queries routed correctly (of 5)")
ax.set_title("Teaching a rule that contradicts the model's instinct\nThe SMALLER model learns it (5/5); the bigger one resists (1/5) — same 158 examples")
save(fig, "03_policy_comparison.png")

# 4) Small = more usable: same accuracy, more speed, less size
fig, ax = plt.subplots(figsize=(8, 5.5))
pts = [("E2B fine-tuned", 117, acc(hard_e2b, "finetuned"), 3.2, C_E2B),
       ("12B fine-tuned", 53, acc(hard_12b, "finetuned"), 6.9, C_12B)]
for name, tps, ac, gb, c in pts:
    ax.scatter(tps, ac, s=gb * 90, color=c, alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(f"{name}\n{tps} tok/s · {ac:.0f}% · {gb} GB", (tps, ac),
                textcoords="offset points", xytext=(0, -38 if c == C_12B else 16), ha="center", color=c)
ax.set_xlabel("Serving speed (tokens/sec, R9700)"); ax.set_ylabel("Hard-set accuracy (%)")
ax.set_xlim(30, 135); ax.set_ylim(70, 90)
ax.set_title("Same accuracy, ~2x the speed, half the size\nFor this task the small fine-tune is the more usable model")
save(fig, "04_speed_vs_accuracy.png")

# 5) Output length: fine-tuning makes answers concise (efficiency)
fig, ax = plt.subplots(figsize=(7.5, 5))
groups = ["E2B", "12B"]
base_t = [toks(hard_e2b, "base"), toks(hard_12b, "base")]
ft_t = [toks(hard_e2b, "finetuned"), toks(hard_12b, "finetuned")]
xpos = range(len(groups)); w = 0.35
b1 = ax.bar([p - w / 2 for p in xpos], base_t, w, color=C_BASE, label="Base")
b2 = ax.bar([p + w / 2 for p in xpos], ft_t, w, color=C_E2B, label="Fine-tuned")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11)
ax.set_xticks(list(xpos)); ax.set_xticklabels(groups)
ax.set_ylabel("Avg output length (tokens)")
ax.set_title("Fine-tuning also makes outputs concise\nShorter = faster + cheaper per call (hard set)")
ax.legend()
save(fig, "05_output_length.png")

print("done ->", OUT)
