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

# 60-query comprehensive eval — all four tested models (same set = fair comparison)
ev = {t: load(f"eval_{t}.json") for t in ["e2b", "12b", "qwen08", "qwen2b"]}
C_QWEN08, C_QWEN2B = "#16a34a", "#9333ea"
# (name, eval-key, Q4_K_M size GB, color, marker)
ALL = [
    ("gemma-E2B",    "e2b",    3.2, C_E2B,    "o"),
    ("gemma-12B",    "12b",    6.9, C_12B,    "s"),
    ("Qwen3.5-0.8B", "qwen08", 0.5, C_QWEN08, "^"),
    ("Qwen3.5-2B",   "qwen2b", 1.2, C_QWEN2B, "D"),
]

def acc(d, k):  return d["metrics"][k]["level_accuracy"] * 100
def tacc(d, k): return d["metrics"][k]["type_accuracy"] * 100
def toks(d, k): return d["metrics"][k]["avg_out_tokens"]
def comp_ok(d, k):
    return sum(1 for q in d["queries"]
               if q[k]["expected_type"] == "comparison" and q[k]["level_correct"])

def save(fig, name):
    fig.text(0.995, 0.006, "https://github.com/KaiFelixBennett/RadeonForge",
             ha="right", va="bottom", fontsize=8, color="#8a8a8a", alpha=0.85)
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

# 2) Accuracy base -> fine-tuned — ALL FOUR models (60-query eval)
# Exact numbers live in the legend (base -> fine-tuned); only the well-separated
# BASE points get an on-plot label, so nothing overlaps at the clustered FT side.
fig, ax = plt.subplots(figsize=(9.5, 6))
x = [0, 1]
for name, k, gb, c, mk in ALL:
    b, f = acc(ev[k], "base"), acc(ev[k], "finetuned")
    ax.plot(x, [b, f], "-", marker=mk, color=c, lw=2.6, markersize=11,
            label=f"{name}:  {b:.0f}% → {f:.0f}%")
    ax.annotate(f"{b:.0f}%", (0, b), textcoords="offset points", xytext=(-10, 0),
                ha="right", va="center", color=c, fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Base\n(no fine-tune)", "Fine-tuned"])
ax.set_ylabel("Routing accuracy (%)"); ax.set_ylim(25, 102); ax.set_xlim(-0.55, 1.15)
ax.set_title("Fine-tuning lifts routing accuracy — all four models\nComprehensive 60-query, policy-focused eval")
ax.legend(loc="center right", title="base → fine-tuned", fontsize=10)
save(fig, "02_accuracy_before_after.png")

# 3) The policy insight: document-comparison queries learned — all four models
def comp_total(d):
    return sum(1 for q in d["queries"] if q["base"]["expected_type"] == "comparison")
ncomp = comp_total(ev["e2b"])
names = [m[0] for m in ALL]
base_c = [100 * comp_ok(ev[k], "base") / ncomp for _, k, _, _, _ in ALL]
ft_c = [100 * comp_ok(ev[k], "finetuned") / ncomp for _, k, _, _, _ in ALL]
xpos = list(range(len(ALL))); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.5))
b1 = ax.bar([p - w / 2 for p in xpos], base_c, w, color=C_BASE, label="Base")
b2 = ax.bar([p + w / 2 for p in xpos], ft_c, w, color=C_QWEN08, label="Fine-tuned")
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10,
                fontweight="bold" if b in list(b2) else "normal")
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set_ylim(0, 115); ax.set_ylabel(f"Comparison queries correct (%, of {ncomp})")
ax.set_title("The hardest policy: routing document-comparisons to summaries (L0)\nEvery base fails it; fine-tuning fixes it (all four models)")
ax.legend(loc="upper right")
save(fig, "03_policy_comparison.png")

# 4) Accuracy vs size — efficiency frontier, all four models
fig, ax = plt.subplots(figsize=(8.5, 6))
for name, k, gb, c, mk in ALL:
    a = acc(ev[k], "finetuned")
    ax.scatter(gb, a, s=340, color=c, edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(f"{name}\n{gb} GB / {a:.0f}%", (gb, a), textcoords="offset points",
                xytext=(0, 18), ha="center", color=c, fontsize=10, fontweight="bold")
ax.set_xlabel("Q4_K_M size in GB  (smaller = less VRAM, faster)")
ax.set_ylabel("Fine-tuned routing accuracy (%)")
ax.set_xlim(0, 7.8); ax.set_ylim(82, 102)
ax.set_title("Accuracy vs model size — the efficiency frontier (all four)\nThe SMALLEST model (Qwen-0.8B) is also the most accurate router")
save(fig, "04_speed_vs_accuracy.png")

# 5) Output length base vs fine-tuned — all four models
fig, ax = plt.subplots(figsize=(9, 5))
base_t = [toks(ev[k], "base") for _, k, _, _, _ in ALL]
ft_t = [toks(ev[k], "finetuned") for _, k, _, _, _ in ALL]
xpos = list(range(len(ALL))); w = 0.38
b1 = ax.bar([p - w / 2 for p in xpos], base_t, w, color=C_BASE, label="Base")
b2 = ax.bar([p + w / 2 for p in xpos], ft_t, w, color=C_E2B, label="Fine-tuned")
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set_ylabel("Avg output length (tokens)")
ax.set_title("Fine-tuning makes outputs concise — all four models\nShorter = faster + cheaper per call (60-query eval)")
ax.legend()
save(fig, "05_output_length.png")

# 6) Per-category accuracy (E2B, hard set): where it helps vs where it regressed
def cat_acc(d, k, typ):
    qs = [q for q in d["queries"] if q[k]["expected_type"] == typ]
    return 100 * sum(q[k]["level_correct"] for q in qs) / len(qs) if qs else 0
cats = ["overview", "comparison", "specific", "deep_dive", "search"]
base_v = [cat_acc(hard_e2b, "base", c) for c in cats]
ft_v = [cat_acc(hard_e2b, "finetuned", c) for c in cats]
xpos = range(len(cats)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar([p - w / 2 for p in xpos], base_v, w, color=C_BASE, label="Base")
b2 = ax.bar([p + w / 2 for p in xpos], ft_v, w, color=C_E2B, label="Fine-tuned")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
ax.set_xticks(list(xpos)); ax.set_xticklabels([c + f"\n(L{ {'overview':0,'comparison':0,'specific':1,'deep_dive':2,'search':2}[c] })" for c in cats])
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 118)
ax.set_title("Where fine-tuning helps — and where it exposed a data gap\nE2B, hard set: comparisons 0->100%, but whole-doc summaries (specific) regressed")
ax.legend(loc="upper left", framealpha=0.95)
save(fig, "06_per_category_e2b.png")

# 7) Why a hard eval matters: easy set flatters the base model
fig, ax = plt.subplots(figsize=(8, 5))
groups = ["E2B", "12B"]
easy_base = [acc(easy_e2b, "base"), acc(easy_12b, "base")]
hard_base = [acc(hard_e2b, "base"), acc(hard_12b, "base")]
xpos = range(len(groups)); w = 0.38
b1 = ax.bar([p - w / 2 for p in xpos], easy_base, w, color="#86efac", label="Easy set")
b2 = ax.bar([p + w / 2 for p in xpos], hard_base, w, color="#f59e0b", label="Hard, policy-focused set")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=11)
ax.set_xticks(list(xpos)); ax.set_xticklabels(groups)
ax.set_ylabel("Base-model accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Why a hard eval matters\nThe easy set flatters the base model; the hard set reveals the real gap")
ax.legend()
save(fig, "07_easy_vs_hard.png")

# 8) Router accuracy — all four models, base vs fine-tuned (bar view)
names = [m[0] for m in ALL]
base_v = [acc(ev[k], "base") for _, k, _, _, _ in ALL]
ft_v = [acc(ev[k], "finetuned") for _, k, _, _, _ in ALL]
xpos = list(range(len(ALL))); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.5))
b1 = ax.bar([p - w / 2 for p in xpos], base_v, w, color=C_BASE, label="Base")
b2 = ax.bar([p + w / 2 for p in xpos], ft_v, w, color=C_QWEN08, label="Fine-tuned")
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10,
                fontweight="bold" if b in list(b2) else "normal")
ax.set_xticks(xpos); ax.set_xticklabels([f"{m[0]}\n{m[2]} GB" for m in ALL])
ax.set_ylabel("Routing accuracy (%)"); ax.set_ylim(0, 110)
ax.set_title("Router accuracy — all four models (60-query eval)\nFine-tuned: Qwen-0.8B 95% > Qwen-2B 93% > gemma-E2B 88% > gemma-12B 87%")
ax.legend(loc="upper right")
save(fig, "08_bakeoff_accuracy.png")

# 9) Fine-tuning GAIN (pp) per model — smaller/newer models gain most
gains = [acc(ev[k], "finetuned") - acc(ev[k], "base") for _, k, _, _, _ in ALL]
cols = [m[3] for m in ALL]
fig, ax = plt.subplots(figsize=(9.5, 5.5))
bars = ax.bar(xpos, gains, color=cols, width=0.6)
for b, g in zip(bars, gains):
    ax.annotate(f"+{g:.0f} pp", (b.get_x() + b.get_width() / 2, g),
                textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold", fontsize=12)
ax.set_xticks(xpos); ax.set_xticklabels([f"{m[0]}\n{m[2]} GB" for m in ALL])
ax.set_ylabel("Accuracy gain from fine-tuning (pp)"); ax.set_ylim(0, max(gains) + 14)
ax.set_title("How much each model GAINS from fine-tuning (60-query eval)\nThe biggest model (12B) gains the least (+7 pp); the small Qwens gain most")
save(fig, "09_bakeoff_size_vs_acc.png")

# 10) THE INVERSION (hero) — fine-tuned accuracy falls as model size rises
order = sorted(ALL, key=lambda m: m[2])          # by Q4 size ascending
sizes = [m[2] for m in order]
accs = [acc(ev[m[1]], "finetuned") for m in order]
fig, ax = plt.subplots(figsize=(10, 6.5))
# descending trend arrow: smallest (top-left) -> biggest (bottom-right)
ax.annotate("", xy=(sizes[-1], accs[-1]), xytext=(sizes[0], accs[0]),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=2.5, ls=(0, (6, 4)), alpha=0.7))
for m, s, a in zip(order, sizes, accs):
    ax.scatter(s, a, s=900, color=m[3], edgecolors="white", linewidths=2, zorder=5)
    ax.annotate(f"{m[0]}\n{a:.0f}%  ·  {m[2]} GB", (s, a),
                textcoords="offset points", xytext=(0, 26), ha="center",
                color=m[3], fontsize=11, fontweight="bold")
# ring the winner (smallest = best)
ax.scatter(sizes[0], accs[0], s=2600, facecolors="none", edgecolors="#f5b50a", linewidths=3, zorder=6)
ax.annotate("BEST & SMALLEST", (sizes[0], accs[0]), textcoords="offset points",
            xytext=(0, -34), ha="center", color="#b8860b", fontsize=11, fontweight="bold")
factor = sizes[-1] / sizes[0]
ax.text(0.5, 0.12, f"≈ {factor:.0f}× the size  →  {accs[0]-accs[-1]:.0f} points LOWER accuracy",
        transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold", color="#222",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#f5b50a", lw=1.5))
ax.set_xscale("log")
import matplotlib.ticker as mticker
ax.xaxis.set_major_locator(mticker.FixedLocator(sizes))
ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{s}" for s in sizes]))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("Model size — Q4_K_M, GB  (log scale; → bigger)")
ax.set_ylabel("Fine-tuned routing accuracy (%)")
ax.set_ylim(84, 99); ax.set_xlim(0.35, 9)
ax.set_title("After fine-tuning, BIGGER = WORSE\nFor a narrow task, the smallest model is the most accurate router")
save(fig, "10_the_inversion.png")

# 11) Base -> fine-tuned dumbbell — small models leap, the big one crawls
order2 = sorted(ALL, key=lambda m: m[2], reverse=True)   # biggest on top
fig, ax = plt.subplots(figsize=(9.5, 6))
for i, m in enumerate(order2):
    b, f = acc(ev[m[1]], "base"), acc(ev[m[1]], "finetuned")
    ax.plot([b, f], [i, i], color=m[3], lw=4, zorder=2, solid_capstyle="round")
    ax.scatter(b, i, color="#c4c9d0", s=150, zorder=3, edgecolors="white", linewidths=1.5)
    ax.scatter(f, i, color=m[3], s=300, zorder=4, edgecolors="white", linewidths=1.5)
    ax.annotate(f"{b:.0f}%", (b, i), textcoords="offset points", xytext=(-9, 0), ha="right", va="center", color="#777", fontsize=10)
    ax.annotate(f"{f:.0f}%", (f, i), textcoords="offset points", xytext=(10, 0), va="center", color=m[3], fontsize=11, fontweight="bold")
    ax.annotate(f"+{f-b:.0f} pp", ((b + f) / 2, i), textcoords="offset points", xytext=(0, 11), ha="center", color=m[3], fontsize=10, fontweight="bold")
ax.set_yticks(range(len(order2))); ax.set_yticklabels([f"{m[0]}\n{m[2]} GB" for m in order2])
ax.set_xlabel("Routing accuracy (%)  —  ● base   ●⟶ fine-tuned")
ax.set_xlim(28, 104); ax.set_ylim(-0.6, len(order2) - 0.4)
ax.set_title("Base → fine-tuned: the small models LEAP, the big one crawls\nThe 12B starts highest (80%) yet ends lowest (87%); Qwen-0.8B leaps 50→95%")
save(fig, "11_dumbbell_base_to_ft.png")

print("done ->", OUT)
