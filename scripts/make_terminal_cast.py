#!/usr/bin/env python
"""Generate a self-contained, animated SVG "terminal cast" of the RadeonForge quickstart.

  python scripts/make_terminal_cast.py   # -> assets/terminal-cast.svg

Pure stdlib, no deps. SMIL animation: lines reveal in sequence, looping, with a blinking
cursor. Animates inline on GitHub (like svg-term casts); where animation is suppressed it
degrades to a clean static terminal showing the whole flow. Brand-neutral by construction.
"""
import html
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "terminal-cast.svg"

# palette
BG, BAR = "#0a0e16", "#141b29"
P, CMD, MUT, OUTC, OK, CY, RED, GRN = "#10b981", "#eef5ff", "#5f7088", "#9fb4d4", "#10b981", "#22d3ee", "#f87171", "#34d399"

# each line = list of (text, color) segments
LINES = [
    [("$ ", P), ("make doctor", CMD)],
    [("  ✓ ROCm 7.2 · torch 2.9.1+rocm7.2.0 · ", OUTC), ("gfx1201 visible", CY)],
    [("  ✓ bitsandbytes 4-bit OK · transformers / peft / trl ready", OUTC)],
    [("$ ", P), ("make smoke", CMD)],
    [("  step 50/50 · loss ", OUTC), ("5.78 → 0.15", CY), ("   ✓ this box can train", OK)],
    [("$ ", P), ("make train", CMD), ("   # gemma-4 · QLoRA 4-bit", MUT)],
    [("  loss ", OUTC), ("3.32 → 0.34", CY), ("  ·  ~11 min on one Radeon R9700", OUTC)],
    [("  ✓ LoRA adapter saved", OK)],
    [("$ ", P), ("bash examples/gemma4-12b-qlora/export_gguf.sh", CMD)],
    [("  merge → GGUF → Q4_K_M   ", OUTC), ("23.8 GB → 6.9 GB", CY)],
    [("$ ", P), ("# serve on the GPU (llama.cpp · HIP)", MUT)],
    [("  routing A/B ", OUTC), ("79%", RED), (" → ", MUT), ("100%", GRN),
     ("   ·   ~117 tok/s   ·   ", OUTC), ("100% local", CY)],
]

# geometry
PAD_X, TOP, LH, FS, CW = 22, 52, 25, 15, 8.6
W = 880
H = TOP + len(LINES) * LH + 20
T = 15.0                       # loop length (s)
REVEAL = [0.5 + i * 1.05 for i in range(len(LINES))]   # per-line reveal time


def seg_spans(segs):
    out = []
    for text, color in segs:
        out.append(f'<tspan fill="{color}" xml:space="preserve">{html.escape(text)}</tspan>')
    return "".join(out)


def line_anim(ri):
    a = ri / T
    b = min(0.999, (ri + 0.22) / T)
    return (f'<animate attributeName="opacity" values="0;0;1;1;0" '
            f'keyTimes="0;{a:.4f};{b:.4f};0.965;1" dur="{T}s" repeatCount="indefinite"/>')


rows = []
last_y = TOP
last_w = 0
for i, segs in enumerate(LINES):
    y = TOP + i * LH
    rows.append(f'<text x="{PAD_X}" y="{y}" opacity="0">{seg_spans(segs)}{line_anim(REVEAL[i])}</text>')
    last_y = y
    last_w = sum(len(t) for t, _ in segs)

# blinking cursor parked at the end of the last line
cur_x = PAD_X + last_w * CW + 3
cursor = (f'<rect x="{cur_x:.0f}" y="{last_y-13}" width="9" height="17" fill="{CY}" opacity="0.9">'
          f'<animate attributeName="opacity" values="0.9;0.9;0;0" keyTimes="0;0.5;0.5;1" '
          f'dur="1.0s" repeatCount="indefinite"/></rect>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono',monospace" font-size="{FS}">
  <defs>
    <linearGradient id="rim" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#22d3ee" stop-opacity="0.5"/>
      <stop offset="0.5" stop-color="#8b5cf6" stop-opacity="0.35"/>
      <stop offset="1" stop-color="#ff6a2b" stop-opacity="0.45"/>
    </linearGradient>
  </defs>
  <rect x="1.5" y="1.5" width="{W-3}" height="{H-3}" rx="12" fill="{BG}" stroke="url(#rim)" stroke-width="1.5"/>
  <rect x="1.5" y="1.5" width="{W-3}" height="34" rx="12" fill="{BAR}"/>
  <rect x="1.5" y="22" width="{W-3}" height="14" fill="{BAR}"/>
  <circle cx="22" cy="18" r="6" fill="#ff5f56"/><circle cx="42" cy="18" r="6" fill="#ffbd2e"/><circle cx="62" cy="18" r="6" fill="#27c93f"/>
  <text x="{W/2}" y="23" text-anchor="middle" fill="{MUT}" font-size="12.5">RadeonForge — quickstart</text>
  {"".join(rows)}
  {cursor}
</svg>
'''

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg, encoding="utf-8")
print("wrote", OUT, f"({len(LINES)} lines, {W}x{H})")
