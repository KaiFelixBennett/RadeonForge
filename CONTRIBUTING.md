# Contributing to RadeonForge

This repo is meant to be **handed to the community**. The single most valuable
contribution is a **dated hardware/version report**: "I got QLoRA working on _card X_
with _ROCm Y / torch Z_ on _date_" — or "it broke, here's how I fixed it." On this
hardware the stack rots fast, so **a version entry without a date is worse than none.**

## Ways to help (most → least valuable)

1. **Report a working (or broken) stack.** Open a *Hardware / version report* issue, or
   PR an entry into [VERSIONS.md](VERSIONS.md) with your **exact** versions **and the date**.
2. **Confirm another card.** RDNA3 (`gfx1100`, e.g. RX 7900 XTX) and other RDNA4 cards
   (RX 9070 / 9070 XT) are *expected* to work but untested — tell us.
3. **A new silent-failure → fix.** If you hit a trap not in
   [docs/troubleshooting.md](docs/troubleshooting.md), add the symptom → fix row.
4. **Docs & examples.** Clearer steps, a new worked example, a better diagram.

## Ground rules

- **Run the smoke test before claiming "it works":** `make smoke` (or
  `python scripts/smoke_test.py`). A green smoke test is our reproducibility guarantee —
  see the README "Philosophy" section.
- **Date everything dated.** Pinned versions belong in `VERSIONS.md` with the date you
  verified them.
- **Keep it honest.** If something is *no different* after fine-tuning, say so. The
  credibility of this repo comes from not overclaiming.
- **No secrets, no internal names.** This is a public, brand-neutral toolkit. Keep
  example data generic (a query-router is the running example) and never commit weights,
  `.gguf`/`.safetensors`, private datasets, or company-specific identifiers.

## Dev setup

```bash
git clone https://github.com/KaiFelixBennett/RadeonForge && cd RadeonForge
make demo          # see the dashboard with no GPU (sanity-check your checkout)
make doctor        # on a real RDNA box: environment health check
make smoke         # 50-step QLoRA — does the loss actually fall?
```

Regenerating the visual assets (optional, needs Node + ffmpeg): `make assets`.

## PRs

- One topic per PR; describe **what you tested it on** (card, OS, ROCm/torch versions, date).
- Don't commit large binaries or generated demo data (`.gitignore` covers the usual cases).
- Be kind. This is a volunteer, field-notes project.
