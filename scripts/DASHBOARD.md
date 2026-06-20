# Training Progress Dashboard (reusable)

A project-agnostic **live dashboard for fine-tuning runs**: 3D WebGL background,
live **t/s** and **loss** charts (SVG-exportable), **data-driven workflow flow + roadmap**
("you are here" advances automatically), GPU telemetry and a live log.
Pure **Python stdlib** — no server framework, no `pip install`, offline (no CDN).

## Quick start
```bash
python scripts/progress_dashboard.py \
  --data-dir ./data \
  --tracks ./scripts/dashboard.tracks.example.json \
  --open
```
- Default view: `http://127.0.0.1:8765/`
- **3D variant with charts + HUD:** `http://127.0.0.1:8765/gl`
- `--host 0.0.0.0` → reachable on the LAN / from your phone (also open the TCP port in the firewall).
- More flags: `--port` (default 8765), `--runs-dir` (default `<data-dir>/_runs`), `--log-dir`, `--title`.

## Concept: 3 data sources — everything from files, read live per request
| Source | Where | What |
|---|---|---|
| **Training runs** | `<runs-dir>/*.json` | step / epoch / loss / tps / `loss_series` / `tps_series` / status — written by the `ProgressCallback` in the trainer |
| **Results** | `<data-dir>/dashboard_results.json` | the **before/after line chart** |
| **Datasets** | `*.jsonl` + `--tracks` manifest | track progress bars (lines vs. `target`) |

`dashboard_results.json` schema:
```json
{ "title": "Results",
  "models":  [ {"name": "base", "values": {"score": 56}}, {"name": "tuned", "values": {"score": 100}} ],
  "metrics": [ {"key": "score", "label": "Citation coverage", "unit": "%"} ] }
```

## The manifest (`--tracks`)
```json
{ "title": "...", "subtitle": "...",
  "flow":     { "w":1000, "h":430, "nodes":[ ... ], "edges":[ ... ] },
  "pipeline": [ {"phase":"...", "status":"...", "detail":"...", "done_if":{...}, "active_if":{...}} ],
  "tracks":   [ {"label":"...", "file":"x.jsonl", "target":200, "group":"..."} ] }
```

### Data-driven status (Flow **and** Roadmap)
Every flow node / every pipeline phase can carry `done_if` / `active_if` — the server derives
the status **live from real data** (no manual upkeep). Without a condition, the `status` value
acts as a fallback. "current" automatically advances to the **first unfinished** step.

Condition forms:
| Condition | true when … |
|---|---|
| `{"run":"name","status":"done"}` | run `name` has this status (or `{"run":"name","running":true}`) |
| `{"dataset":"file.jsonl","min":N}` | dataset has ≥ N lines (optional `"below":M` for "still below target") |
| `{"result":"model","metric":"key","min":N}` | `dashboard_results.json` metric ≥ N |
| `{"any_run_active":true}` | some run is currently running |
| `{"any":[A,B,…]}` | **OR** of the sub-conditions (a list `[A,B]` = **AND**) |

### Flow nodes & edges
```json
{"id":"train", "label":"Training", "detail":"SFT", "status":"todo",
 "x":400, "y":210, "w":160, "h":52,
 "done_if":{"run":"my-run","status":"done"}, "active_if":{"any_run_active":true}}
```
Status colors: `done` (green ✓), `current` (cyan, pulsing, "you are here"), `todo` (muted), `goal` (purple ★).
Edges: `["a","b"]`; special cases `["a","b","dash"]` (dashed) and `["a","a","loop"]` (iteration loop).
Completed edges show an **animated energy pulse**.

## Trainer integration (`examples/gemma4-12b-qlora/train_qlora.py`)
The `ProgressCallback` writes the live status. Set this in the training config:
```yaml
progress_file: /path/to/<runs-dir>/<run-name>.json
train:
  logging_steps: 1          # -> dashboard ticks ~every step
  save_strategy: steps      # reboot-safe checkpoints
  save_steps: 25
```
The callback fills `step/epoch/loss/tps/loss_series/tps_series` → the loss and t/s charts build up live.

## Reuse for a new project
1. Set `progress_file` in the training config (writes to `<runs-dir>/`).
2. Create your own `dashboard.tracks.json` (title, flow with `done_if`/`active_if`, tracks) — use `dashboard.tracks.example.json` as a template.
3. Optionally a script that generates `dashboard_results.json` (for the before/after chart).
4. Start: `python scripts/progress_dashboard.py --data-dir … --tracks …`.
5. Optionally place a **thin wrapper** in the project that pre-fills the paths (example:
   a `progress_dashboard.py` wrapper in the project → `runpy` onto this engine with project-specific defaults).

## Security / LAN
`--host 0.0.0.0` makes the dashboard reachable for everyone on the same network (without auth). Harmless on a home
network, but don't leave it open on networks you don't control. Firewall rule (Windows, Admin):
```powershell
New-NetFirewallRule -DisplayName 'dashboard-8765' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -Profile Any
```

## Credit
This dashboard engine is part of [RadeonForge](https://github.com/KaiFelixBennett/RadeonForge) by
[@KaiFelixBennett](https://github.com/KaiFelixBennett). It renders a small footer credit/link by
default — if you reuse it (especially in a product), please keep that credit in place. Thanks! ★
