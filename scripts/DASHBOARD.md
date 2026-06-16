# Training-Progress-Dashboard (wiederverwendbar)

Ein projektneutrales **Live-Dashboard für Fine-Tuning-Läufe**: 3D-WebGL-Hintergrund,
Live-**t/s**- und **Loss**-Charts (SVG-exportierbar), **datengetriebener Ablauf-Flow + Roadmap**
(„Du bist hier" rückt automatisch vor), GPU-Telemetrie und Live-Log.
Reines **Python-stdlib** — kein Server-Framework, kein `pip install`, offline (kein CDN).

## Schnellstart
```bash
python scripts/progress_dashboard.py \
  --data-dir ./data \
  --tracks ./scripts/dashboard.tracks.example.json \
  --open
```
- Standard-Ansicht: `http://127.0.0.1:8765/`
- **3D-Variante mit Charts + HUD:** `http://127.0.0.1:8765/gl`
- `--host 0.0.0.0` → im LAN/vom Handy erreichbar (zusätzlich den TCP-Port in der Firewall freigeben).
- Weitere Flags: `--port` (Default 8765), `--runs-dir` (Default `<data-dir>/_runs`), `--log-dir`, `--title`.

## Konzept: 3 Datenquellen — alles aus Dateien, live pro Request gelesen
| Quelle | Wo | Was |
|---|---|---|
| **Trainings** | `<runs-dir>/*.json` | step / epoch / loss / tps / `loss_series` / `tps_series` / status — geschrieben vom `ProgressCallback` im Trainer |
| **Ergebnisse** | `<data-dir>/dashboard_results.json` | das **Vorher/Nachher-Linienchart** |
| **Datensätze** | `*.jsonl` + `--tracks`-Manifest | Track-Fortschrittsbalken (Zeilen vs. `target`) |

`dashboard_results.json`-Schema:
```json
{ "title": "Ergebnisse",
  "models":  [ {"name": "base", "values": {"score": 56}}, {"name": "tuned", "values": {"score": 100}} ],
  "metrics": [ {"key": "score", "label": "Citation-Coverage", "unit": "%"} ] }
```

## Das Manifest (`--tracks`)
```json
{ "title": "...", "subtitle": "...",
  "flow":     { "w":1000, "h":430, "nodes":[ ... ], "edges":[ ... ] },
  "pipeline": [ {"phase":"...", "status":"...", "detail":"...", "done_if":{...}, "active_if":{...}} ],
  "tracks":   [ {"label":"...", "file":"x.jsonl", "target":200, "group":"..."} ] }
```

### Datengetriebener Status (Flow **und** Roadmap)
Jeder Flow-Knoten / jede Pipeline-Phase kann `done_if` / `active_if` tragen — der Server leitet
den Status **live aus echten Daten** ab (kein Hand-Pflegen). Ohne Bedingung gilt der `status`-Wert
als Fallback. „current" rückt automatisch auf den **ersten unfertigen** Schritt.

Bedingungs-Formen:
| Bedingung | wahr, wenn … |
|---|---|
| `{"run":"name","status":"done"}` | Lauf `name` hat diesen Status (oder `{"run":"name","running":true}`) |
| `{"dataset":"file.jsonl","min":N}` | Datensatz hat ≥ N Zeilen (optional `"below":M` für „noch unter Ziel") |
| `{"result":"model","metric":"key","min":N}` | `dashboard_results.json`-Metrik ≥ N |
| `{"any_run_active":true}` | irgendein Lauf läuft gerade |
| `{"any":[A,B,…]}` | **ODER** der Teilbedingungen (eine Liste `[A,B]` = **UND**) |

### Flow-Knoten & -Kanten
```json
{"id":"train", "label":"Training", "detail":"SFT", "status":"todo",
 "x":400, "y":210, "w":160, "h":52,
 "done_if":{"run":"mein-run","status":"done"}, "active_if":{"any_run_active":true}}
```
Status-Farben: `done` (grün ✓), `current` (cyan, pulsierend, „Du bist hier"), `todo` (gedämpft), `goal` (lila ★).
Kanten: `["a","b"]`; Sonderfälle `["a","b","dash"]` (gestrichelt) und `["a","a","loop"]` (Iterations-Schleife).
Erledigte Kanten zeigen einen **animierten Energie-Puls**.

## Trainer-Integration (`examples/gemma4-12b-qlora/train_qlora.py`)
Der `ProgressCallback` schreibt den Live-Status. In der Trainings-Config setzen:
```yaml
progress_file: /pfad/zu/<runs-dir>/<lauf-name>.json
train:
  logging_steps: 1          # -> Dashboard tickt ~jeden Step
  save_strategy: steps      # reboot-sichere Checkpoints
  save_steps: 25
```
Der Callback füllt `step/epoch/loss/tps/loss_series/tps_series` → Loss- und t/s-Chart bauen sich live auf.

## Für ein neues Projekt wiederverwenden
1. In der Trainings-Config `progress_file` setzen (schreibt nach `<runs-dir>/`).
2. Eigenes `dashboard.tracks.json` anlegen (Titel, Flow mit `done_if`/`active_if`, Tracks) — `dashboard.tracks.example.json` als Vorlage.
3. Optional ein Skript, das `dashboard_results.json` erzeugt (für das Vorher/Nachher-Chart).
4. Starten: `python scripts/progress_dashboard.py --data-dir … --tracks …`.
5. Optional einen **dünnen Wrapper** ins Projekt legen, der die Pfade vorbelegt (Beispiel:
   `custom-rag/finetune/progress_dashboard.py` → `runpy` auf diese Engine mit projektspezifischen Defaults).

## Sicherheit / LAN
`--host 0.0.0.0` macht das Dashboard für alle im selben Netz erreichbar (ohne Auth). Im Heimnetz
unkritisch, in fremden Netzen nicht offen lassen. Firewall-Regel (Windows, Admin):
```powershell
New-NetFirewallRule -DisplayName 'dashboard-8765' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -Profile Any
```
