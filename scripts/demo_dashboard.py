#!/usr/bin/env python
"""RadeonForge — one-command live DEMO of the progress dashboard (no GPU, no training).

Seeds a self-contained, **brand-neutral** example data dir (`examples/dashboard-demo/`,
generated — gitignored) with everything the rich `/gl` view renders:
  • two training runs (one finished, one **live**) with loss_series + tps_series
  • a base-vs-fine-tuned scorecard + radar fingerprint
  • a before/after results line-chart
  • a live eval pass/fail grid
  • a data-driven flow + roadmap
  • two dataset tracks (generated routing JSONL)

Then it serves the dashboard (reusing `progress_dashboard.py`) and runs a tiny
**simulator** that advances the live run + log so the page genuinely animates — the
HUD, tokens/s pulse, growing loss curve and ETA all move, with zero hardware.

The numbers are realistic stand-ins for a QLoRA query-router fine-tune; this is a
DEMO, clearly labelled as such. Reuse it for any project by pointing the real
dashboard at your own data (see scripts/DASHBOARD.md) — this file only fabricates
sample data so newcomers see the full picture in one command.

  python scripts/demo_dashboard.py --open                 # live demo (Ctrl+C to stop)
  python scripts/demo_dashboard.py --seed-only            # just write the files (for capture)
  python scripts/demo_dashboard.py --seconds 8 --no-open  # run sim for 8s then exit
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import random
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
DEMO = (HERE.parent / "examples" / "dashboard-demo").resolve()
RUNS = DEMO / "_runs"
EVAL = DEMO / "_eval"

# ── load the dashboard engine (single source of truth for serving) ──────────────
_spec = importlib.util.spec_from_file_location("progress_dashboard", HERE / "progress_dashboard.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)

LIVE_RUN = "gemma-4-12b · router"      # the run the simulator drives
DONE_RUN = "gemma-4-e2b · router"      # a finished run (for the multi-line loss chart)
TOTAL = 138                            # steps in the live run
EPOCHS = 2


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _loss_at(frac: float, hi: float, lo: float, k: float = 4.2) -> float:
    """Smooth exponential decay hi→lo with light, reproducible step-noise."""
    base = lo + (hi - lo) * math.exp(-k * frac)
    return round(base + (random.random() - 0.5) * 0.06 * (0.4 + (1 - frac)), 4)


def _tps_at(frac: float) -> int:
    return max(380, round(1150 + 230 * math.sin(frac * 7.5) + (random.random() - 0.5) * 120))


def _series(n: int, hi: float, lo: float, seed: int):
    r = random.Random(seed)
    out_loss, out_tps = [], []
    for i in range(n):
        f = i / max(1, n - 1)
        out_loss.append(round(lo + (hi - lo) * math.exp(-4.2 * f) + (r.random() - 0.5) * 0.07, 4))
        out_tps.append(max(380, round(1150 + 230 * math.sin(f * 7.5) + (r.random() - 0.5) * 120)))
    return out_loss, out_tps


# ── routing dataset generator (doubles as a usable sample) ───────────────────────
_OVERVIEW = ["Welche Dokumente kennst du?", "Gib mir einen Überblick über die Inhalte.",
             "Liste alle verfügbaren Berichte auf.", "Was ist alles im Wissensspeicher?"]
_COMPARE = ["Vergleiche die Aussagen aus beiden Verträgen.",
            "Worin unterscheiden sich Bericht A und Bericht B?",
            "Stelle die beiden Angebote gegenüber.", "Was ist der Unterschied zwischen V1 und V2?"]
_SPECIFIC = ["Was steht in Abschnitt 3.2 zum Thema Haftung?",
             "Welche Frist nennt das Dokument für die Kündigung?",
             "Wie hoch ist die im Vertrag genannte Vergütung?"]
_DEEP = ["Erkläre die Architektur des Retrieval-Systems im Detail.",
         "Wie funktioniert das Routing über mehrere Ebenen?",
         "Beschreibe den vollständigen Deployment-Prozess."]
_SEARCH = ["Finde alle Stellen, die 'Service Mesh' erwähnen.",
           "Wo wird der Token-Refresh beim Login behandelt?",
           "Suche Passagen zu Rate-Limiting."]
_CATS = [("overview", 0, _OVERVIEW), ("comparison", 0, _COMPARE), ("specific", 1, _SPECIFIC),
         ("deep_dive", 2, _DEEP), ("search", 2, _SEARCH)]
_SYS = ("Du bist ein Query-Router. Klassifiziere die Anfrage und gib NUR JSON zurück: "
        '{"query_type": <overview|comparison|specific|deep_dive|search>, "target_level": <0|1|2>}.')


def _dataset(path: pathlib.Path, n: int, seed: int) -> None:
    r = random.Random(seed)
    lines = []
    for i in range(n):
        cat, lvl, pool = _CATS[i % len(_CATS)]
        q = r.choice(pool)
        row = {"messages": [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f"Analyze this query:\n{q}"},
            {"role": "assistant", "content": json.dumps({"query_type": cat, "target_level": lvl})},
        ]}
        lines.append(json.dumps(row, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _eval_cases():
    # held-out, policy-focused: comparisons fully fixed, one specific gap (honest)
    dims = {"overview": (4, 4, 4), "comparison": (5, 5, 5), "specific": (4, 3, 4),
            "deep_dive": (3, 3, 3), "search": (5, 4, 5)}
    cases = []
    for k, (total, passed, done) in dims.items():
        for i in range(done):
            cases.append({"dim": k, "ok": i < passed})
    out = {k: {"total": t, "pass": p, "done": d} for k, (t, p, d) in dims.items()}
    return out, cases


def seed(reseed: bool = False) -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "README.txt").write_text(
        "GENERATED by scripts/demo_dashboard.py — brand-neutral sample data for the dashboard demo.\n"
        "Safe to delete; it is re-created on the next `make demo`. Not real training output.\n",
        encoding="utf-8")

    # finished run: gemma-4-e2b (3 epochs, ends ~0.21)
    dl, dt = _series(120, 2.86, 0.21, seed=7)
    _write_json(RUNS / "gemma-4-e2b-router.json", {
        "name": DONE_RUN, "status": "done", "step": 120, "total_steps": 120,
        "epoch": 3.0, "epochs": 3, "loss": dl[-1], "tps": dt[-1],
        "loss_series": dl, "tps_series": dt, "updated": time.time() - 6 * 3600})

    # scorecard (base vs fine-tuned, E2B held-out) — drives table + radar fingerprint
    _write_json(DEMO / "scorecard.json", {
        "axes": [
            {"key": "route", "label": "Routing-Acc", "tip": "Anteil korrekt geroutete Anfragen (target_level)."},
            {"key": "type", "label": "Type-Acc", "tip": "Korrekt erkannter Anfrage-Typ."},
            {"key": "json", "label": "JSON-valide", "tip": "Anteil syntaktisch valider JSON-Ausgaben."},
            {"key": "policy", "label": "Compare-Policy", "tip": "Dokument-Vergleiche korrekt auf Zusammenfassung (L0) geroutet."},
            {"key": "eff", "label": "Effizienz", "tip": "Normierter Score aus kürzerer Ausgabe (-19%) + schnellerem Serving."},
        ],
        "models": {
            "base": {"route": 79, "type": 64, "json": 100, "policy": 0, "eff": 55},
            "fine-tuned": {"route": 100, "type": 100, "json": 100, "policy": 100, "eff": 78},
        },
        "detail": {
            "base": {"route": "79%", "type": "64%", "json": "100%", "policy": "0/5", "eff": "55"},
            "fine-tuned": {"route": "100%", "type": "100%", "json": "100%", "policy": "5/5", "eff": "78"},
        },
    })

    # before/after line chart
    _write_json(DEMO / "dashboard_results.json", {
        "title": "Router — Base vs. Fine-tuned (held-out)",
        "models": [
            {"name": "base", "values": {"route": 79, "type": 64, "policy": 0, "json": 100}},
            {"name": "fine-tuned", "values": {"route": 100, "type": 100, "policy": 100, "json": 100}},
        ],
        "metrics": [
            {"key": "route", "label": "Routing-Acc", "unit": "%"},
            {"key": "type", "label": "Type-Acc", "unit": "%"},
            {"key": "policy", "label": "Compare-Policy", "unit": "%"},
            {"key": "json", "label": "JSON-valide", "unit": "%"},
        ],
    })

    # manifest: title/subtitle/flow/pipeline/tracks (data-driven status)
    _write_json(DEMO / "demo.tracks.json", {
        "title": "RadeonForge — gemma-4 QLoRA router",
        "subtitle": "gemma-4 · QLoRA 4-bit · AMD Radeon AI PRO R9700 (RDNA4 · gfx1201)",
        "flow": {"w": 1000, "h": 300, "nodes": [
            {"id": "data", "label": "Daten", "detail": "158 Beispiele", "status": "done", "x": 120, "y": 150, "w": 150, "h": 52,
             "done_if": {"dataset": "routing_train.jsonl", "min": 100}},
            {"id": "train", "label": "QLoRA", "detail": "4-bit · LoRA", "status": "todo", "x": 330, "y": 150, "w": 150, "h": 52,
             "done_if": {"run": DONE_RUN, "status": "done"}, "active_if": {"any_run_active": True}},
            {"id": "gguf", "label": "Merge → GGUF", "detail": "Q4_K_M", "status": "todo", "x": 540, "y": 150, "w": 150, "h": 52,
             "done_if": {"run": DONE_RUN, "status": "done"}},
            {"id": "eval", "label": "Eval", "detail": "held-out A/B", "status": "todo", "x": 750, "y": 150, "w": 150, "h": 52,
             "done_if": {"result": "fine-tuned", "metric": "route", "min": 90}},
            {"id": "serve", "label": "Serve", "detail": "llama.cpp HIP", "status": "goal", "x": 930, "y": 150, "w": 120, "h": 52},
        ], "edges": [["data", "train"], ["train", "gguf"], ["gguf", "eval"], ["eval", "serve"]]},
        "pipeline": [
            {"phase": "GPU-Stack geprüft", "status": "done", "detail": "doctor.sh + smoke_test.py grün (loss 5.78 → 0.15)"},
            {"phase": "Datensatz aufgebaut", "status": "todo", "detail": "158 Routing-Beispiele, balanciert",
             "done_if": {"dataset": "routing_train.jsonl", "min": 100}},
            {"phase": "QLoRA-Training (E2B)", "status": "done", "detail": "3 Epochen, ~2 min auf R9700"},
            {"phase": "QLoRA-Training (12B)", "status": "todo", "detail": "2 Epochen, ~11 min auf R9700",
             "done_if": {"run": LIVE_RUN, "status": "done"}, "active_if": {"run": LIVE_RUN, "running": True}},
            {"phase": "Merge → GGUF (Q4_K_M)", "status": "todo", "detail": "6.9 GB, llama.cpp HIP-Backend"},
            {"phase": "Held-out A/B + Serving", "status": "todo", "detail": "79% → 100% · ~117 tok/s",
             "done_if": {"result": "fine-tuned", "metric": "route", "min": 90}},
        ],
        "tracks": [
            {"label": "Routing-Trainingsdaten", "file": "routing_train.jsonl", "target": 158, "group": "Datensätze"},
            {"label": "Held-out Eval-Set", "file": "routing_eval.jsonl", "target": 21, "group": "Datensätze"},
        ],
    })

    if reseed or not (DEMO / "routing_train.jsonl").exists():
        _dataset(DEMO / "routing_train.jsonl", 158, seed=11)
        _dataset(DEMO / "routing_eval.jsonl", 21, seed=23)


def write_live(step: int) -> None:
    frac = step / TOTAL
    ls, ts = _series(max(2, step), 3.32, 0.30, seed=3)
    _write_json(RUNS / "gemma-4-12b-router.json", {
        "name": LIVE_RUN, "status": "running", "step": step, "total_steps": TOTAL,
        "epoch": round(EPOCHS * frac, 2), "epochs": EPOCHS,
        "loss": ls[-1], "tps": ts[-1], "loss_series": ls, "tps_series": ts,
        "updated": time.time()})
    # live eval grid (kept fresh so the panel stays visible)
    dims, cases = _eval_cases()
    _write_json(EVAL / "heldout.json", {
        "tag": "held-out · gemma-4-e2b", "status": "running", "updated": time.time(),
        "dims": dims, "cases": cases})
    # live log -> ETA + s/Step chips + nowbar
    sit = 2.26
    remain = max(0, TOTAL - step) * sit
    mm, ss = divmod(int(remain), 60)
    el = int(step * sit); emm, ess = divmod(el, 60)
    bar = f"{int(frac*100):3d}%|{'█'*int(frac*10):<10}| {step}/{TOTAL} [{emm:02d}:{ess:02d}<{mm:02d}:{ss:02d}, {sit:.2f}s/it]"
    (DEMO / "train.log").write_text(
        "[demo] loading gemma-4-12B-it in 4-bit (nf4) · LoRA r=16 alpha=32 target=all-linear\n"
        "[demo] RDNA4 workarounds: optim=adamw_torch · math SDPA · gradient_checkpointing\n"
        f"[demo] {{'loss': {ls[-1]}, 'grad_norm': 0.74, 'learning_rate': 0.00014, 'epoch': {round(EPOCHS*frac,2)}, 'num_tokens': {step*2048}}}\n"
        f"{bar}\n", encoding="utf-8")


def simulate(seconds: float, tick: float, start_step: int) -> None:
    step = start_step
    t0 = time.time()
    while seconds <= 0 or (time.time() - t0) < seconds:
        write_live(step)
        step += 1
        if step > TOTAL:        # loop the demo so it never goes idle
            step = int(TOTAL * 0.45)
        time.sleep(tick)


def serve(host: str, port: int) -> None:
    pd.CFG.update({"data_dir": str(DEMO), "runs_dir": str(RUNS), "log_dir": str(DEMO),
                   "tracks": str(DEMO / "demo.tracks.json"),
                   "title": "RadeonForge — gemma-4 QLoRA router"})
    srv = pd.ThreadingHTTPServer((host, port), pd.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description="RadeonForge live dashboard DEMO (no GPU needed)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seconds", type=float, default=0, help="sim duration; 0 = run forever")
    ap.add_argument("--tick", type=float, default=0.7, help="seconds between simulated steps")
    ap.add_argument("--start-step", type=int, default=int(TOTAL * 0.62), help="initial step of the live run")
    ap.add_argument("--seed-only", action="store_true", help="write demo files + one live frame, then exit")
    ap.add_argument("--reseed", action="store_true", help="regenerate the JSONL datasets too")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--open", dest="open_", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    seed(reseed=args.reseed)
    write_live(args.start_step)
    print(f"[demo] seeded {DEMO}")
    if args.seed_only:
        return

    serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/gl"
    print(f"[demo] dashboard live -> {url}   (Ctrl+C to stop)")
    if args.open_ and not args.no_open:
        import webbrowser
        webbrowser.open(url)
    try:
        simulate(args.seconds, args.tick, args.start_step)
    except KeyboardInterrupt:
        print("\n[demo] stopped")


if __name__ == "__main__":
    main()
