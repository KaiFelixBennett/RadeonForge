#!/usr/bin/env python
"""RadeonForge — wiederverwendbares Live-Dashboard für Trainings-Läufe & Datensätze.

Eine Datei, nur Python-Stdlib (kein CDN), offline-tauglich. Dient eine HTML-Seite unter /
und eine JSON-API unter /api/progress. Auto-Refresh alle 3 s. Animierter 3D-Canvas-Hintergrund
(iridescente Knotenwolke, langsam rotierend/driftend) — pulsiert stärker, während ein Training läuft.

Drei Quellen, alle generisch:
  1) ROADMAP    — optionale `pipeline` im Manifest: [{phase,status:done|current|todo,detail}].
  2) TRAININGS  — <runs-dir>/*.json vom ProgressCallback in train_qlora.py (Step/Epoche/Loss).
  3) DATENSÄTZE — --tracks Manifest ODER Auto-Discovery aller *.jsonl in --data-dir.

Beispiele:
  python scripts/progress_dashboard.py --runs-dir /root/router-pilot/_runs --open
  python scripts/progress_dashboard.py --data-dir ./data --tracks ./dashboard.tracks.example.json --open

Manifest (JSON): { "title","subtitle","pipeline":[{phase,status,detail}], "tracks":[{label,file,target,group}] }
"subtitle" eignet sich z. B. für "Teacher: <Modell> · Student: <Modell>" (projekt-konfigurierbar).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTIVE_SECS = 90
CFG = {}   # gesetzt in main(): data_dir, runs_dir, tracks, title
_HERE = pathlib.Path(__file__).resolve().parent
GL_FILE = _HERE / "progress_dashboard_gl.html"   # optionale WebGL-Variante unter /gl


def report_training(runs_dir, name: str, *, status: str = "running", step=None,
                    total_steps=None, epoch=None, epochs=None, loss=None, note: str = "") -> None:
    """Schreibt/aktualisiert <runs-dir>/<name>.json. Reine Stdlib, aus jedem Prozess (auch WSL)."""
    d = pathlib.Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({
        "name": name, "status": status, "step": step, "total_steps": total_steps,
        "epoch": epoch, "epochs": epochs, "loss": loss, "note": note, "updated": time.time(),
    }, ensure_ascii=False), encoding="utf-8")


def _count(p: pathlib.Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


def _age(p: pathlib.Path):
    return (time.time() - p.stat().st_mtime) if p.exists() else None


def _manifest() -> dict:
    f = CFG.get("tracks")
    if f and pathlib.Path(f).exists():
        try:
            return json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_tracks():
    """-> list[(label, filename, target, group)] aus Manifest oder Auto-Discovery."""
    data_dir = CFG.get("data_dir")
    listed = []
    seen = set()
    man = _manifest()
    if man:
        for t in man.get("tracks", []):
            listed.append((t["label"], t["file"], t.get("target"), t.get("group", "Datensätze")))
            seen.add(t["file"])
    if data_dir:
        for p in sorted(pathlib.Path(data_dir).glob("*.jsonl")):
            if p.name not in seen:
                listed.append((p.name, p.name, None, "Weitere"))
    return listed


def collect() -> dict:
    data_dir = pathlib.Path(CFG["data_dir"]) if CFG.get("data_dir") else None
    datasets = []
    for label, fn, target, group in _load_tracks():
        p = data_dir / fn if data_dir else pathlib.Path(fn)
        n = _count(p)
        age = _age(p)
        datasets.append({"label": label, "file": fn, "group": group, "count": n, "target": target,
                         "pct": (round(100 * n / target) if target else None),
                         "age_secs": (round(age) if age is not None else None),
                         "active": (age is not None and age < ACTIVE_SECS), "exists": p.exists()})
    trainings = []
    runs_dir = pathlib.Path(CFG["runs_dir"]) if CFG.get("runs_dir") else None
    if runs_dir and runs_dir.exists():
        # NUR Status-JSONs, NICHT die <name>.meta.json (Config-Snapshots) -> sonst "undefined"-Karten
        for p in sorted(x for x in runs_dir.glob("*.json") if not x.name.endswith(".meta.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not d.get("name"):
                continue
            age = time.time() - d.get("updated", p.stat().st_mtime)
            st, tot = d.get("step"), d.get("total_steps")
            d["age_secs"] = round(age)
            d["active"] = age < ACTIVE_SECS and d.get("status") == "running"
            d["stale"] = (d.get("status") == "running" and age >= ACTIVE_SECS)   # "running" aber tot
            d["pct"] = (round(100 * st / tot) if (st and tot) else None)
            trainings.append(d)
    man = _manifest()
    results = {}
    if data_dir and (data_dir / "dashboard_results.json").exists():
        try:
            results = json.loads((data_dir / "dashboard_results.json").read_text(encoding="utf-8"))
        except Exception:
            results = {}
    flow = _resolve_flow(man.get("flow", {}), trainings, datasets, results)
    pipeline = _resolve_pipeline(man.get("pipeline", []), trainings, datasets, results)
    scorecard = {}
    if data_dir and (data_dir / "scorecard.json").exists():
        try:
            scorecard = json.loads((data_dir / "scorecard.json").read_text(encoding="utf-8"))
        except Exception:
            scorecard = {}
    # Live-Eval: jüngste data/_eval/<tag>.json (von eval_progress.EvalProgress) -> Live-Panel
    eval_live = {}
    edir = (data_dir / "_eval") if data_dir else None
    if edir and edir.exists():
        newest, nt = None, -1.0
        for p in edir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            u = d.get("updated", p.stat().st_mtime)
            if u > nt:
                nt, newest = u, d
        if newest is not None:
            newest["age_secs"] = round(time.time() - nt)
            eval_live = newest
    return {"generated_at": _dt.datetime.now().strftime("%H:%M:%S"),
            "title": CFG.get("title", "Training-Progress"), "subtitle": man.get("subtitle", ""),
            "pipeline": pipeline, "flow": flow, "scorecard": scorecard, "eval_live": eval_live,
            "results": results, "datasets": datasets, "trainings": trainings}


def _cond_met(c, trainings, ds_by_file, res_models) -> bool:
    """Wertet eine Daten-Bedingung gegen den Live-Stand aus (für datengetriebenen Flow)."""
    if isinstance(c, list):
        return all(_cond_met(x, trainings, ds_by_file, res_models) for x in c)
    if not isinstance(c, dict):
        return False
    if "any" in c:
        return any(_cond_met(x, trainings, ds_by_file, res_models) for x in c["any"])
    if c.get("any_run_active"):
        return any(t.get("active") for t in trainings)
    if "run" in c:
        t = next((t for t in trainings if t.get("name") == c["run"]), None)
        if not t:
            return False
        if c.get("running"):
            return bool(t.get("active"))
        return t.get("status") == c.get("status", "done")
    if "dataset" in c:
        n = (ds_by_file.get(c["dataset"], {}) or {}).get("count", 0) or 0
        if n < c.get("min", 1):
            return False
        if "below" in c and not (n < c["below"]):
            return False
        return True
    if "result" in c:
        m = res_models.get(c["result"])
        if not m:
            return False
        return (m.get("values", {}).get(c["metric"]) or -1) >= c.get("min", 0)
    return False


def _resolve_flow(flow, trainings, datasets, results):
    """Status jedes Flow-Knotens datengetrieben ableiten: done_if/active_if gegen echte Daten.
    'current' rückt automatisch auf den ersten unfertigen Knoten -> 'Du bist hier' ist immer korrekt."""
    if not flow or not flow.get("nodes"):
        return flow
    ds_by_file = {d["file"]: d for d in datasets}
    res_models = {m.get("name"): m for m in (results.get("models") or [])}
    nodes = []
    for nd in flow["nodes"]:
        nd = dict(nd)
        if nd.get("done_if") and _cond_met(nd["done_if"], trainings, ds_by_file, res_models):
            nd["status"] = "done"
        elif nd.get("active_if") and _cond_met(nd["active_if"], trainings, ds_by_file, res_models):
            nd["status"] = "current"
        nodes.append(nd)
    if not any(n.get("status") == "current" for n in nodes):
        for n in nodes:
            if n.get("status") not in ("done", "goal"):
                n["status"] = "current"
                break
    flow = dict(flow)
    flow["nodes"] = nodes
    return flow


def _resolve_pipeline(pipeline, trainings, datasets, results):
    """Gleiche Daten-Auflösung für die Roadmap-Phasen (optional je Phase via done_if/active_if)."""
    if not pipeline:
        return pipeline
    ds_by_file = {d["file"]: d for d in datasets}
    res_models = {m.get("name"): m for m in (results.get("models") or [])}
    out = []
    for ph in pipeline:
        ph = dict(ph)
        if ph.get("done_if") and _cond_met(ph["done_if"], trainings, ds_by_file, res_models):
            ph["status"] = "done"
        elif ph.get("active_if") and _cond_met(ph["active_if"], trainings, ds_by_file, res_models):
            ph["status"] = "current"
        out.append(ph)
    if not any(p.get("status") == "current" for p in out):
        for p in out:
            if p.get("status") not in ("done",):
                p["status"] = "current"
                break
    return out


# ── GPU-Telemetrie: portabel & graceful (probiert mehrere Tools, blendet sonst aus) ──
_GPU = {"cmd": None, "kind": None, "tried": False, "cache": None, "ts": 0.0}


_WIN_PS = (r"$u=((Get-Counter '\GPU Engine(*)\Utilization Percentage' -EA SilentlyContinue)."
           r"CounterSamples.CookedValue|Measure-Object -Maximum).Maximum;"
           r"$m=((Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -EA SilentlyContinue)."
           r"CounterSamples.CookedValue|Measure-Object -Sum).Sum;"
           r"Write-Output ('{0}|{1}' -f [int]$u,[int64]$m)")


def _gpu_probe():
    cands = [
        ("nvidia", ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"]),
        ("rocm", ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"]),
        ("win", ["powershell", "-NoProfile", "-Command", _WIN_PS]),   # Windows-Host (AMD/NVIDIA), für WSL2-GPU
        ("wsl-rocm", ["wsl", "rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"]),
    ]
    for kind, cmd in cands:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if r.returncode == 0 and r.stdout.strip():
                _GPU["kind"], _GPU["cmd"] = kind, cmd
                return
        except Exception:
            pass


def gpu_stats() -> dict:
    if not _GPU["tried"]:
        _GPU["tried"] = True
        try:
            _gpu_probe()
        except Exception:
            pass
    if not _GPU["cmd"]:
        return {}
    if _GPU["cache"] is not None and (time.time() - _GPU["ts"]) < 2.0:
        return _GPU["cache"]
    res = {}
    try:
        out = subprocess.run(_GPU["cmd"], capture_output=True, text=True, timeout=8).stdout
        if _GPU["kind"] == "nvidia":
            n, u, used, tot = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            res = {"name": n, "util": round(float(u)), "used": round(float(used) / 1024, 1),
                   "total": round(float(tot) / 1024, 1)}
        elif _GPU["kind"] == "win":
            u, m = out.strip().split("|")
            res = {"name": "GPU (Windows-Host)", "util": min(100, round(float(u)))}
            if float(m) > 0:
                res["used"] = round(float(m) / 1e9, 1)
        else:
            d = json.loads(out)
            card = {}
            for v in d.values():
                if isinstance(v, dict) and any("memory" in k.lower() for k in v):
                    card = v
                    break

            def find(must, nope=()):
                for k, v in card.items():
                    kl = k.lower()
                    if all(t in kl for t in must) and not any(x in kl for x in nope):
                        try:
                            return float(str(v).strip())
                        except Exception:
                            pass
                return None
            util = find(["use"], ["memory", "vram"])
            used = find(["used", "memory"]); tot = find(["total", "memory"], ["used"])
            res = {"name": "ROCm GPU"}
            if util is not None: res["util"] = round(util)
            if used and tot: res["used"] = round(used / 1e9, 1); res["total"] = round(tot / 1e9, 1)
    except Exception:
        res = {}
    _GPU["cache"], _GPU["ts"] = res, time.time()
    return res


_BAR = re.compile(r"\d+%\||\d+/\d+ \[|it/s\]|s/it\]")           # tqdm-Fortschrittszeilen
_LOSS = re.compile(r"'loss':\s*'?([0-9.]+)")


def tail_log(n=42):
    """Tail der zuletzt geänderten *.log im log-dir (generisch, folgt der aktiven Phase).
    Filtert tqdm-Balken raus (zeigt nur die aktuelle Fortschrittszeile), extrahiert Loss-Serie."""
    d = CFG.get("log_dir") or CFG.get("data_dir")
    if not d:
        return {}
    logs = sorted(pathlib.Path(d).glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return {}
    p = logs[-1]
    raw = p.read_bytes()[-65536:].decode("utf-8", "replace")
    segs = [s.strip() for s in re.split(r"[\r\n]+", raw) if s.strip()]
    clean = [s for s in segs if not _BAR.search(s)]
    last_bar = next((s for s in reversed(segs) if _BAR.search(s)), None)
    lines = clean[-n:]
    eta = sit = None
    if last_bar:
        lines = lines + ["▸ " + last_bar]
        me = re.search(r"<([0-9:]+)", last_bar)          # tqdm: [elapsed<REMAINING, x s/it]
        if me:
            eta = me.group(1)
        ms = re.search(r"([0-9.]+)s/it", last_bar)
        if ms:
            sit = float(ms.group(1))
    loss = [float(m) for s in segs for m in _LOSS.findall(s)][-80:]
    return {"file": p.name, "age_secs": round(time.time() - p.stat().st_mtime),
            "lines": lines, "loss": loss, "eta": eta, "sec_per_it": sit}


PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Progress</title>
<style>
:root{--card:rgba(17,30,52,.58);--line:rgba(90,130,185,.28);--cyan:#22d3ee;--green:#10b981;
--purple:#8b5cf6;--blue:#3b9cff;--txt:#eaf2ff;--mut:#8ea6c8}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:#04060c;color:var(--txt);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
#bg{position:fixed;inset:0;width:100%;height:100%;z-index:0;display:block}
.wrap{position:relative;z-index:1;max-width:1040px;margin:0 auto;padding:26px 22px 60px}
h1{font-size:19px;font-weight:680;margin:0 0 2px;letter-spacing:.2px}
.sub{color:var(--mut);font-size:12px;margin-bottom:18px}
.sec{margin:22px 0 10px;color:var(--cyan);font-size:11px;letter-spacing:.12em;text-transform:uppercase;text-shadow:0 0 12px rgba(34,211,238,.35)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:8px 0;
backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);box-shadow:0 8px 34px rgba(0,0,0,.4)}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.name{font-weight:600;font-size:14px}.meta{color:var(--mut);font-size:12px;white-space:nowrap}
.big{font-variant-numeric:tabular-nums}.big b{font-size:20px}.tgt{color:var(--mut);font-size:13px}
.bar{height:8px;background:rgba(4,12,24,.7);border-radius:6px;margin-top:10px;overflow:hidden}
.fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--blue),var(--cyan));border-radius:6px;width:0;
transition:width .5s ease;box-shadow:0 0 14px rgba(59,156,255,.55)}
.badge{font-size:11px;padding:2px 9px;border-radius:999px;border:1px solid var(--line);color:var(--mut)}
.badge.live{color:#04140c;background:var(--green);border-color:var(--green);font-weight:700;box-shadow:0 0 14px rgba(16,185,129,.6)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 1.1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.empty{color:var(--mut);font-size:13px;font-style:italic;padding:6px 2px}
.rmhead{font-size:13px;color:var(--txt);margin:2px 2px 10px}.rmhead b{color:var(--cyan)}
.rm{display:flex;flex-direction:column}
.step{display:flex;align-items:flex-start;gap:12px;padding:8px 2px;position:relative}
.step .ic{flex:0 0 auto;width:22px;height:22px;border-radius:50%;display:grid;place-items:center;font-size:12px;font-weight:700;border:1.5px solid var(--line);color:var(--mut);background:rgba(4,12,24,.6);z-index:1}
.step.done .ic{background:var(--green);border-color:var(--green);color:#04140c}
.step.current .ic{background:var(--cyan);border-color:var(--cyan);color:#04121a;box-shadow:0 0 16px rgba(34,211,238,.8);animation:pulse 1.3s infinite}
.step:not(:last-child):before{content:'';position:absolute;left:10.5px;top:26px;bottom:-4px;width:1.5px;background:var(--line)}
.step.done:not(:last-child):before{background:rgba(16,185,129,.5)}
.step .ph{font-size:14px;font-weight:600;line-height:22px}.step.todo .ph{color:var(--mut);font-weight:500}
.step .dt{color:var(--mut);font-size:12px;margin-top:1px}
.foot{color:var(--mut);font-size:11px;margin-top:26px;border-top:1px solid var(--line);padding-top:10px}
</style></head><body>
<canvas id="bg"></canvas>
<div class="wrap"><h1 id="title">Progress</h1><div class="sub" id="sub">lädt …</div><div id="root"></div>
<div class="foot">RadeonForge Live-Dashboard · Auto-Refresh 3 s · <code>/api/progress</code></div></div>
<script>
/* ===================== 3D-Knotenwolke (Canvas, iridescent, langsam) ===================== */
const cv=document.getElementById('bg'),cx=cv.getContext('2d');let W,H,DPR;
function resize(){DPR=Math.min(window.devicePixelRatio||1,2);W=cv.width=innerWidth*DPR;H=cv.height=innerHeight*DPR;}
addEventListener('resize',resize);resize();
const N=66,nodes=[],edges=[];
for(let i=0;i<N;i++){const t=i/N,phi=Math.acos(1-2*t),th=Math.PI*(1+Math.sqrt(5))*i,r=(i%2?1.06:0.64)+(i%5)*0.03;
 nodes.push({x:Math.cos(th)*Math.sin(phi)*r,y:Math.sin(th)*Math.sin(phi)*r,z:Math.cos(phi)*r,hj:(i*97)%16});}
for(let i=0;i<N;i++){const d=nodes.map((n,j)=>[(n.x-nodes[i].x)**2+(n.y-nodes[i].y)**2+(n.z-nodes[i].z)**2,j]).sort((a,b)=>a[0]-b[0]);
 for(let k=1;k<=3;k++){const j=d[k][1];if(i<j)edges.push([i,j]);}}
let pulses=[],trainingActive=false;
function hsl(h,s,l,a){return 'hsla('+(((h%360)+360)%360)+','+s+'%,'+l+'%,'+a+')';}
function frame(now){
 cx.clearRect(0,0,W,H);cx.globalCompositeOperation='lighter';
 const hd=now*0.0012;                                          /* sehr langsamer Farb-Drift */
 const cxp=W*(0.5+Math.sin(now*0.000022)*0.12),cyp=H*(0.5+Math.cos(now*0.000017)*0.08),scale=Math.min(W,H)*0.32,CAM=4.2;
 const ay=now*0.000017,ax=Math.sin(now*0.000012)*0.25;        /* sehr langsame Rotation, große Kameradistanz */
 const ca=Math.cos(ay),sa=Math.sin(ay),cb=Math.cos(ax),sb=Math.sin(ax);
 const proj=nodes.map(n=>{let x=n.x*ca-n.z*sa,z=n.x*sa+n.z*ca,y=n.y;let y2=y*cb-z*sb,z2=y*sb+z*cb;
  const persp=CAM/(CAM-z2);return{sx:cxp+x*scale*persp,sy:cyp+y2*scale*persp,depth:(z2+1.25)/2.5,persp,hue:230-x*46+hd+n.hj};});
 edges.forEach(([i,j])=>{const a=proj[i],b=proj[j],dep=(a.depth+b.depth)/2,al=0.05+0.14*dep,
  g=cx.createLinearGradient(a.sx,a.sy,b.sx,b.sy);g.addColorStop(0,hsl(a.hue,92,63,al));g.addColorStop(1,hsl(b.hue,92,63,al));
  cx.strokeStyle=g;cx.lineWidth=(0.7+1.5*dep)*DPR;cx.beginPath();cx.moveTo(a.sx,a.sy);cx.lineTo(b.sx,b.sy);cx.stroke();});
 const cp=0.6+0.4*Math.sin(now*0.0013*(trainingActive?1.8:1));
 [[1.2,0.09,278],[0.72,0.15,250],[0.34,0.30,222]].forEach(function(L){const cg=cx.createRadialGradient(cxp,cyp,0,cxp,cyp,scale*L[0]);
  cg.addColorStop(0,hsl(L[2]+hd,96,76,L[1]*cp));cg.addColorStop(0.5,hsl(L[2]+hd,90,60,L[1]*cp*0.5));cg.addColorStop(1,'hsla(230,90%,50%,0)');
  cx.fillStyle=cg;cx.beginPath();cx.arc(cxp,cyp,scale*L[0],0,7);cx.fill();});
 const core=cx.createRadialGradient(cxp,cyp,0,cxp,cyp,scale*0.17);
 core.addColorStop(0,'rgba(255,255,255,'+(0.9*cp)+')');core.addColorStop(1,'hsla(265,100%,70%,0)');
 cx.fillStyle=core;cx.beginPath();cx.arc(cxp,cyp,scale*0.17,0,7);cx.fill();
 proj.map((p,i)=>[p,i]).sort((a,b)=>a[0].depth-b[0].depth).forEach(([p])=>{const rad=(2+7*p.depth)*DPR,l=42+40*p.depth,a0=0.22+0.72*p.depth;
  const g=cx.createRadialGradient(p.sx,p.sy,0,p.sx,p.sy,rad*3);g.addColorStop(0,hsl(p.hue,95,Math.min(96,l+26),a0));
  g.addColorStop(0.45,hsl(p.hue,92,l,a0*0.5));g.addColorStop(1,hsl(p.hue,90,l,0));
  cx.fillStyle=g;cx.beginPath();cx.arc(p.sx,p.sy,rad*3,0,7);cx.fill();
  cx.fillStyle=hsl(p.hue,92,Math.min(97,l+30),a0);cx.beginPath();cx.arc(p.sx,p.sy,rad*0.6,0,7);cx.fill();});
 if(Math.random()<(trainingActive?0.10:0.03))pulses.push({e:edges[(Math.random()*edges.length)|0],p:0});
 pulses=pulses.filter(pl=>{pl.p+=0.004*(trainingActive?1.7:1);if(pl.p>1)return false;const a=proj[pl.e[0]],b=proj[pl.e[1]],
  x=a.sx+(b.sx-a.sx)*pl.p,y=a.sy+(b.sy-a.sy)*pl.p,g=cx.createRadialGradient(x,y,0,x,y,5*DPR);
  g.addColorStop(0,'rgba(255,255,255,.95)');g.addColorStop(1,hsl(205,100,70,0));cx.fillStyle=g;cx.beginPath();cx.arc(x,y,5*DPR,0,7);cx.fill();return true;});
 requestAnimationFrame(frame);}
requestAnimationFrame(frame);

/* ===================== Daten / Render ===================== */
function bar(p){return '<div class="bar"><div class="fill" style="width:'+(p||0)+'%"></div></div>';}
function age(s){if(s==null)return '—';if(s<90)return '<span class="dot"></span>vor '+s+' s';
 if(s<5400)return 'vor '+Math.round(s/60)+' min';return 'vor '+(s/3600).toFixed(1)+' h';}
function roadmap(items){if(!items||!items.length)return '';
 const done=items.filter(x=>x.status==='done').length,cur=items.find(x=>x.status==='current');
 let h='<div class="sec">Gesamtprozess</div><div class="card"><div class="rmhead"><b>'+done+'/'+items.length+
   '</b> Phasen erledigt'+(cur?(' · aktuell: <b>'+cur.phase+'</b>'):'')+'</div><div class="rm">';
 let n=0;items.forEach(x=>{n++;const ic=x.status==='done'?'✓':(x.status==='current'?'●':n);
  h+='<div class="step '+x.status+'"><div class="ic">'+ic+'</div><div><div class="ph">'+x.phase+'</div>'+
     (x.detail?('<div class="dt">'+x.detail+'</div>'):'')+'</div></div>';});
 return h+'</div></div>';}
function trCard(t){const live=t.active?'<span class="badge live">läuft · GPU</span>':'<span class="badge">'+(t.status||'—')+'</span>';
 const prog=(t.step!=null&&t.total_steps!=null)?('Step '+t.step+'/'+t.total_steps):(t.status||'');
 const ep=(t.epoch!=null)?(' · Epoche '+(+t.epoch).toFixed(2)+(t.epochs?('/'+t.epochs):'')):'';
 const loss=(t.loss!=null)?(' · loss '+(+t.loss).toFixed(4)):'';
 let h='<div class="card"><div class="row"><span class="name">'+t.name+'</span>'+live+'</div>'+
  '<div class="row" style="margin-top:6px"><span class="big">'+prog+ep+loss+'</span><span class="meta">'+age(t.age_secs)+'</span></div>';
 if(t.pct!=null)h+=bar(t.pct);if(t.note)h+='<div class="meta" style="margin-top:6px">'+t.note+'</div>';return h+'</div>';}
function dsCard(d){const live=d.active?'<span class="badge live">läuft</span>':'<span class="badge">idle</span>';
 const cnt='<span class="big"><b>'+d.count+'</b></span>'+(d.target?' <span class="tgt">/ '+d.target+'</span>':'');
 const pct=d.pct!=null?(' · '+d.pct+'%'):'';
 let h='<div class="card"><div class="row"><span class="name">'+d.label+'</span>'+live+'</div>'+
  '<div class="row" style="margin-top:6px"><span>'+cnt+pct+'</span><span class="meta">'+age(d.age_secs)+'</span></div>';
 if(d.target)h+=bar(d.pct);return h+'</div>';}
function grp(title,items,render,empty){const h='<div class="sec">'+title+'</div>';
 return items.length?h+items.map(render).join(''):h+'<div class="empty">'+empty+'</div>';}
async function tick(){try{
 const j=await(await fetch('/api/progress',{cache:'no-store'})).json();
 trainingActive=(j.trainings||[]).some(t=>t.active);
 document.title=j.title;document.getElementById('title').textContent=j.title;
 document.getElementById('sub').textContent=(j.subtitle?j.subtitle+'  ·  ':'')+'Stand '+j.generated_at+(trainingActive?'  ·  ● Training aktiv':'');
 let html=roadmap(j.pipeline);
 html+=grp('Modell-Training',j.trainings||[],trCard,'noch kein Lauf registriert');
 const groups={};(j.datasets||[]).forEach(d=>{(groups[d.group]=groups[d.group]||[]).push(d);});
 Object.keys(groups).forEach(g=>{html+=grp(g,groups[g],dsCard,'—');});
 document.getElementById('root').innerHTML=html;
}catch(e){document.getElementById('sub').textContent='Fehler: '+e;}}
tick();setInterval(tick,3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/progress"):
            body = json.dumps(collect(), ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        elif self.path.startswith("/api/log"):
            body = json.dumps(tail_log(), ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        elif self.path.startswith("/api/gpu"):
            body = json.dumps(gpu_stats(), ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        elif self.path.startswith("/gl"):
            body = (GL_FILE.read_bytes() if GL_FILE.exists()
                    else b"<p>progress_dashboard_gl.html fehlt</p>")
            ctype = "text/html; charset=utf-8"
        else:
            body = PAGE.encode("utf-8")
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="RadeonForge Live-Progress-Dashboard")
    ap.add_argument("--data-dir", help="Ordner mit *.jsonl (Datensatz-Fortschritt; optional)")
    ap.add_argument("--runs-dir", help="Ordner mit Trainings-Status-JSONs (Default: <data-dir>/_runs)")
    ap.add_argument("--log-dir", help="Ordner mit *.log fürs Live-Log (Default: <data-dir>)")
    ap.add_argument("--tracks", help="Manifest (JSON: title/subtitle/pipeline/tracks; optional)")
    ap.add_argument("--title", default="Training-Progress")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    title = args.title
    if args.tracks and pathlib.Path(args.tracks).exists():
        title = json.loads(pathlib.Path(args.tracks).read_text(encoding="utf-8")).get("title", title)
    runs_dir = args.runs_dir or (os.path.join(args.data_dir, "_runs") if args.data_dir else None)
    CFG.update({"data_dir": args.data_dir, "runs_dir": runs_dir, "log_dir": args.log_dir or args.data_dir,
                "tracks": args.tracks, "title": title})

    url = f"http://{args.host}:{args.port}/"
    print(f"Dashboard '{title}' auf  {url}   (Strg+C beendet)")
    print(f"  data-dir={args.data_dir}  runs-dir={runs_dir}  tracks={args.tracks}")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        ThreadingHTTPServer((args.host, args.port), H).serve_forever()
    except KeyboardInterrupt:
        print("\n(Dashboard beendet)")


if __name__ == "__main__":
    main()
