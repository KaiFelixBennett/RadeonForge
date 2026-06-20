#!/usr/bin/env python
"""make_dataset.py — build a chat-JSONL fine-tuning set THREE ways, with a review step.

Fine-tuning is "show the student model thousands of (input -> ideal output) examples".
The *student* is the model you train (gemma-4, etc.). A *teacher* is a stronger model that
writes/labels examples for you (knowledge distillation). This tool makes that easy:

  generate  describe a task (+ a few seeds) -> a TEACHER writes N diverse examples
  label     you provide inputs              -> a TEACHER writes the ideal output for each
  convert   you already have rough data     -> reformat to chat-JSONL (no teacher needed)

Every mode then writes a **review page** (open in a browser, keep/drop/edit each example,
download the cleaned set) — because the #1 rule is *review your labels before you train*.

Teacher providers (pluggable, --provider): ollama (local, free), anthropic, openai, mock (offline).
Output = one chat per line, exactly what examples/.../train_qlora.py expects:
  {"messages":[{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}]}

Examples
  python scripts/make_dataset.py generate --example --provider mock --n 12 --out data/example.jsonl
  python scripts/make_dataset.py generate --task "answer cooking questions" \
      --system "You are a concise cooking assistant." --n 50 --provider ollama --model llama3.1
  python scripts/make_dataset.py label --inputs my_questions.txt \
      --system "You are a helpful support agent." --provider anthropic --model claude-opus-4-8 --out data/sft.jsonl
  python scripts/make_dataset.py convert --in pairs.csv --user-col question --assistant-col answer --out data/sft.jsonl
  python scripts/make_dataset.py review --in data/sft.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.request
import webbrowser

DEFAULT_SYSTEM = "You are a helpful, concise assistant."
EXAMPLE_TASK = ("answer everyday how-to and explanation questions clearly and concisely, "
                "in a friendly, practical tone")
EXAMPLE_SEEDS = [
    {"user": "How do I reverse a list in Python?",
     "assistant": "Use slicing: `my_list[::-1]` returns a reversed copy, or call `my_list.reverse()` to reverse it in place."},
    {"user": "What's the difference between TCP and UDP?",
     "assistant": "TCP is connection-oriented and reliable (it guarantees order and delivery); UDP is connectionless and faster but makes no such guarantees. Use TCP for files/web, UDP for live video/games."},
]


# ───────────────────────── teacher providers (stdlib only) ─────────────────────────
def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def complete(provider: str, model: str, system: str, user: str, *, kind: str = "answer",
             temperature: float = 0.7, timeout: int = 120) -> str:
    """Return the teacher's text. `kind` is 'pair' (expect JSON {user,assistant}) or 'answer'."""
    if provider == "mock":
        return _mock(kind, user)
    if provider == "ollama":
        d = _post("http://127.0.0.1:11434/api/chat",
                  {"model": model, "stream": False, "options": {"temperature": temperature},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                  {}, timeout)
        return d["message"]["content"]
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit("Set ANTHROPIC_API_KEY for --provider anthropic")
        d = _post("https://api.anthropic.com/v1/messages",
                  {"model": model, "max_tokens": 1024, "temperature": temperature,
                   "system": system, "messages": [{"role": "user", "content": user}]},
                  {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout)
        return "".join(b.get("text", "") for b in d.get("content", []))
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY") or sys.exit("Set OPENAI_API_KEY for --provider openai")
        d = _post("https://api.openai.com/v1/chat/completions",
                  {"model": model, "temperature": temperature,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                  {"Authorization": f"Bearer {key}"}, timeout)
        return d["choices"][0]["message"]["content"]
    sys.exit(f"unknown provider: {provider} (use ollama | anthropic | openai | mock)")


def _mock(kind: str, user: str) -> str:
    """Deterministic offline 'teacher' so the whole pipeline is testable without a model/key."""
    h = int(hashlib.sha1(user.encode("utf-8")).hexdigest(), 16)
    n = h % 1000
    if kind == "pair":
        topics = ["resetting a forgotten password", "reading a CSV file in Python",
                  "writing a polite refund reply", "how DNS resolves a domain",
                  "a haiku about autumn rain", "converting cups to grams in baking",
                  "git rebase vs. merge", "a regex that matches email addresses",
                  "why the sky is blue", "keyboard shortcuts for faster editing"]
        t = topics[h % len(topics)]
        return json.dumps({"user": f"Can you help me with {t}? (variant #{n})",
                           "assistant": f"Sure — here's a clear, practical explanation of {t}. "
                                        f"[mock teacher output #{n}: replace me with a real --provider]"})
    return f"[mock] A concise, helpful reply to: {user.strip()[:90]} (#{n})"


# ───────────────────────── helpers ─────────────────────────
def to_msgs(system: str, user: str, assistant: str) -> dict:
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def parse_pair(raw: str):
    """Pull the first {"user":..,"assistant":..} JSON object out of a teacher response."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        if isinstance(d, dict) and "user" in d and "assistant" in d:
            return {"user": str(d["user"]).strip(), "assistant": str(d["assistant"]).strip()}
    except Exception:
        return None
    return None


def dedup(records: list) -> list:
    seen, out = set(), []
    for r in records:
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        k = re.sub(r"\s+", " ", u.lower()).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def write_jsonl(path: str, records: list) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} examples -> {path}")


def read_jsonl(path: str) -> list:
    return [json.loads(ln) for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


# ───────────────────────── modes ─────────────────────────
def mode_generate(a) -> None:
    system = a.system or DEFAULT_SYSTEM
    task = a.task
    seeds = []
    if a.example:
        task = a.task or EXAMPLE_TASK
        system = a.system or DEFAULT_SYSTEM
        seeds = EXAMPLE_SEEDS
    elif a.seeds:
        seeds = [{"user": m["messages"][1]["content"], "assistant": m["messages"][2]["content"]}
                 for m in read_jsonl(a.seeds)] if a.seeds.endswith(".jsonl") else []
    if not task:
        sys.exit("generate needs --task \"...\" (or --example for a built-in demo)")
    seed_txt = ""
    if seeds:
        seed_txt = "Here are example pairs to match the style:\n" + "\n".join(
            f'- user: {s["user"]}\n  assistant: {s["assistant"]}' for s in seeds[:4]) + "\n"
    print(f"generate · provider={a.provider} model={a.model} n={a.n}")
    recent, out = [], []
    for i in range(a.n):
        avoid = "; ".join(recent[-8:]) or "(none yet)"
        prompt = (f"You build training data for an assistant.\n"
                  f"The assistant's task: {task}\n"
                  f"The assistant uses this system prompt:\n\"\"\"{system}\"\"\"\n{seed_txt}"
                  f"Write ONE new, realistic, DIVERSE example (#{i + 1}) as STRICT JSON: "
                  f'{{"user": "<a user message>", "assistant": "<the ideal reply>"}}. '
                  f"Vary topic, phrasing and difficulty. Do NOT repeat these users: {avoid}. "
                  f"Return ONLY the JSON object.")
        raw = complete(a.provider, a.model, "You output only one valid JSON object.", prompt,
                       kind="pair", temperature=a.temperature)
        pair = parse_pair(raw)
        if not pair:
            print(f"  · skipped #{i + 1} (no valid JSON from teacher)")
            continue
        recent.append(pair["user"][:60])
        out.append(to_msgs(system, pair["user"], pair["assistant"]))
    out = dedup(out)
    write_jsonl(a.out, out)
    _review_and_open(a.out, a.no_open)


def mode_label(a) -> None:
    system = a.system or DEFAULT_SYSTEM
    inputs = _read_inputs(a.inputs)
    if not inputs:
        sys.exit(f"no inputs read from {a.inputs}")
    print(f"label · provider={a.provider} model={a.model} inputs={len(inputs)}")
    out = []
    for u in inputs:
        ans = complete(a.provider, a.model, system, u, kind="answer", temperature=a.temperature)
        out.append(to_msgs(system, u, ans.strip()))
    write_jsonl(a.out, out)
    _review_and_open(a.out, a.no_open)


def mode_convert(a) -> None:
    system = a.system or DEFAULT_SYSTEM
    rows = _read_table(a.inp)
    if not rows:
        sys.exit(f"no rows read from {a.inp}")
    # column auto-detection (common names) unless explicitly given
    cols = set().union(*(r.keys() for r in rows))
    uc = a.user_col or _first_present(cols, ["user", "instruction", "prompt", "question", "input"])
    ac = a.assistant_col or _first_present(cols, ["assistant", "response", "completion", "answer", "output"])
    if not uc or not ac:
        sys.exit(f"could not find user/assistant columns in {sorted(cols)} — pass --user-col/--assistant-col")
    print(f"convert · {len(rows)} rows · user='{uc}' assistant='{ac}'")
    out = []
    for r in rows:
        u, ans = r.get(uc), r.get(ac)
        if u in (None, "") or ans in (None, ""):
            continue
        s = r.get(a.system_col) if (a.system_col and r.get(a.system_col)) else system
        out.append(to_msgs(str(s), str(u), str(ans)))
    out = dedup(out)
    write_jsonl(a.out, out)
    _review_and_open(a.out, a.no_open)


def mode_review(a) -> None:
    _review_and_open(a.inp, a.no_open)


def _read_inputs(path: str) -> list:
    p = pathlib.Path(path)
    if p.suffix == ".jsonl":
        rows = read_jsonl(path)
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append(r.get("user") or r.get("instruction") or r.get("prompt") or r.get("input") or "")
            else:
                out.append(str(r))
        return [x for x in out if x]
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _read_table(path: str) -> list:
    p = pathlib.Path(path)
    if p.suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if p.suffix == ".jsonl":
        return read_jsonl(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else [d]
    sys.exit(f"convert: unsupported input {p.suffix} (use .csv / .jsonl / .json)")


def _first_present(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _review_and_open(jsonl_path: str, no_open: bool) -> None:
    records = read_jsonl(jsonl_path)
    title = pathlib.Path(jsonl_path).name
    html = REVIEW_HTML.replace("/*__DATA__*/", json.dumps(records, ensure_ascii=False)).replace("__TITLE__", title)
    out = pathlib.Path(jsonl_path).with_suffix(".review.html")
    out.write_text(html, encoding="utf-8")
    print(f"  review page -> {out}   (keep/drop/edit, then 'Download cleaned JSONL')")
    if not no_open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a chat-JSONL fine-tuning set (generate/label/convert) + review.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--provider", default="mock", choices=["ollama", "anthropic", "openai", "mock"])
        p.add_argument("--model", default="llama3.1")
        p.add_argument("--system", help="the system prompt your STUDENT model will use")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--out", default="data/dataset.jsonl")
        p.add_argument("--no-open", action="store_true")

    g = sub.add_parser("generate", help="teacher writes N diverse examples from a task description")
    common(g)
    g.add_argument("--task", help="plain-language description of what the assistant should do")
    g.add_argument("--seeds", help="optional seed examples (.jsonl in chat format) to match the style")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--example", action="store_true", help="use a built-in instruction->response demo task")
    g.set_defaults(func=mode_generate)

    l = sub.add_parser("label", help="teacher writes the ideal output for each of your inputs")
    common(l)
    l.add_argument("--inputs", required=True, help="a .txt (one input per line) or .jsonl with a 'user' field")
    l.set_defaults(func=mode_label)

    c = sub.add_parser("convert", help="reformat existing data (CSV/JSONL/JSON) into chat-JSONL (no teacher)")
    c.add_argument("--in", dest="inp", required=True)
    c.add_argument("--user-col"); c.add_argument("--assistant-col"); c.add_argument("--system-col")
    c.add_argument("--system"); c.add_argument("--out", default="data/dataset.jsonl"); c.add_argument("--no-open", action="store_true")
    c.set_defaults(func=mode_convert)

    r = sub.add_parser("review", help="(re)build the review page for an existing JSONL")
    r.add_argument("--in", dest="inp", required=True); r.add_argument("--no-open", action="store_true")
    r.set_defaults(func=mode_review)

    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args.func(args)


# ───────────────────────── review page (self-contained, data inlined) ─────────────────────────
REVIEW_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Review · __TITLE__</title>
<style>
:root{--bg:#04060c;--card:rgba(15,27,48,.6);--line:rgba(120,160,210,.2);--cyan:#36d4e8;--green:#37d39b;--red:#ff6b6b;--mut:#8da3c2;--txt:#eef5ff}
*{box-sizing:border-box}body{margin:0;background:#04060c;color:var(--txt);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.bar{position:sticky;top:0;z-index:5;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
 padding:14px 20px;background:rgba(6,12,22,.92);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
.bar h1{font-size:16px;margin:0;font-weight:750}.bar .sp{flex:1}
.stat{font-variant-numeric:tabular-nums;color:var(--mut);font-size:13px}.stat b{color:var(--txt)}
button{font:inherit;font-size:13px;font-weight:650;border:1px solid var(--line);border-radius:9px;padding:8px 13px;cursor:pointer;color:var(--txt);background:rgba(255,255,255,.05)}
button:hover{border-color:var(--cyan)}button.primary{background:linear-gradient(135deg,var(--cyan),#3b9cff);color:#04121a;border:none}
.wrap{max-width:1000px;margin:0 auto;padding:18px 20px 80px}
.sys{margin:10px 0 16px;font-size:12.5px;color:var(--mut)}
.card{border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;background:var(--card)}
.card.drop{opacity:.4;border-color:rgba(255,107,107,.5)}
.row{display:flex;gap:10px;align-items:center;margin-bottom:8px}
.idx{font-size:12px;color:var(--mut)}.role{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--cyan);margin:8px 0 3px}
.role.a{color:var(--green)}
textarea{width:100%;background:rgba(4,12,24,.6);color:var(--txt);border:1px solid var(--line);border-radius:8px;
 padding:9px 11px;font:inherit;font-size:14px;line-height:1.45;resize:vertical;min-height:42px}
.toggle{margin-left:auto}
.keepbtn{border-radius:999px;padding:6px 14px}.keepbtn.keep{background:rgba(55,211,155,.15);border-color:var(--green);color:#bff6df}
.keepbtn.dropped{background:rgba(255,107,107,.15);border-color:var(--red);color:#ffd0d0}
.foot{color:var(--mut);font-size:12px;text-align:center;margin-top:24px}
.hint{background:rgba(54,212,232,.07);border:1px solid var(--line);border-radius:12px;padding:12px 15px;color:#cfe0f5;font-size:13px;margin:12px 0}
</style></head><body>
<div class="bar">
  <h1>Review &amp; clean — __TITLE__</h1>
  <span class="stat"><b id="nkeep">0</b> kept / <b id="ntot">0</b> total</span>
  <span class="sp"></span>
  <button onclick="setAll(true)">Keep all</button>
  <button onclick="setAll(false)">Drop all</button>
  <button class="primary" onclick="download()">⤓ Download cleaned JSONL</button>
</div>
<div class="wrap">
  <div class="hint">Eyeball every example <b>before</b> you train — wrong-but-confident data trains a wrong-but-confident model.
  Edit the text inline, toggle <b>Keep/Drop</b>, then download the cleaned set and point your training config at it.</div>
  <div class="sys" id="sysline"></div>
  <div id="list"></div>
  <div class="foot">RadeonForge · make_dataset.py — your data never leaves this page (everything runs in your browser).</div>
</div>
<script>
const DATA = /*__DATA__*/;
const state = DATA.map(()=>({keep:true}));
function getMsg(rec,role){const m=(rec.messages||[]).find(x=>x.role===role);return m?m.content:'';}
function setAll(v){state.forEach(s=>s.keep=v);render();}
function esc(s){return (s==null?'':''+s)}
const sys0 = DATA.length? getMsg(DATA[0],'system'):'';
document.getElementById('sysline').innerHTML = sys0? ('<b>System prompt</b> (shared): '+esc(sys0).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))) : '';
function render(){
  const list=document.getElementById('list');
  list.innerHTML = DATA.map((rec,i)=>{
    const u=esc(getMsg(rec,'user')), a=esc(getMsg(rec,'assistant'));
    const dropped=!state[i].keep;
    return `<div class="card ${dropped?'drop':''}">
      <div class="row"><span class="idx">#${i+1}</span>
        <button class="keepbtn ${dropped?'dropped':'keep'} toggle" onclick="state[${i}].keep=!state[${i}].keep;render()">${dropped?'Dropped':'Keep'}</button></div>
      <div class="role">User</div><textarea oninput="DATA[${i}].messages.find(m=>m.role==='user').content=this.value" rows="2">${u}</textarea>
      <div class="role a">Assistant (target)</div><textarea oninput="DATA[${i}].messages.find(m=>m.role==='assistant').content=this.value" rows="3">${a}</textarea>
    </div>`;
  }).join('');
  document.getElementById('nkeep').textContent = state.filter(s=>s.keep).length;
  document.getElementById('ntot').textContent = DATA.length;
}
function download(){
  const kept = DATA.filter((_,i)=>state[i].keep);
  const blob = new Blob([kept.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/jsonl'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='__TITLE__'.replace(/\.jsonl$/,'')+'.clean.jsonl';a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1500);
}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
