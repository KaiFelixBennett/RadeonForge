#!/usr/bin/env python
"""
A/B evidence generator — BASE (no training) vs FINE-TUNED, same held-out queries.

This is the proof that the fine-tune did something: it runs the identical prompt
set through (a) the untouched base model and (b) the fine-tuned model, scores
both, and writes BOTH a machine-readable JSON and a human-readable Markdown
report. The contrast (base rambles / wrong level vs tuned = terse correct JSON)
is the evidence.

Usage (WSL, training venv active):
  HSA_ENABLE_DXG_DETECTION=1 python examples/gemma4-12b-qlora/ab_test.py \
      --base google/gemma-4-E2B-it \
      --finetuned /root/aria-pilot/e2b-merged \
      --out-md   /mnt/e/Coding/RadeonForge/examples/gemma4-12b-qlora/PILOT_RESULTS.md \
      --out-json /mnt/e/Coding/RadeonForge/examples/gemma4-12b-qlora/pilot_ab_results.json

--finetuned may be a merged model dir OR pass --adapter <dir> to load base+LoRA.
"""
import argparse, gc, json, os, re
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.enable_flash_sdp(False)        # gfx1201: math SDPA only
torch.backends.cuda.enable_mem_efficient_sdp(False)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Must MATCH the system prompt the dataset was built with (train == inference).
SYS = (
    "You are a HiRAG query-complexity router. Return JSON only (no prose, no markdown): "
    '{"query_type": "overview|specific|deep_dive|comparison|search", '
    '"target_level": 0|1|2, "confidence": 0.0-1.0, "reasoning": "...", '
    '"keywords": ["..."], "requires_cross_doc": true|false}. '
    "target_level 0=document summaries (only for listing which documents exist), "
    "1=chapter/section or a document summary, 2=detail chunks (default)."
)

# Held-out queries (NOT training rows): (query, expected_level, expected_type)
QUERIES = [
    ("Welche Dokumente kennst du?",                       0, "overview"),
    ("Liste alle verfuegbaren Berichte auf",              0, "overview"),
    ("What documents do you have access to?",             0, "overview"),
    ("Zeige mir eine Uebersicht aller Dokumente",         0, "overview"),
    ("Vergleiche die Aussagen aus beiden Vertraegen",     0, "comparison"),
    ("Worin unterscheiden sich Bericht A und Bericht B?", 0, "comparison"),
    ("Fasse Kapitel 3 zusammen",                          1, "specific"),
    ("Gib mir eine Zusammenfassung des Jahresberichts",   1, "specific"),
    ("Summarize section 2 of the manual",                 1, "specific"),
    ("Erklaere das Saga-Pattern im Detail",               2, "deep_dive"),
    ("Wie funktioniert die Kuendigungsfrist genau?",      2, "deep_dive"),
    ("Explain the authentication flow step by step",      2, "deep_dive"),
    ("Was kostet der Sonntagsbrunch?",                    2, "search"),
    ("Welche Frist gilt fuer die Rueckgabe?",             2, "search"),
]

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract(text):
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run(model, tok, max_new=256):
    rows = []
    for q, exp_lvl, exp_type in QUERIES:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"Analyze this query:\n{q}"}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        n_in = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = out[0][n_in:]
        txt = tok.decode(gen, skip_special_tokens=True).strip()
        parsed = extract(txt)
        pred_lvl = parsed.get("target_level") if isinstance(parsed, dict) else None
        pred_type = parsed.get("query_type") if isinstance(parsed, dict) else None
        rows.append({
            "query": q, "expected_level": exp_lvl, "expected_type": exp_type,
            "raw": txt, "out_tokens": int(gen.shape[0]),
            "json_valid": isinstance(parsed, dict) and "target_level" in parsed,
            "pred_level": pred_lvl, "pred_type": pred_type,
            "level_correct": pred_lvl == exp_lvl,
            "type_correct": pred_type == exp_type,
        })
    return rows


def load(path, adapter=None):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def metrics(rows):
    n = len(rows)
    return {
        "n": n,
        "json_valid_rate": round(sum(r["json_valid"] for r in rows) / n, 3),
        "level_accuracy": round(sum(r["level_correct"] for r in rows) / n, 3),
        "type_accuracy": round(sum(r["type_correct"] for r in rows) / n, 3),
        "avg_out_tokens": round(sum(r["out_tokens"] for r in rows) / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", default=None, help="merged model dir")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (loaded on --base)")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--queries", default=None,
                    help="optional JSON: list of {\"q\":..,\"level\":..,\"type\":..} to override the built-in set")
    args = ap.parse_args()

    if args.queries:
        global QUERIES
        with open(args.queries, encoding="utf-8") as fh:
            QUERIES = [(d["q"], d["level"], d["type"]) for d in json.load(fh)]
        print(f"loaded {len(QUERIES)} queries from {args.queries}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)

    print("=== BASE (no training) ===", flush=True)
    m = load(args.base)
    base_rows = run(m, tok)
    del m; gc.collect(); torch.cuda.empty_cache()

    print("=== FINE-TUNED ===", flush=True)
    if args.finetuned:
        ft_tok = AutoTokenizer.from_pretrained(args.finetuned)
        m = load(args.finetuned)
    else:
        ft_tok = tok
        m = load(args.base, adapter=args.adapter)
    ft_rows = run(m, ft_tok)
    del m; gc.collect(); torch.cuda.empty_cache()

    base_m, ft_m = metrics(base_rows), metrics(ft_rows)
    data = {
        "task": "HiRAG query-complexity routing (Task A)",
        "base_model": args.base,
        "finetuned": args.finetuned or f"{args.base} + adapter:{args.adapter}",
        "gpu": torch.cuda.get_device_name(0),
        "metrics": {"base": base_m, "finetuned": ft_m},
        "queries": [{"base": b, "finetuned": f} for b, f in zip(base_rows, ft_rows)],
    }
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    write_md(args.out_md, data, base_rows, ft_rows, base_m, ft_m)
    print(f"\nBASE      level-acc {base_m['level_accuracy']:.0%}  json {base_m['json_valid_rate']:.0%}  ~{base_m['avg_out_tokens']:.0f} tok")
    print(f"FINETUNED level-acc {ft_m['level_accuracy']:.0%}  json {ft_m['json_valid_rate']:.0%}  ~{ft_m['avg_out_tokens']:.0f} tok")
    print(f"\nwrote {args.out_md}\nwrote {args.out_json}")


def cell(r):
    mark = "OK" if r["level_correct"] else "X"
    lvl = r["pred_level"] if r["pred_level"] is not None else "—"
    return f"L{lvl} {mark}" if r["json_valid"] else "no-JSON X"


def write_md(path, data, base_rows, ft_rows, base_m, ft_m):
    L = []
    L.append("# Pilot evidence — fine-tuning proof (base vs fine-tuned)\n")
    L.append(f"**Task:** {data['task']}  ")
    L.append(f"**Base model:** `{data['base_model']}`  ")
    L.append(f"**Fine-tuned:** `{data['finetuned']}`  ")
    L.append(f"**GPU:** {data['gpu']}  ")
    L.append("**Generated by:** `ab_test.py` (greedy decode, identical prompts, math SDPA)\n")
    L.append("> Same held-out queries through both models. The base model is the **control**; "
             "the difference is the effect of training.\n")

    L.append("## Summary\n")
    L.append("| Metric | Base (no training) | Fine-tuned | Δ |")
    L.append("|---|---|---|---|")
    def drow(name, key, pct=True):
        b, f = base_m[key], ft_m[key]
        fmt = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.0f}")
        return f"| {name} | {fmt(b)} | {fmt(f)} | {('+' if f>=b else '')}{fmt(f-b) if pct else int(f-b)} |"
    L.append(drow("target_level accuracy", "level_accuracy"))
    L.append(drow("query_type accuracy", "type_accuracy"))
    L.append(drow("valid-JSON rate", "json_valid_rate"))
    L.append(f"| avg output length (tokens) | {base_m['avg_out_tokens']:.0f} | {ft_m['avg_out_tokens']:.0f} | "
             f"{ft_m['avg_out_tokens']-base_m['avg_out_tokens']:+.0f} |")
    L.append(f"\n**{int(ft_m['level_accuracy']*ft_m['n'])}/{ft_m['n']}** queries routed correctly after training "
             f"vs **{int(base_m['level_accuracy']*base_m['n'])}/{base_m['n']}** before.\n")

    L.append("## Per-query A/B\n")
    L.append("| Query | Expected | Base | Fine-tuned |")
    L.append("|---|---|---|---|")
    for b, f in zip(base_rows, ft_rows):
        L.append(f"| {b['query']} | L{b['expected_level']} ({b['expected_type']}) | {cell(b)} | {cell(f)} |")

    L.append("\n## Side-by-side raw outputs (first 6)\n")
    for b, f in zip(base_rows[:6], ft_rows[:6]):
        L.append(f"### `{b['query']}`  — expected **L{b['expected_level']}**\n")
        L.append(f"**Base ({b['out_tokens']} tok):**\n```\n{b['raw'][:700]}\n```")
        L.append(f"**Fine-tuned ({f['out_tokens']} tok):**\n```\n{f['raw'][:700]}\n```\n")

    L.append("## Reproduce\n```bash")
    L.append("HSA_ENABLE_DXG_DETECTION=1 python examples/gemma4-12b-qlora/ab_test.py \\")
    L.append(f"    --base {data['base_model']} \\")
    L.append("    --finetuned <merged-dir> \\")
    L.append("    --out-md PILOT_RESULTS.md --out-json pilot_ab_results.json")
    L.append("```")
    L.append("\nFull machine-readable data (every output, both models): `pilot_ab_results.json`.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
