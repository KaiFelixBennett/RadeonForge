#!/usr/bin/env python
"""
Minimal OpenAI-compatible serving for HF models that llama.cpp can't run yet
(e.g. Qwen3.5's hybrid Gated-DeltaNet + MoE arch). Loads a merged model in bf16
on the GPU and exposes /v1/chat/completions + /health — so an existing OpenAI
client (custom-rag) talks to it unchanged. Good enough for a small, high-value
router/specialist; for big models prefer llama.cpp/vLLM.

Usage (WSL, training venv; needs: pip install fastapi uvicorn):
  HSA_ENABLE_DXG_DETECTION=1 python scripts/serve_transformers.py \
      --model /root/aria-pilot/qwen08-merged --port 8099
"""
import os, time, argparse, re, torch
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.enable_flash_sdp(False)            # gfx1201: math SDPA
torch.backends.cuda.enable_mem_efficient_sdp(False)
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--max-new-tokens", type=int, default=160)
args = ap.parse_args()

print(f"loading {args.model} (bf16, cuda) ...", flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
print(f"ready: {args.model} on {torch.cuda.get_device_name(0)}", flush=True)

app = FastAPI()

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: list[Msg]
    max_tokens: int | None = None
    temperature: float | None = 0.0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    n_in = enc["input_ids"].shape[1]
    mnt = req.max_tokens or args.max_new_tokens
    do_sample = (req.temperature or 0) > 0
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=mnt, do_sample=do_sample,
                             temperature=(req.temperature or 1.0), pad_token_id=PAD)
    gen = out[0][n_in:]
    txt = tok.decode(gen, skip_special_tokens=True).strip()
    if "</think>" in txt:                                   # drop empty/thinking block
        txt = txt.split("</think>")[-1].strip()
    dt = time.time() - t0
    ntok = int(gen.shape[0])
    return {
        "object": "chat.completion",
        "model": os.path.basename(args.model.rstrip("/")),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": txt}}],
        "usage": {"prompt_tokens": int(n_in), "completion_tokens": ntok,
                  "total_tokens": int(n_in) + ntok},
        "timings": {"predicted_ms": round(dt * 1000, 1),
                    "predicted_per_second": round(ntok / dt, 1) if dt > 0 else None},
    }

uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
