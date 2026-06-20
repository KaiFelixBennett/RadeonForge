#!/usr/bin/env bash
# RadeonForge doctor — checks the Track A (WSL2 + ROCm) environment for gfx1201 fine-tuning.
# Run inside WSL2 (Ubuntu) with your training venv active.
set -uo pipefail

pass() { echo "  ✅ $1"; }
warn() { echo "  ⚠️  $1"; }
fail() { echo "  ❌ $1"; ERR=1; }
ERR=0
PY="$(command -v python || command -v python3 || echo python)"   # fresh Ubuntu has python3, not python

echo "== RadeonForge doctor =="

echo "[1] GPU visible to ROCm"
if command -v rocminfo >/dev/null 2>&1; then
  # NB: rocminfo can exit non-zero on WSL yet still print the GPU — capture its output and
  # grep the string (don't pipe rocminfo directly, or `pipefail` mis-reads a working GPU).
  ROCM_OUT="$(rocminfo 2>/dev/null)" || true
  if printf '%s' "$ROCM_OUT" | grep -qi gfx1201; then pass "rocminfo lists gfx1201 (RDNA4)"
  else fail "rocminfo found but gfx1201 NOT listed — install rocdxg-roct (librocdxg) + 'export HSA_ENABLE_DXG_DETECTION=1'; ensure an Adrenalin WSL2 driver provides /dev/dxg"; fi
else fail "rocminfo not found — ROCm not installed (see docs/track-a-wsl2-rocm.md)"; fi

echo "[2] Required env"
[ "${HSA_ENABLE_DXG_DETECTION:-}" = "1" ] && pass "HSA_ENABLE_DXG_DETECTION=1" || warn "HSA_ENABLE_DXG_DETECTION not set to 1 (WSL GPU discovery)"
[ -n "${PYTORCH_ALLOC_CONF:-}" ] && pass "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" || warn "PYTORCH_ALLOC_CONF not set (recommend expandable_segments:True)"

echo "[3] PyTorch + ROCm"
"$PY" - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok else "—"
    tag = "✅" if (ok and "rocm" in torch.__version__.lower()) else ("⚠️" if ok else "❌")
    print(f"  {tag} torch {torch.__version__} | cuda_available={ok} | device={name}")
    if not ok: sys.exit(3)
    if "rocm" not in torch.__version__.lower():
        print("  ⚠️  torch is not a ROCm build — a CUDA wheel will NOT use the AMD card")
except Exception as e:
    print(f"  ❌ torch import failed: {e}"); sys.exit(3)
PY
[ $? -ne 0 ] && ERR=1

echo "[4] bitsandbytes ROCm backend"
"$PY" - <<'PY'
try:
    import bitsandbytes as bnb, torch
    x = torch.randn(8, 8, device="cuda")
    print(f"  ✅ bitsandbytes {bnb.__version__} imported; CUDA/HIP tensor alloc ok")
except Exception as e:
    print(f"  ❌ bitsandbytes problem: {e}  (use the 1.33.7.preview wheel or compile -DBNB_ROCM_ARCH=gfx1201)")
PY

echo "[5] HF stack versions"
"$PY" - <<'PY'
for m in ("transformers","peft","trl","accelerate","datasets"):
    try:
        mod = __import__(m); print(f"  ✅ {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  ❌ {m} missing: {e}")
PY

echo
if [ "$ERR" = "1" ]; then
  echo "doctor: PROBLEMS found ↑  — fix before training. Then run: python scripts/smoke_test.py"
  exit 1
else
  echo "doctor: looks OK. Next: python scripts/smoke_test.py  (proves the loss actually falls)"
fi
