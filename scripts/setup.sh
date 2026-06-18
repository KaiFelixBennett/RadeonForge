#!/usr/bin/env bash
# RadeonForge — one-command training-env bootstrap for AMD RDNA4 (gfx1201) on WSL2/Linux.
#
# Automates the part everyone gets stuck on (RUNBOOK §2): the librocdxg bridge, the
# ROCm-Triton --find-links trap, and the bitsandbytes --no-deps trap. Idempotent —
# safe to re-run. It does NOT install the ROCm base stack itself (that needs distro
# choices + sudo + a reboot-free WSL driver); if ROCm is missing it tells you the exact
# command from the RUNBOOK and stops.
#
#   bash scripts/setup.sh                 # venv at ~/.venvs/rdna4-train
#   bash scripts/setup.sh --venv ~/.venvs/myenv
#   bash scripts/setup.sh --no-librocdxg  # skip the WSL bridge .deb (native Linux)
#
# Pinned + dated (2026-06-15) — versions rot on this hardware; re-verify against VERSIONS.md.
set -euo pipefail

VENV="${HOME}/.venvs/rdna4-train"
INSTALL_LIBROCDXG=1
ROCM_FIND_LINKS="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/"
TORCH_WHL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
BNB_WHL="https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"
LIBROCDXG_DEB="https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2;;
    --no-librocdxg) INSTALL_LIBROCDXG=0; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

say() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()  { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "Run this inside WSL2 (Ubuntu) or native Linux — not from Windows. See RUNBOOK §1."

# ── 1. librocdxg bridge (WSL2 ↔ GPU) — the MISSING-PACKAGE gotcha (RUNBOOK §2b) ──
if [ "$INSTALL_LIBROCDXG" = 1 ]; then
  if [ -e /opt/rocm/lib/librocdxg.so ] || ldconfig -p 2>/dev/null | grep -q librocdxg; then
    ok "librocdxg present"
  else
    say "Installing librocdxg bridge (.deb) — required for rocminfo under WSL2"
    tmp="$(mktemp -d)"; wget -qO "$tmp/rocdxg-roct.deb" "$LIBROCDXG_DEB" \
      && sudo apt install -y "$tmp/rocdxg-roct.deb" && ok "librocdxg installed" \
      || die "librocdxg install failed — see RUNBOOK §2b"
  fi
fi

# ── 2. ROCm present? (we don't install the base stack — that's a documented manual step) ──
say "Checking ROCm / GPU visibility"
if command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -qi gfx; then
  ok "rocminfo sees a GPU: $(rocminfo 2>/dev/null | grep -i -m1 'gfx' | tr -s ' ')"
else
  die "ROCm not found / no GPU visible. Install it first (RUNBOOK §2a):
     wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
     sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
     sudo amdgpu-install -y --usecase=wsl,rocm,hip --no-dkms
   then re-run this script."
fi

# ── 3. Python venv ──
say "Python venv at $VENV"
command -v python3.12 >/dev/null 2>&1 || die "python3.12 missing → sudo apt install -y python3.12 python3.12-venv python3-pip git"
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null && ok "pip/wheel up to date"

# ── 4. PyTorch-ROCm (needs --find-links for the ROCm-only Triton wheel; RUNBOOK §2c) ──
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  ok "torch-ROCm already working: $(python -c 'import torch;print(torch.__version__)')"
else
  say "Installing PyTorch-ROCm 7.2 (+ ROCm Triton via --find-links)"
  pip install --find-links "$ROCM_FIND_LINKS" "$TORCH_WHL" \
    && ok "torch installed" || die "torch-ROCm install failed — see RUNBOOK §2c"
fi

# ── 5. Training libraries ──
say "Installing training libraries (transformers/peft/accelerate/trl/datasets …)"
pip install "transformers>=5.5.0" peft accelerate trl datasets pillow pyyaml >/dev/null && ok "HF training stack"

# ── 6. bitsandbytes 4-bit — ALWAYS --no-deps (else it pulls CUDA-torch & nukes ROCm; §2e) ──
if python -c "import bitsandbytes" 2>/dev/null; then
  ok "bitsandbytes present"
else
  say "Installing bitsandbytes (4-bit) with --no-deps"
  pip install --no-deps "$BNB_WHL" && ok "bitsandbytes installed" || die "bnb install failed — see RUNBOOK §2e"
fi

# ── 7. Verify: doctor + smoke test (the real reproducibility guarantee) ──
say "Verifying the environment"
bash "$HERE/scripts/doctor.sh" || die "doctor.sh reported problems — fix them before training"
say "Smoke test: 50-step 4-bit QLoRA (fails loudly if loss→0/NaN)"
python "$HERE/scripts/smoke_test.py" \
  && ok "Smoke test passed — this box can train. Next: make train" \
  || die "Smoke test FAILED — see docs/troubleshooting.md (likely a paged-optimizer or SDPA trap)"
