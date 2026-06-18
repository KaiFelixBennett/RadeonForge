#!/usr/bin/env bash
# Regenerate all brand-neutral visual assets from the live demo dashboard:
#   assets/dashboard/*.png + dashboard.gif   (screenshots + live GIF)
#   assets/hero.png, social-preview.png, before-after.png   (branding cards)
# Needs: Node (Playwright auto-installs Chromium) + a system ffmpeg on PATH (for the GIF).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PORT="${1:-9137}"
PY="$(command -v python || command -v python3)"

[ -d "$HERE/node_modules" ]              || ( cd "$HERE" && npm install --no-audit --no-fund )
[ -d "$REPO/tools/branding/node_modules" ] || ( cd "$REPO/tools/branding" && npm install --no-audit --no-fund )

echo "▸ starting live demo on :$PORT (slow tick -> calm GIF)"
"$PY" "$REPO/scripts/demo_dashboard.py" --port "$PORT" --no-open --tick 2.5 --start-step 88 &
DEMO_PID=$!
trap 'kill $DEMO_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/api/progress" >/dev/null 2>&1 && break; sleep 0.5; done

echo "▸ capturing dashboard assets"
node "$HERE/capture.mjs" "http://127.0.0.1:$PORT" "$REPO/assets/dashboard"
echo "▸ rendering branding cards"
node "$REPO/tools/branding/render.mjs"
echo "✓ assets regenerated -> $REPO/assets"
