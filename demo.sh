#!/usr/bin/env bash
# RadeonForge — one-command LIVE dashboard demo (no GPU, no install).
# Pure-stdlib Python; simulates a training run so you see the full UI in ~10 s.
#   ./demo.sh            # opens http://127.0.0.1:8765/gl
#   ./demo.sh 8770       # custom port
set -euo pipefail
PORT="${1:-8765}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)"
echo "RadeonForge live demo -> http://127.0.0.1:${PORT}/gl  (Ctrl+C to stop)"
exec "$PY" "$HERE/scripts/demo_dashboard.py" --port "$PORT" --open
