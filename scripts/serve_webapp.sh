#!/bin/bash
# Launch the LLMServingSim Web UI.
# Source: PLAN.md and webapp/config.py SIM_ENV (LD_LIBRARY_PATH + PATH).
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export LD_LIBRARY_PATH="/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"

PORT="${LLMSS_PORT:-8000}"
HOST="${LLMSS_HOST:-0.0.0.0}"

echo "Starting LLMServingSim Web UI on http://${HOST}:${PORT}"
exec python3 -m uvicorn webapp.app:app --host "$HOST" --port "$PORT" --reload
