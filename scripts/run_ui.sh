#!/usr/bin/env bash
# Start a local LiveKit server, the agent and the web UI together. Ctrl+C stops all three.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if ! command -v livekit-server >/dev/null; then
  echo "livekit-server not found. Install it with 'brew install livekit', or run it in Docker:"
  echo "  docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev --bind 0.0.0.0"
  exit 1
fi

if lsof -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8080 is already in use. Is run_ui.sh already running in another terminal?"
  echo "Stop it with Ctrl+C there (two copies would register two agents)."
  exit 1
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if lsof -iTCP:7880 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "A LiveKit server is already listening on :7880; using it."
else
  livekit-server --dev --bind 127.0.0.1 >logs/livekit-server.log 2>&1 &
  pids+=($!)
  sleep 1
fi

uv run agent.py dev &
pids+=($!)
uv run web/server.py &
pids+=($!)

sleep 3
echo
echo "Open http://localhost:8080 and click 'Start conversation'."
if [ -z "${NO_BROWSER:-}" ] && command -v open >/dev/null; then
  open http://localhost:8080
fi
wait
