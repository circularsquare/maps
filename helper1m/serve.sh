#!/bin/sh
# Serve the helper1m viewer over HTTP (file:// can't fetch the GeoJSON).
# Usage: ./serve.sh [port]    — default 8000, then open http://localhost:8000/
cd "$(dirname "$0")" || exit 1
PORT="${1:-8000}"
echo "helper1m: http://localhost:$PORT/"
exec python -m http.server "$PORT"
