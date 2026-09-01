#!/bin/bash
# Serve visualizer/ over HTTP so it can be viewed from your local browser while
# SSHed into this machine. data.js is tens of MB, so file:// works too, but a
# server avoids browser file:// restrictions on some setups.
#
# Usage (on this machine, over your existing SSH session):
#   ./view_visualizer.sh [port]
#
# Then, from your LOCAL machine, forward the port over SSH:
#   ssh -L 8000:localhost:8000 <user>@<this-host>
# and open http://localhost:8000/visualizer.html in your browser.
#
# If you're already on a multi-hop SSH session, add -L 8000:localhost:8000 to
# each hop, or just re-connect once with the full chain and that one flag.
set -euo pipefail

cd "$(dirname "$0")/visualizer"

PORT="${1:-8000}"

if [ ! -f data.js ]; then
    echo "WARNING: visualizer/data.js not found — run precompute_visualizer.py first." >&2
fi

# Reuse port if an old http.server is still bound (common after a dropped SSH session).
if fuser "${PORT}/tcp" >/dev/null 2>&1; then
    echo "Port ${PORT} in use — stopping existing listener..."
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 1
fi

echo "Serving visualizer/ at http://localhost:${PORT}/visualizer.html"
echo "From your local machine: ssh -L ${PORT}:localhost:${PORT} $(whoami)@$(hostname -f 2>/dev/null || hostname)"
echo "(Ctrl+C to stop)"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
