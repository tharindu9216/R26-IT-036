#!/usr/bin/env bash
# Start the support node (Linux/macOS).
#
#   ./run.sh                first run: creates .venv, installs requirements
#   ./run.sh --skip-install later runs: straight to the server
#
# Host/port come from config.yaml (override the file with NODE_CONFIG).

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${NODE_CONFIG:-config.yaml}"
SKIP_INSTALL=0
[[ "${1:-}" == "--skip-install" ]] && SKIP_INSTALL=1

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv..."
    python3 -m venv .venv
    SKIP_INSTALL=0
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    echo "Installing requirements (this takes a while the first time)..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

export NODE_CONFIG="$CONFIG"
read -r NODE_HOST NODE_PORT < <(.venv/bin/python - <<'PY'
import os, yaml
config = yaml.safe_load(open(os.environ["NODE_CONFIG"], encoding="utf-8"))
print(config.get("host", "0.0.0.0"), config.get("port", 8010))
PY
)

echo
echo "Support node starting on http://${NODE_HOST}:${NODE_PORT}"
echo "Point the backend's remote_services at this machine's LAN address."
echo

exec .venv/bin/python -m uvicorn server:app --host "$NODE_HOST" --port "$NODE_PORT"
