#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Python was not found. Install Python 3.8+ and retry."
  exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  "$PYTHON_CMD" -m venv .venv
fi

VENV_PY=".venv/bin/python"

echo "Installing dependencies..."
"$VENV_PY" -m pip install -r requirements.txt

APP_URL="http://127.0.0.1:5000"

(
  sleep 3
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$APP_URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$APP_URL" >/dev/null 2>&1 || true
  fi
) &

echo "Starting Doc2JSON backend..."
exec "$VENV_PY" app.py
