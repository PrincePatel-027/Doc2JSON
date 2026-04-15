#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD=""

resolve_python_cmd() {
  if command -v python3 >/dev/null 2>&1 && python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    return 0
  fi

  if command -v python >/dev/null 2>&1 && python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1; then
    PYTHON_CMD="python"
    return 0
  fi

  return 1
}

run_with_privileges() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Automatic Python install needs sudo/root privileges. Please install Python 3.8+ manually and retry."
    exit 1
  fi
}

ensure_brew_on_path() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    return 0
  fi

  if [ -x /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
    return 0
  fi

  return 1
}

install_python_linux() {
  echo "Python 3.8+ not found. Installing Python..."

  if command -v apt-get >/dev/null 2>&1; then
    run_with_privileges apt-get update
    run_with_privileges apt-get install -y python3 python3-venv python3-pip
    return 0
  fi

  if command -v dnf >/dev/null 2>&1; then
    run_with_privileges dnf install -y python3 python3-pip
    return 0
  fi

  if command -v yum >/dev/null 2>&1; then
    run_with_privileges yum install -y python3 python3-pip
    return 0
  fi

  if command -v pacman >/dev/null 2>&1; then
    run_with_privileges pacman -Sy --noconfirm python python-pip
    return 0
  fi

  if command -v zypper >/dev/null 2>&1; then
    run_with_privileges zypper --non-interactive install python3 python3-pip
    return 0
  fi

  echo "No supported Linux package manager was found. Install Python 3.8+ manually and retry."
  exit 1
}

install_python_macos() {
  echo "Python 3.8+ not found. Installing Python..."

  if ! ensure_brew_on_path; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to install Homebrew automatically. Install Homebrew/Python manually and retry."
      exit 1
    fi

    echo "Homebrew not found. Installing Homebrew..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ensure_brew_on_path || {
      echo "Homebrew installation completed, but brew is still unavailable in this shell."
      echo "Run this script again."
      exit 1
    }
  fi

  brew install python
}

if ! resolve_python_cmd; then
  case "$(uname -s)" in
    Darwin)
      install_python_macos
      ;;
    Linux)
      install_python_linux
      ;;
    *)
      echo "Unsupported OS. Install Python 3.8+ manually and retry."
      exit 1
      ;;
  esac

  if ! resolve_python_cmd; then
    echo "Python installation did not make Python 3.8+ available in this shell. Run this script again."
    exit 1
  fi
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
