#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"

# Keep an existing environment; never remove a user's virtualenv during setup.
VENV_DIR="${VENV_DIR:-.venv}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
if [ ! -e "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    printf 'Expected a Python virtual environment at %s\n' "$VENV_DIR" >&2
    exit 1
fi
VENV_DIR="$(cd -- "$VENV_DIR" && pwd)"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url "$PYTORCH_INDEX_URL"
python -m pip install -e ".[test,evaluation]"
printf 'Environment ready. Activate it with: source %q\n' "$VENV_DIR/bin/activate"
echo "Prepare data once with: python prepare.py --dataset uniref50 --output-dir data/uniref50"
