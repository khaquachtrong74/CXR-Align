#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load config from .env
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env"

echo "============================================================"
echo "medalign-pipeline Environment Setup"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Venv: $VENV_DIR"

if command -v uv >/dev/null 2>&1; then
    echo "uv: $(uv --version)"
else
    echo "Installing uv..."
    pip install uv
fi

mkdir -p "$(dirname "$VENV_DIR")"
uv venv --python python "$VENV_DIR"
UV_NO_PROGRESS=1 uv pip install --python "$VENV_DIR/bin/python" -r "$PROJECT_ROOT/requirements.txt"

echo "============================================================"
echo "Dependencies installed: $VENV_DIR"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "No nvidia-smi detected"
fi