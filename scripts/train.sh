#!/bin/bash
set -e
PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python"


"$PYTHON" src/train.py 
