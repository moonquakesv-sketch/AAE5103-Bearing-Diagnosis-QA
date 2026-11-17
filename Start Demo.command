#!/bin/zsh
set -eu
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
exec "${PYTHON:-python3}" -B app.py "$@"
