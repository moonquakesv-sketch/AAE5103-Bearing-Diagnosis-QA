#!/bin/zsh
set -eu
DEMO_DIR="${0:A:h}"
exec "${PYTHON:-python3}" -B "$DEMO_DIR/launch.py" "$@"
