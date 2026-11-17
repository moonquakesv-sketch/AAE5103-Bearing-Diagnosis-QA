#!/bin/zsh
set -eu
SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h:h:h}"
print "This is a consumed historical evaluation batch. Recomputing saved result arithmetic only."
exec "${PYTHON:-python3}" -B "$PROJECT_DIR/scripts/recompute_submission_results.py"
