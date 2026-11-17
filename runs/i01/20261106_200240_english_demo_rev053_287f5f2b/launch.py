"""Launch the submission with the caller's Python interpreter."""
from pathlib import Path
import os
import sys

PROJECT = Path(__file__).resolve().parents[3]
os.execv(sys.executable, [sys.executable, "-B", str(PROJECT / "app.py"), *sys.argv[1:]])
