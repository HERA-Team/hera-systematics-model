#!/usr/bin/env python
"""Render figures from a versioned diagnostic artifact.

Run with --help for the package command arguments.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hera_systematics_model.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["plot", *sys.argv[1:]]))
