#!/usr/bin/env python
"""Compatibility entrypoint for nested physical-time residual evaluation.

Inputs are versioned paired samples with an explicit matched ideal branch.
Run with --help for the current argument contract.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hera_systematics_model.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
