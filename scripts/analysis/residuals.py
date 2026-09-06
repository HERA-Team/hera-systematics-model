#!/usr/bin/env python
"""Compatibility import for the residual-transform API."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hera_systematics_model.residuals import *  # noqa: F401,F403,E402
