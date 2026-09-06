"""Representation arithmetic and compatibility checks."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hera_systematics_model.residuals import TRANSFORMS, INVERSES


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_inverse_recovers_signed_power(name):
    power = np.array([-4.0, 0.0, 5.0, 100.0])
    ideal = np.array([1.0, -2.0, 0.0, 90.0])
    pn = np.array([2.0, 1.0, 3.0, 4.0])
    params = {"floor": 10.0} if name == "log_ratio" else {}
    transformed = TRANSFORMS[name](power, ideal, pn, **params)
    np.testing.assert_allclose(
        INVERSES[name](transformed, ideal, pn, **params), power, atol=1e-12
    )


def test_log_retains_undefined_inputs():
    result = TRANSFORMS["log_ratio"](np.array([-1., 0., 2.]), np.ones(3))
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2])


def test_legacy_selfcheck():
    script = Path(__file__).parents[1] / "scripts/analysis/check_residuals.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no failures" in result.stdout
