import importlib.util
from pathlib import Path

import numpy as np
import pytest


def trim_module():
    path = Path(__file__).parents[1] / "scripts/analysis/trim_aligned_windows.py"
    spec = importlib.util.spec_from_file_location("trim_aligned_windows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_trimming_retains_measured_zero_and_rejects_missing_validity():
    module = trim_module()
    values = {"matrix": np.array([[0., 0.], [9., 9.], [1., np.nan]]),
              "valid": np.array([[True, True], [False, False], [True, True]]),
              "pn_eff": np.ones((3, 2))}
    mask = np.ones(2, bool)
    np.testing.assert_equal(module.empty_rows(values, mask), [1])
    values["pn_eff"][2, 0] = 0.
    np.testing.assert_equal(module.empty_rows(values, mask), [1, 2])
    del values["valid"]
    with pytest.raises(ValueError, match="explicit validity"):
        module.empty_rows(values, mask)
