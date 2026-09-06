import json

import pytest

from hera_systematics_model.notebook import notebook_parameters


def test_spectral_outputs_are_explicit_and_scientific_settings_preserved(tmp_path):
    path = tmp_path / "source.ipynb"
    source = "SINGLE_BL_FILE: str = None\nOUT_PSPEC_FILE: str = None\nOUT_TAVG_PSPEC_FILE: str = None\nNINTERLEAVE: int = 4\nSAVE_RESULTS: bool = True"
    path.write_text(json.dumps({"cells": [{"metadata": {"tags": ["parameters"]}, "source": source}]}))
    result = notebook_parameters(path, {"NINTERLEAVE": 4}, tmp_path / "shared.uvh5", tmp_path)
    assert result["NINTERLEAVE"] == 4
    assert result["OUT_PSPEC_FILE"] == str(tmp_path / "spectrum.pspec.h5")
    assert result["OUT_TAVG_PSPEC_FILE"] == str(tmp_path / "spectrum.tavg.pspec.h5")
    with pytest.raises(ValueError, match="unknown"):
        notebook_parameters(path, {"OTHER": 0}, "input", tmp_path)
    with pytest.raises(ValueError, match="required"):
        notebook_parameters(path, {"SAVE_RESULTS": False}, "input", tmp_path)
