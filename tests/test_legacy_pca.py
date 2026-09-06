from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("whiten", ["none", "pn-median", "pn-cell"])
def test_legacy_pca_saves_every_direction_and_reconstructs_scaling(tmp_path, whiten):
    power = np.random.default_rng(2).normal(size=(48, 50))
    power[0] = 0.
    valid = np.ones(power.shape, bool)
    valid[2, -1] = False
    power[2, -1] = 1e100
    pn = np.linspace(1., 4., power.size).reshape(power.shape)
    source = tmp_path / "samples"
    source.mkdir()
    np.savez_compressed(source / "sum.aligned.spw00.npz", matrix=power, valid=valid, pn_eff=pn,
        time_grid=np.arange(48.), lst_grid=np.linspace(6., 7., 48) % (2*np.pi), cube_shape=[48, 5, 10],
        blp_lens=np.arange(5.), dlys=np.arange(10.), kperps=np.arange(5.), kparas=np.arange(10.))
    script = Path(__file__).parents[1] / "scripts/analysis/run_pca_aligned.py"
    command = [sys.executable, str(script), "--samples-dir", str(source), "--label", "sum",
               "--outdir", str(tmp_path / "pca"), "--whiten", whiten, "--variance-threshold", "1"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with np.load(tmp_path / "pca/sum.pca.spw00.npz") as z:
        assert z["components"].shape == (48, 50)
        assert z["n_for_threshold"][0] <= len(z["components"])
        assert not z["feature_mask"][-1]
        assert z["valid"][0].all()
        scaled = z["mean"] + z["scores"] @ z["components"]
        if whiten == "pn-median":
            scaled *= z["feature_scale"]
        elif whiten == "pn-cell":
            scaled *= z["pn_eff"]
        np.testing.assert_allclose(scaled[:, :-1], power[:, :-1], atol=1e-13)
    rejected = subprocess.run(command + ["--subtract-dir", str(source)], capture_output=True, text=True)
    assert rejected.returncode == 2 and "common contributors" in rejected.stderr
