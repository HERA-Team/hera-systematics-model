import json
from dataclasses import replace

import numpy as np
import pytest

from hera_systematics_model.cli import main
from hera_systematics_model.evaluation import Evaluation
from test_samples import paired


def test_cli_verifies_real_sample_state_and_rejects_corruption(tmp_path, paired):
    path = tmp_path / "samples.npz"
    paired.save(path)
    assert main(["verify", str(path)]) == 0
    path.write_bytes(b"damaged")
    with pytest.raises(SystemExit) as error:
        main(["verify", str(path)])
    assert error.value.code == 2


def test_cli_incomplete_analysis_keeps_evidence_and_fails_exit(tmp_path, paired):
    source = tmp_path / "samples.npz"
    paired.save(source)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_rank": 0, "include_kernel": False}))
    output = tmp_path / "evaluation.npz"
    assert main(["evaluate", "--samples", str(source), "--config", str(config), "--output", str(output)]) == 2
    result = Evaluation.load(output)
    assert not result.metadata["complete"]
    assert result.metadata["input"]["sha256"]
    assert result.metadata["runtime"]["package_sources"]["models.py"]
    assert main(["verify", str(output)]) == 2


def test_cli_frozen_guard_binds_samples_and_primary_artifacts(tmp_path, paired):
    ids = np.arange(120)
    expanded = {name: np.repeat(getattr(paired, name)[:1], len(ids), axis=0)
                for name in ("corrupted", "ideal", "pn", "valid", "weights")}
    sample = replace(paired, **expanded, window_ids=ids,
        time_jd=2450000. + (ids + .5) * 270. / 86400., lst_rad=(6.1 + ids * .02) % (2 * np.pi))
    source, primary, sensitivity = (tmp_path / name for name in ("samples.npz", "primary.npz", "guard.npz"))
    sample.save(source)
    config = tmp_path / "config.json"
    values = {"max_rank": 0, "include_kernel": False}
    config.write_text(json.dumps(values))
    arguments = ["evaluate", "--samples", str(source), "--config", str(config)]
    assert main(arguments + ["--output", str(primary)]) == 0
    config.write_text(json.dumps({**values, "guard": 8}))
    assert main(arguments + ["--selection-from", str(primary), "--output", str(sensitivity)]) == 0
    result = Evaluation.load(sensitivity)
    assert result.metadata["selection_input"]["sha256"]
    assert not result.metadata["primary_selection"]["hyperparameter_selection_repeated"]
    replay = tmp_path / "replay.json"
    assert main(["replay", "--samples", str(source), "--evaluation", str(sensitivity), "--output", str(replay)]) == 0
    assert json.loads(replay.read_text())["passed"]
    config.write_text(json.dumps({**values, "guard": 8, "region": "horizon"}))
    with pytest.raises(SystemExit) as error:
        main(arguments + ["--selection-from", str(primary), "--output", str(tmp_path / "bad.npz")])
    assert error.value.code == 2
    assert not (tmp_path / "bad.npz").exists()
