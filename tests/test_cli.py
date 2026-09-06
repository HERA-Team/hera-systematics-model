import json

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
