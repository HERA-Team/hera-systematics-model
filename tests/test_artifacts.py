import json

import numpy as np
import pytest

from hera_systematics_model.artifacts import read_artifact, write_artifact


def test_roundtrip_and_exclusive_creation(tmp_path):
    path = tmp_path / "sample.npz"
    arrays = {"power": np.array([0., -3., np.nan]), "ids": np.array(["a", "b"])}
    metadata = {"unavailable": {"value": None, "reason": "no_support"}}
    write_artifact(path, "paired-samples", arrays, metadata)
    actual, info = read_artifact(path, "paired-samples")
    for key in arrays:
        np.testing.assert_equal(actual[key], arrays[key])
    assert info == metadata
    with pytest.raises(FileExistsError):
        write_artifact(path, "paired-samples", arrays, metadata)


@pytest.mark.parametrize("change", ["bytes", "version", "shape", "kind"])
def test_corruption_rejected(tmp_path, change):
    path = tmp_path / "sample.npz"
    write_artifact(path, "evaluation", {"x": np.ones(3)}, {})
    sidecar = path.with_suffix(".json")
    info = json.loads(sidecar.read_text())
    if change == "bytes":
        with path.open("ab") as stream:
            stream.write(b"extra")
    elif change == "version":
        info["schema_version"] = 200
    elif change == "shape":
        info["arrays"]["x"]["shape"] = [4]
    else:
        info["kind"] = "paired-samples"
    sidecar.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        read_artifact(path, "evaluation")


def test_invalid_metadata_and_object_arrays_leave_no_file(tmp_path):
    path = tmp_path / "sample.npz"
    with pytest.raises(ValueError):
        write_artifact(path, "evaluation", {"x": np.ones(1)}, {"loss": float("nan")})
    with pytest.raises(ValueError):
        write_artifact(path, "evaluation", {"x": np.array([{}], dtype=object)}, {})
    assert not path.exists()
