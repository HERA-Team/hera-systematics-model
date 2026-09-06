from dataclasses import fields

import numpy as np
import pytest

from hera_systematics_model.artifacts import write_artifact
from hera_systematics_model.models import LinearModel, fit_complete, measured_arrays
from hera_systematics_model.evaluation import evaluate_nested
from test_models import low_rank_data


def test_complex_cross_power_cannot_be_silently_cast_to_real():
    arrays = list(low_rank_data())
    arrays[0] = arrays[0].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        measured_arrays(*arrays)


@pytest.mark.parametrize("field_name", ["feature_mask", "components", "singular_values", "training_ids"])
def test_hash_valid_but_inconsistent_model_is_rejected(tmp_path, field_name):
    model = fit_complete(*low_rank_data(), 2)
    arrays = {f.name: getattr(model, f.name) for f in fields(model) if f.name != "metadata"}
    arrays[field_name] = arrays[field_name][:-1]
    path = tmp_path / "model.npz"
    write_artifact(path, "fitted-model", arrays, model.metadata)
    if field_name == "training_ids":
        # Repeated identifiers are invalid regardless of the stored array hash.
        arrays[field_name][:] = 0
        path = tmp_path / "duplicate.npz"
        write_artifact(path, "fitted-model", arrays, model.metadata)
    with pytest.raises(ValueError):
        LinearModel.load(path)


def test_no_common_training_support_records_candidate_failure():
    arrays = list(low_rank_data())
    arrays[3][:] = False
    result = evaluate_nested(arrays, np.arange(40), (1, 30),
        [{"method": "mean", "rank": 0, "representation": "linear"}], guard=1)
    assert not result.metadata["complete"]
    assert all(f["status"] == "candidate_failure" for f in result.metadata["folds"])
