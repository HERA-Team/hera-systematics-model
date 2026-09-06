import json

import numpy as np
import pytest

from hera_systematics_model.evaluation import Evaluation, evaluate_nested
from hera_systematics_model.model_io import ConstantModel, load_model, predict_samples
from hera_systematics_model.views import analysis_view
from test_evaluation import series
from test_samples import paired


@pytest.mark.parametrize("method", ["zero", "mean"])
def test_constant_prediction_round_trip(tmp_path, method):
    arrays = series()
    model = ConstantModel.fit(arrays, np.arange(100), method)
    model.save(tmp_path / "model.npz")
    restored = load_model(tmp_path / "model.npz")
    mask = np.ones(30, bool)
    np.testing.assert_array_equal(model.predict(*arrays, mask)[0], restored.predict(*arrays, mask)[0])


def test_evaluation_saves_all_prediction_state_and_checks_metadata_hash(tmp_path):
    candidates = [{"rank": 2, "method": "complete", "representation": "linear"}]
    result = evaluate_nested(series(), np.arange(100), (1, 30), candidates, guard=3)
    result.metadata["identity"] = {"spw": 1, "groups": ["a"]}
    path = tmp_path / "evaluation.npz"
    result.save(path)
    restored = Evaluation.load(path)
    assert len(restored.models) == 4
    for key, models in result.models.items():
        np.testing.assert_equal(models[0].components, restored.models[str(key)][0].components)
    entry = restored.metadata["models"]["0"][0]
    metadata_path = (tmp_path / entry["path"]).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["rank"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash"):
        Evaluation.load(path)


def test_model_prediction_rejects_changed_physical_features(paired):
    arrays, _, identity = analysis_view(paired)
    model = ConstantModel.fit(arrays, paired.window_ids, "mean")
    model.metadata["identity"] = identity
    predict_samples(model, paired, np.ones(8, bool))
    paired.delay_s = paired.delay_s * 2
    with pytest.raises(ValueError, match="identities"):
        predict_samples(model, paired, np.ones(8, bool))
