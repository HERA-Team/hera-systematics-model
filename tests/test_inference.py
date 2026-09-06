import numpy as np
import pytest

from hera_systematics_model.prediction import predict_partitioned
from hera_systematics_model.splits import feature_partitions
from test_models import low_rank_data


@pytest.mark.parametrize("method", ["complete", "masked", "kernel", "mean", "zero"])
def test_retained_inference_uses_only_predictors_and_reconstructs_targets(method):
    arrays = low_rank_data()
    windows = np.arange(40)
    partitions = feature_partitions((1, 30), guard=2)
    candidate = {"method": method, "rank": 0 if method in ("zero", "mean") else 2, "representation": "linear"}
    result = predict_partitioned(arrays, windows, np.arange(25), np.arange(25, 40), partitions,
                                 candidate, keep_models=True)
    assert len(result.inference) == len(partitions)
    for entry, partition in zip(result.inference, partitions):
        assert not entry["predictor_support"][:, ~partition.predictor].any()
        model = result.models[entry["model_index"]]
        prediction, scores = model.predict(*(a[25:] for a in arrays), partition.predictor)
        np.testing.assert_allclose(scores, entry["scores"])
        np.testing.assert_allclose(prediction[:, partition.target], result.prediction[:, partition.target])
        if candidate["rank"]:
            assert np.isfinite(entry["condition_number"]).all()
            assert np.all(entry["effective_rank"] == 2)
        else:
            assert not entry["predictor_support"].any()
            assert entry["condition_unavailable_reason"] is not None
