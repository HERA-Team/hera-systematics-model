import numpy as np
import pytest

from hera_systematics_model.evaluation import evaluate_nested, Evaluation
from hera_systematics_model.evaluation_replay import replay_evaluation
from hera_systematics_model.sensitivity import GeometricRegion
from test_evaluation import series


@pytest.mark.parametrize("method,representation", [("complete", name) for name in
    ("linear", "noise_weighted", "signed_asinh", "log_ratio")] + [("masked", "linear"), ("kernel", "signed_asinh")])
def test_saved_models_replay_predictor_coefficients_decoders_and_losses(tmp_path, method, representation):
    arrays = series()
    candidate = {"method": method, "representation": representation, "rank": 2}
    if method == "kernel":
        candidate.update(bandwidth=1., alpha=1.)
    result = evaluate_nested(arrays, np.arange(100), (1, 30), [candidate], guard=3)
    assert result.metadata["complete"]
    path = tmp_path / "evaluation.npz"
    result.save(path)
    report = replay_evaluation(arrays, np.arange(100), Evaluation.load(path))
    assert report["passed"] and report["evaluation_complete"]
    assert all(fold["target_cells"] == 750 for fold in report["folds"])
    assert not report["basis_refitted"] and not report["selection_repeated"]


@pytest.mark.parametrize("field", ["prediction", "window_loss", "inference_0_0_scores"])
def test_replay_rejects_changed_numerical_evidence(field):
    arrays = series()
    result = evaluate_nested(arrays, np.arange(100), (1, 30),
        [{"method": "complete", "representation": "linear", "rank": 2}], guard=3)
    result.arrays[field].flat[0] += 1
    with pytest.raises(ValueError, match="replay mismatch"):
        replay_evaluation(arrays, np.arange(100), result)


def test_replay_preserves_region_exclusions_and_mean_only_predictions():
    arrays = series()
    mask = np.arange(30) > 6
    region = GeometricRegion(mask, "horizon")
    result = evaluate_nested(arrays, np.arange(100), (1, 30),
        [{"method": "mean", "representation": "linear", "rank": 0}], guard=3, training_filter=region)
    assert replay_evaluation(arrays, np.arange(100), result, training_filter=region)["passed"]
    with pytest.raises(ValueError, match="support or exclusions"):
        replay_evaluation(arrays, np.arange(100), result)
