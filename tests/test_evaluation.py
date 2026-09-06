import numpy as np

from hera_systematics_model.artifacts import read_artifact
from hera_systematics_model.evaluation import evaluate_nested, select_within
from hera_systematics_model.prediction import candidate_grid
from hera_systematics_model.splits import feature_partitions, time_folds


def series(noise_only=False):
    rng = np.random.default_rng(717)
    power = (rng.normal(size=(100, 30)) if noise_only else
             rng.normal(size=(100, 2)) @ rng.normal(size=(2, 30)) + 3)
    return power, np.zeros_like(power), np.ones_like(power), np.ones(power.shape, bool)


def test_nested_validation_recovers_structure_and_saves_evidence(tmp_path):
    arrays = series()
    candidates = candidate_grid(3, include_kernel=False, representations=["linear"], methods=["complete"])
    result = evaluate_nested(arrays, np.arange(100), (1, 30), candidates, guard=3)
    assert result.metadata["complete"]
    assert all(fold["selected"]["rank"] == 2 for fold in result.metadata["folds"])
    assert np.max(result.arrays["window_loss"]) < 1e-20
    assert result.arrays["target"].all()
    result.save(tmp_path / "evaluation.npz")
    actual, meta = read_artifact(tmp_path / "evaluation.npz", "evaluation")
    np.testing.assert_equal(actual["prediction"], result.arrays["prediction"])
    assert len(meta["folds"]) == 4


def test_outer_targets_cannot_change_inner_selection():
    arrays = series()
    ids = np.arange(100)
    outer = time_folds(ids, guard=3)[0]
    partitions = feature_partitions((1, 30))
    candidates = candidate_grid(3, False, ["linear"], ["complete"])
    original = select_within(arrays, ids, outer.train, partitions, candidates, guard=3)
    changed = [a.copy() for a in arrays]
    changed[0][outer.test] *= 1e10
    modified = select_within(changed, ids, outer.train, partitions, candidates, guard=3)
    assert modified[0] == original[0]
    np.testing.assert_equal(modified[1], original[1])


def test_noise_only_does_not_require_extra_modes():
    candidates = candidate_grid(3, False, ["linear"], ["complete"])
    result = evaluate_nested(series(True), np.arange(100), (1, 30), candidates, guard=3)
    assert result.metadata["complete"]
    assert all(fold["selected"]["rank"] == 0 for fold in result.metadata["folds"])
    np.testing.assert_array_equal(result.arrays["modeled"] | result.arrays["mean_only"]
                                  | result.arrays["zero_only"], result.arrays["target"])


def test_insufficient_time_support_is_not_reported_as_complete():
    arrays = tuple(a[:10] for a in series())
    result = evaluate_nested(arrays, np.arange(10), (1, 30), guard=12)
    assert not result.metadata["complete"]
    assert all(fold["status"] == "insufficient_support" for fold in result.metadata["folds"])
