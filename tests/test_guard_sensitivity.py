import numpy as np
import pytest

import hera_systematics_model.evaluation as evaluation
from hera_systematics_model.prediction import candidate_grid
from test_evaluation import series


def primary_evaluation():
    arrays = series()
    ids = np.arange(100)
    ids[50:] += 4
    result = evaluation.evaluate_nested(arrays, ids, (1, 30),
        candidate_grid(2, False, ["linear"], ["complete"]), guard=12)
    assert result.metadata["complete"]
    return arrays, ids, result


@pytest.mark.parametrize("guard", [8, 16])
def test_guards_refit_frozen_choices_without_reselection(guard, monkeypatch, tmp_path):
    arrays, ids, primary = primary_evaluation()

    def forbidden(*args, **kwargs):
        raise AssertionError("selection must remain frozen")

    monkeypatch.setattr(evaluation, "select_within", forbidden)
    result = evaluation.evaluate_guard_sensitivity(primary, arrays, ids, guard)
    assert result.metadata["complete"]
    assert result.metadata["selection_mode"] == "frozen_outer_choices"
    assert not any(name.startswith("inner_losses_") for name in result.arrays)
    np.testing.assert_equal(result.arrays["eligible"], primary.arrays["eligible"])
    for before, after in zip(primary.metadata["folds"], result.metadata["folds"]):
        assert before["selected"] == after["selected"]
        assert before["test_window_ids"] == after["test_window_ids"]
        assert before["train_window_ids"] != after["train_window_ids"]
        train, test = np.array(after["train_window_ids"]), np.array(after["test_window_ids"])
        assert np.min(np.abs(train[:, None] - test)) > guard
    path = tmp_path / "guard.npz"
    result.save(path)
    loaded = evaluation.Evaluation.load(path)
    assert loaded.metadata["primary_selection"] == result.metadata["primary_selection"]


def test_guard_sensitivity_rejects_changed_identities_and_support():
    arrays, ids, primary = primary_evaluation()
    with pytest.raises(ValueError, match="physical windows"):
        evaluation.evaluate_guard_sensitivity(primary, arrays, ids + 1, 8)
    changed = [a.copy() for a in arrays]
    changed[3][0, 0] = False
    with pytest.raises(ValueError, match="eligible support"):
        evaluation.evaluate_guard_sensitivity(primary, changed, ids, 8)
    with pytest.raises(ValueError, match="guard 8 or 16"):
        evaluation.evaluate_guard_sensitivity(primary, arrays, ids, 12)


def test_guard_targets_cannot_change_frozen_model_or_predictor_coefficients():
    arrays, ids, primary = primary_evaluation()
    result = evaluation.evaluate_guard_sensitivity(primary, arrays, ids, 16)
    modified = [a.copy() for a in arrays]
    rows = result.arrays["outer_fold"] == 0
    targets = result.arrays["feature_targets"][0]
    modified[0][np.ix_(rows, targets)] += 1e12
    changed = evaluation.evaluate_guard_sensitivity(primary, modified, ids, 16)
    assert changed.metadata["folds"][0]["selected"] == result.metadata["folds"][0]["selected"]
    for left, right in zip(result.models[0], changed.models[0]):
        np.testing.assert_equal(left.components, right.components)
    for entry in result.metadata["folds"][0]["inference"]:
        if entry["partition_index"] == 0:
            key = entry["array_prefix"] + "_scores"
            np.testing.assert_equal(result.arrays[key], changed.arrays[key])
    np.testing.assert_equal(result.arrays["prediction"][np.ix_(rows, targets)],
                            changed.arrays["prediction"][np.ix_(rows, targets)])
