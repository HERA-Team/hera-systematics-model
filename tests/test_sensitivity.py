import numpy as np
import pytest

from hera_systematics_model.evaluation import evaluate_nested, select_within
from hera_systematics_model.prediction import candidate_grid
from hera_systematics_model.sensitivity import GroupExclusion
from hera_systematics_model.splits import feature_partitions, time_folds
from test_evaluation import series


@pytest.mark.parametrize("metric", ["leading_linear_loading", "noise_weighted_energy"])
def test_exclusion_and_inner_selection_ignore_outer_values(metric):
    arrays = series()
    ids = np.arange(100)
    outer = time_folds(ids, guard=3)[0]
    exclusion = GroupExclusion((5, 6), tuple("abcde"), 1, metric)
    original = exclusion(arrays, outer.train)[1]
    changed = [a.copy() for a in arrays]
    changed[0][outer.test, :6] += 1e10
    assert exclusion(changed, outer.train)[1] == original
    partitions = feature_partitions((5, 6))
    candidates = candidate_grid(2, False, ["linear"], ["complete"])
    left = select_within(arrays, ids, outer.train, partitions, candidates, guard=3, training_filter=exclusion)
    right = select_within(changed, ids, outer.train, partitions, candidates, guard=3, training_filter=exclusion)
    assert left[0] == right[0]
    assert left[2] == right[2]
    np.testing.assert_equal(left[1], right[1])


def test_sensitivity_keeps_original_coverage_denominators():
    result = evaluate_nested(series(), np.arange(100), (5, 6),
        candidate_grid(2, False, ["linear"], ["complete"]), guard=3,
        training_filter=GroupExclusion((5, 6), tuple("abcde"), 1))
    assert result.metadata["complete"]
    arrays = result.arrays
    assert arrays["excluded"].sum() == 600
    assert arrays["eligible"].sum() == 3000
    np.testing.assert_array_equal(arrays["target"] | arrays["unavailable"] | arrays["excluded"], arrays["eligible"])
    assert not (arrays["target"] & arrays["excluded"]).any()
    for report in result.metadata["folds"]:
        assert report["eligible_cells"] == report["scored_cells"] + report["excluded_cells"]
        assert len(report["inner"]["training_filters"]) == 3
