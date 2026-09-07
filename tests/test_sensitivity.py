import numpy as np
import pytest

from hera_systematics_model.evaluation import evaluate_nested, select_within
from hera_systematics_model.prediction import candidate_grid
from hera_systematics_model.sensitivity import GroupExclusion
from hera_systematics_model.splits import feature_partitions, time_folds
from hera_systematics_model.fitting import select_final_fit
from test_evaluation import series
from test_samples import paired
from hera_systematics_model.configuration import AnalysisConfig
from hera_systematics_model.sensitivity import CombinedFilters, GeometricRegion
from hera_systematics_model.views import analysis_view, geometry_masks


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


def test_descriptive_exclusion_is_selected_by_inner_training_partitions():
    exclusion = GroupExclusion((5, 6), tuple("abcde"), 1)
    result = select_final_fit(series(), np.arange(100), (5, 6),
        candidate_grid(2, False, ["linear"], ["complete"]), guard=3, training_filter=exclusion)
    assert result.metadata["complete"]
    assert len(result.metadata["inner"]["training_filters"]) == 3
    assert result.metadata["training_filter"]["training_rows"] == list(range(100))
    np.testing.assert_array_equal(result.arrays["target"] | result.arrays["excluded"] | result.arrays["unavailable"],
                                  result.arrays["eligible"])


def test_region_preserves_coverage_and_ignores_excluded_power(tmp_path):
    arrays = series()
    mask = np.tile([False, False, True, True, True, True], 5)
    region = GeometricRegion(mask, "horizon")
    candidates = candidate_grid(2, False, ["linear"], ["complete"])
    left = evaluate_nested(arrays, np.arange(100), (5, 6), candidates,
                           guard=3, training_filter=region)
    changed = [a.copy() for a in arrays]
    changed[0][:, ~mask] = 1e50
    right = evaluate_nested(changed, np.arange(100), (5, 6), candidates,
                            guard=3, training_filter=region)
    assert left.metadata["complete"] and right.metadata["complete"]
    assert left.arrays["eligible"].sum() == 3000
    assert left.arrays["excluded"].sum() == 1000
    for name in ("prediction", "window_loss", "modeled", "mean_only"):
        np.testing.assert_equal(left.arrays[name], right.arrays[name])
    np.testing.assert_equal(left.arrays["target"] | left.arrays["excluded"] | left.arrays["unavailable"],
                            left.arrays["eligible"])
    left.save(tmp_path / "regional.npz")
    restored = type(left).load(tmp_path / "regional.npz")
    np.testing.assert_equal(restored.arrays["excluded"], left.arrays["excluded"])


@pytest.mark.parametrize("slice_config", [{}, {"group": 1}, {"delay": 2}])
def test_config_region_uses_the_same_physical_slice(paired, slice_config):
    paired.baseline_length_m[:] = [40., 90.]
    config = AnalysisConfig(region="horizon", **slice_config)
    arrays, shape, _ = analysis_view(paired, **slice_config)
    filtered, report = config.training_filter(paired)(arrays, [0, 1])
    mask = geometry_masks(paired, paired.baseline_length_m / 299792458.)["horizon"]
    if "group" in slice_config:
        mask = mask[1:2]
    if "delay" in slice_config:
        mask = mask[:, 2:3]
    np.testing.assert_equal(filtered[3], arrays[3] & mask.ravel())
    assert report["retained_features"] == int(mask.sum())
    assert report["original_features"] == np.prod(shape)


def test_region_then_group_exclusion_uses_only_retained_training_cells():
    arrays = series()
    mask = np.tile([False, False, True, True, True, True], 5)
    filters = CombinedFilters((GeometricRegion(mask, "horizon"),
                              GroupExclusion((5, 6), tuple("abcde"), 1)))
    train = np.arange(50)
    filtered, report = filters(arrays, train)
    changed = [a.copy() for a in arrays]
    changed[0][:, ~mask] += 1e30
    changed[0][50:] += 1e20
    other, other_report = filters(changed, train)
    assert report == other_report
    np.testing.assert_equal(filtered[3], other[3])
    assert len(report["ordered_filters"]) == 2


def test_empty_region_is_explicit_insufficient_candidate_support():
    result = evaluate_nested(series(), np.arange(100), (5, 6),
        candidate_grid(0, False, ["linear"], ["complete"]), guard=3,
        training_filter=GeometricRegion(np.zeros(30, bool), "horizon_buffer"))
    assert not result.metadata["complete"]
    assert not result.arrays["target"].any()
    assert all("geometric region" in fold["inner"]["failures"][0]["reason"]
               for fold in result.metadata["folds"])
