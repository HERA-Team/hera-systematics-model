import numpy as np
import pytest

from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.configuration import AnalysisConfig
from hera_systematics_model.evaluation import evaluate_nested
from hera_systematics_model.evidence import evaluation_evidence, fold_summary
from hera_systematics_model.fitting import select_final_fit


def products(n=123, zero=False):
    rows = np.arange(n)
    power = np.zeros((n, 30)) if zero else np.broadcast_to(np.sin(rows[:, None] / 7), (n, 30)).copy()
    arrays = power, np.zeros_like(power), np.ones_like(power), np.ones(power.shape, bool)
    config = AnalysisConfig(max_rank=0, include_kernel=False)
    evaluation = evaluate_nested(arrays, rows, (2, 15), config.candidates())
    fit = select_final_fit(arrays, rows, (2, 15), config.candidates())
    for artifact in (evaluation, fit):
        artifact.metadata.update(identity={"spw": 0, "power_units": "mK2 Mpc3 / h3"},
            configuration=config.as_dict(), input={"sha256": "a"}, input_metadata={"sha256": "b"})
    return evaluation, fit


def test_evidence_separates_descriptive_choice_and_equal_time_fold_means():
    evaluation, fit = products()
    report = evaluation_evidence(evaluation, fit)
    means = [evaluation.arrays["window_loss"][evaluation.arrays["outer_fold"] == i].mean() for i in range(4)]
    np.testing.assert_allclose(report["predictive_loss"]["selected"]["mean"], np.mean(means))
    np.testing.assert_allclose(report["predictive_loss"]["selected"]["standard_error"], np.std(means, ddof=1) / 2)
    assert report["descriptive_fit"]["selected"] == fit.metadata["selected"]
    assert not report["descriptive_fit"]["performance_estimator"]
    assert report["coverage"]["eligible_cells"] == 123 * 30
    assert report["coverage"]["target"]["fraction_of_eligible"] == 1
    assert len(report["folds"]) == 4
    assert report["baseline_comparisons"]["mean"]["available_paired_folds"] == 4
    canonical_json(report)


def test_incomplete_folds_never_get_a_successful_subset_average():
    report = fold_summary([1., np.nan, 3., 4.], 4)
    assert report["available_folds"] == 3 and not report["complete"]
    assert report["mean"] is None and report["standard_error"] is None
    evaluation, fit = products(n=20)
    report = evaluation_evidence(evaluation, fit)
    assert not report["evaluation_complete"]
    assert report["predictive_loss"]["selected"]["mean"] is None
    assert report["coverage"]["unavailable"]["fraction_of_eligible"] == 1
    canonical_json(report)


def test_zero_baseline_ratio_is_explicitly_undefined():
    report = evaluation_evidence(*products(zero=True))
    for comparison in report["baseline_comparisons"].values():
        assert comparison["selected_to_baseline_loss_ratio"] is None
        assert comparison["ratio_unavailable_reason"]
        assert comparison["baseline_minus_selected"]["mean"] == 0
    canonical_json(report)


def test_evidence_rejects_different_sample_or_region_definitions():
    evaluation, fit = products()
    fit.metadata["input"] = {"sha256": "different"}
    with pytest.raises(ValueError, match="identities disagree"):
        evaluation_evidence(evaluation, fit)
    fit.metadata["input"] = evaluation.metadata["input"]
    fit.metadata["configuration"]["region"] = "horizon"
    with pytest.raises(ValueError, match="configurations disagree"):
        evaluation_evidence(evaluation, fit)
