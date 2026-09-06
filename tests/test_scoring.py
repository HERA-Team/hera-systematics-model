import numpy as np
import pytest

from hera_systematics_model.scoring import CandidateFailure, choose_simplest, score_predictions, training_mean


def test_common_weighted_loss_and_empty_rows():
    truth = np.zeros((2, 3))
    pred = np.array([[1., 3., np.nan], [np.nan, np.nan, np.nan]])
    noise = np.array([[1., 2., 0.], [0., 0., 0.]])
    mask = np.array([[True, True, False], [False, False, False]])
    score = score_predictions(pred, truth, noise, mask)
    assert score.mean == pytest.approx((1 + 9 / 4) / (1 + 1 / 4))
    np.testing.assert_array_equal(score.counts, [2, 0])
    assert np.isnan(score.per_window[1])
    pred[0, 1] = np.nan
    with pytest.raises(CandidateFailure):
        score_predictions(pred, truth, noise, mask)


def test_missing_values_do_not_change_training_mean():
    data = np.array([[0., 10., np.nan], [2., 9999., np.nan]])
    valid = np.array([[True, True, False], [True, False, False]])
    np.testing.assert_equal(training_mean(data, valid), [1., 10., np.nan])


def test_failed_candidates_cannot_win_by_omitting_a_fold():
    candidates = [{"rank": k, "method": "complete", "representation": "linear"} for k in [0, 1, 2]]
    losses = [[3., 3., 3.], [2., 3., 4.], [0., np.nan, 0.]]
    selected, info = choose_simplest(candidates, losses)
    assert selected == 0
    assert info["best_index"] == 0


def test_one_standard_error_prefers_fewer_components():
    candidates = [{"rank": k, "method": "complete", "representation": "linear"} for k in [0, 1, 2]]
    selected, info = choose_simplest(candidates, [[2., 2., 2.], [1.1, 1.1, 1.1], [.6, 1., 1.4]])
    assert selected == 1
    assert info["best_index"] == 2
