import numpy as np
import pytest

from hera_systematics_model.models import LinearModel, fit_complete


def low_rank_data():
    rng = np.random.default_rng(13)
    residual = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 30)) + 2
    ideal = np.full(residual.shape, 10.)
    return residual + ideal, ideal, np.ones_like(ideal), np.ones(ideal.shape, bool)


def test_withheld_targets_cannot_change_coefficients_or_predictions(tmp_path):
    power, ideal, pn, valid = low_rank_data()
    model = fit_complete(power[:25], ideal[:25], pn[:25], valid[:25], 2)
    predictor = np.arange(30) < 15
    prediction, scores = model.predict(power[25:], ideal[25:], pn[25:], valid[25:], predictor)
    changed = power[25:].copy()
    changed[:, ~predictor] += 1e10
    other, other_scores = model.predict(changed, ideal[25:], pn[25:], valid[25:], predictor)
    np.testing.assert_allclose(other, prediction)
    np.testing.assert_allclose(other_scores, scores)
    np.testing.assert_allclose(prediction[:, ~predictor], (power - ideal)[25:, ~predictor], atol=1e-12)
    model.save(tmp_path / "model.npz")
    restored = LinearModel.load(tmp_path / "model.npz")
    np.testing.assert_allclose(restored.predict(power[25:], ideal[25:], pn[25:], valid[25:], predictor)[0], prediction)
    assert len(restored.components) == len(restored.singular_values)


def test_incomplete_features_use_raw_training_mean():
    power, ideal, pn, valid = low_rank_data()
    valid[0, -1] = False
    model = fit_complete(power[:25], ideal[:25], pn[:25], valid[:25], 2)
    assert not model.feature_mask[-1]
    prediction, _ = model.predict(power[25:], ideal[25:], pn[25:], valid[25:], np.ones(30, bool))
    np.testing.assert_allclose(prediction[:, -1], (power - ideal)[1:25, -1].mean())


@pytest.mark.parametrize("representation", ["linear", "noise_weighted", "signed_asinh", "log_ratio"])
def test_zero_rank_and_zero_variance_are_finite(representation):
    power = np.zeros((10, 8))
    model = fit_complete(power, power, np.ones_like(power), np.ones(power.shape, bool), 0, representation)
    np.testing.assert_array_equal(model.explained_variance_ratio, 0)
    prediction, scores = model.predict(power, power, np.ones_like(power), np.ones(power.shape, bool), np.ones(8, bool))
    np.testing.assert_array_equal(prediction, 0)
    assert scores.shape == (10, 0)
