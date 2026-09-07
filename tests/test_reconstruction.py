import numpy as np
import pytest

from hera_systematics_model.model_io import load_model
from hera_systematics_model.prediction import fit_candidate
from hera_systematics_model.reconstruction import decode_scores, individual_mode_contrasts, physical_mode_energy
from test_models import low_rank_data


@pytest.mark.parametrize("method", ["complete", "masked", "kernel"])
@pytest.mark.parametrize("representation", ["linear", "noise_weighted", "signed_asinh", "log_ratio"])
def test_saved_scores_reconstruct_predictions_without_corrupted_targets(tmp_path, method, representation):
    arrays = low_rank_data()
    arrays[2][:] = np.linspace(1., 2., 30)
    predictor = np.arange(30) < 15
    candidate = {"method": method, "representation": representation, "rank": 2}
    model = fit_candidate(arrays, np.arange(25), candidate, predictor, np.arange(40))
    prediction, scores = model.predict(*(value[25:] for value in arrays), predictor)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = load_model(path)
    decoded = decode_scores(loaded, scores, arrays[1][25:], arrays[2][25:])
    np.testing.assert_allclose(decoded, prediction, rtol=1e-12, atol=1e-12)


def test_linear_physical_mode_energy_uses_measured_noise_once_and_retains_coverage():
    arrays = low_rank_data()
    arrays[2][:] = np.linspace(1., 3., 30)
    rows, predictor = np.arange(40), np.ones(30, bool)
    model = fit_candidate(arrays, rows, {"method": "complete", "representation": "noise_weighted", "rank": 2}, predictor, rows)
    prediction, scores = model.predict(*arrays, predictor)
    contrasts = list(individual_mode_contrasts(model, scores, arrays[1], arrays[2]))
    np.testing.assert_allclose(contrasts[0], scores[:, :1] * model.components[0] * arrays[2], atol=1e-12)
    baseline = decode_scores(model, np.zeros_like(scores), arrays[1], arrays[2])
    np.testing.assert_allclose(sum(contrasts) + baseline, prediction, atol=1e-12)
    valid = arrays[3].copy()
    valid[0, -1] = False
    values, metadata = physical_mode_energy(model, scores, arrays[1], arrays[2], valid, np.arange(30) >= 15)
    assert values["high_k_counts"][0] == 14
    assert metadata["high_k_valid_cells"] == 40 * 15 - 1
    assert metadata["additive_contrasts"] and not metadata["additive_energy"]
    np.testing.assert_allclose(values["high_k_window_energy"][0, 0], np.mean(contrasts[0][0, 15:29] ** 2))


@pytest.mark.parametrize("method", ["zero", "mean"])
def test_rank_zero_has_no_fabricated_mode_energy(method):
    arrays = low_rank_data()
    model = fit_candidate(arrays, np.arange(40), {"method": method, "rank": 0}, np.ones(30, bool), np.arange(40))
    prediction, scores = model.predict(*arrays, np.ones(30, bool))
    np.testing.assert_equal(decode_scores(model, scores, arrays[1], arrays[2]), prediction)
    output, metadata = physical_mode_energy(model, scores, arrays[1], arrays[2], arrays[3], np.ones(30, bool))
    assert output["mean_squared_contrast"].shape == (0, 30)
    assert metadata["unavailable_reason"]
