import numpy as np
import pytest

from hera_systematics_model.kernel import KernelModel, fit_kernel
from hera_systematics_model.scoring import CandidateFailure
from test_models import low_rank_data


def test_kernel_decoder_ignores_held_targets_and_survives_roundtrip(tmp_path):
    power, ideal, pn, valid = low_rank_data()
    predictor = np.arange(30) < 15
    model = fit_kernel(power[:25], ideal[:25], pn[:25], valid[:25], 2, predictor)
    prediction, scores = model.predict(power[25:], ideal[25:], pn[25:], valid[25:], predictor)
    changed = power[25:].copy()
    changed[:, ~predictor] += 1e8
    other, other_scores = model.predict(changed, ideal[25:], pn[25:], valid[25:], predictor)
    np.testing.assert_allclose(other, prediction)
    np.testing.assert_allclose(other_scores, scores)
    model.save(tmp_path / "kernel.npz")
    restored = KernelModel.load(tmp_path / "kernel.npz")
    np.testing.assert_allclose(restored.predict(changed, ideal[25:], pn[25:], valid[25:], predictor)[0], prediction)


def test_kernel_cannot_silently_impute_required_predictors():
    power, ideal, pn, valid = low_rank_data()
    predictor = np.arange(30) < 15
    model = fit_kernel(power[:25], ideal[:25], pn[:25], valid[:25], 2, predictor)
    valid[25, 0] = False
    with pytest.raises(CandidateFailure, match="incomplete"):
        model.predict(power[25:], ideal[25:], pn[25:], valid[25:], predictor)
