import numpy as np
import pytest

from hera_systematics_model.masked import fit_masked
from hera_systematics_model.models import fit_complete
from hera_systematics_model.scoring import CandidateFailure
from test_models import low_rank_data


def test_fully_observed_subspace_matches_complete_pca():
    power, ideal, pn, valid = low_rank_data()
    complete = fit_complete(power[:25], ideal[:25], pn[:25], valid[:25], 2)
    masked = fit_masked(power[:25], ideal[:25], pn[:25], valid[:25], 2)
    assert masked.metadata["converged"]
    np.testing.assert_allclose(masked.components.T @ masked.components,
                               complete.components[:2].T @ complete.components[:2], atol=1e-10)


def test_partial_observations_recover_structure_without_zero_bias():
    power, ideal, pn, valid = low_rank_data()
    rng = np.random.default_rng(21)
    valid[:25] = rng.uniform(size=(25, 30)) > .15
    damaged = power.copy()
    damaged[~valid] = -1e100
    model = fit_masked(damaged[:25], ideal[:25], pn[:25], valid[:25], 2)
    assert model.metadata["converged"]
    prediction, _ = model.predict(power[25:], ideal[25:], pn[25:], valid[25:], np.arange(30) < 15)
    np.testing.assert_allclose(prediction, (power - ideal)[25:], atol=1e-7)
    changed = damaged.copy()
    changed[~valid] = 1e100
    second = fit_masked(changed[:25], ideal[:25], pn[:25], valid[:25], 2)
    np.testing.assert_allclose(second.components, model.components)
    assert len(model.explained_variance_ratio) == 0


def test_nonconvergence_cannot_be_used_for_prediction():
    power, ideal, pn, valid = low_rank_data()
    valid[np.arange(40), np.arange(40) % 30] = False
    model = fit_masked(power, ideal, pn, valid, 2, max_iter=1)
    assert not model.metadata["converged"]
    with pytest.raises(CandidateFailure, match="converge"):
        model.predict(power, ideal, pn, valid, np.ones(30, bool))
