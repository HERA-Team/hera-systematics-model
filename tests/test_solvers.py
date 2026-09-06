import json

import numpy as np
import pytest

from hera_systematics_model.scoring import CandidateFailure
from hera_systematics_model.solvers import solve_observed
from hera_systematics_model.masked import fit_masked
from hera_systematics_model.models import fit_complete
from test_models import low_rank_data


def test_rank_failure_retains_conditioning_without_nonfinite_json():
    with pytest.raises(CandidateFailure) as caught:
        solve_observed(np.ones((5, 2)), np.arange(5.))
    evidence = caught.value.diagnostics
    assert evidence["effective_rank"] == 1
    assert evidence["condition_number"] is None
    json.dumps(evidence, allow_nan=False)


def test_predictor_diagnostics_and_scores_ignore_withheld_values():
    arrays = low_rank_data()
    model = fit_complete(*(a[:25] for a in arrays), 2)
    predictor = np.arange(30) < 15
    before, after = [], []
    model.predict(*(a[25:] for a in arrays), predictor, diagnostics=before)
    changed = arrays[0][25:].copy()
    changed[:, ~predictor] = 1e25
    model.predict(changed, *(a[25:] for a in arrays[1:]), predictor, diagnostics=after)
    assert before == after
    assert all(row["effective_rank"] == 2 and row["condition_number"] >= 1 for row in before)
    assert all(row["predictor_features"] == list(range(15)) for row in before)


def test_masked_fit_retains_convergence_path_and_conditioning():
    model = fit_masked(*low_rank_data(), 2)
    evidence = model.metadata
    assert len(evidence["observed_loss_history"]) == evidence["iterations"] + 1
    assert evidence["observed_loss_history"][-1] == evidence["observed_loss"]
    assert evidence["conditioning"]["row_max_condition"] >= 1
    assert evidence["conditioning"]["feature_max_condition"] >= 1
    json.dumps(evidence, allow_nan=False)
