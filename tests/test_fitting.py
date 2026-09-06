import numpy as np

from hera_systematics_model.evaluation import Evaluation
from hera_systematics_model.fitting import select_final_fit
from hera_systematics_model.prediction import candidate_grid
from test_evaluation import series


def test_descriptive_fit_has_distinct_statistics_and_round_trip(tmp_path):
    arrays = series()
    result = select_final_fit(arrays, np.arange(100), (1, 30),
                              candidate_grid(3, False, ["linear"], ["complete"]), guard=3)
    assert result.metadata["complete"]
    assert result.metadata["selected"]["rank"] == 2
    assert result.metadata["purpose"] == "descriptive_fit"
    assert not result.metadata["performance_estimator"]
    assert not result.metadata["rank_ceiling_selected"]
    assert "window_loss" not in result.arrays
    assert result.metadata["training_loss"] < 1e-20
    result.save(tmp_path / "fit.npz")
    restored = Evaluation.load(tmp_path / "fit.npz")
    model = restored.models["descriptive"][0]
    np.testing.assert_allclose(model.predict(*arrays, np.ones(30, bool))[0], result.arrays["training_prediction"])


def test_configured_rank_ceiling_is_declared():
    result = select_final_fit(series(), np.arange(100), (1, 30),
                              candidate_grid(2, False, ["linear"], ["complete"]), guard=3)
    assert result.metadata["rank_ceiling_selected"]


def test_final_insufficient_support_cannot_be_reported_as_fit():
    arrays = tuple(a[:10] for a in series())
    result = select_final_fit(arrays, np.arange(10), (1, 30), guard=12)
    assert not result.metadata["complete"]
    assert not result.models
