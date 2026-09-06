import numpy as np

from hera_systematics_model.prediction import candidate_grid, predict_partitioned
from hera_systematics_model.splits import feature_partitions
from test_models import low_rank_data


def test_structured_predictions_beat_mean_on_unseen_rows_and_cells():
    arrays = low_rank_data()
    partitions = feature_partitions((1, 30))
    train, test = np.arange(25), np.arange(25, 40)
    candidate = {"method": "complete", "rank": 2, "representation": "linear"}
    result = predict_partitioned(arrays, np.arange(40), train, test, partitions, candidate)
    mean = predict_partitioned(arrays, np.arange(40), train, test, partitions,
                               {"method": "mean", "rank": 0, "representation": "linear"})
    assert result.loss.mean < 1e-20
    assert mean.loss.mean > .1
    assert result.target.all()
    assert result.modeled.all()


def test_candidate_grid_includes_baselines_and_all_log_margins():
    grid = candidate_grid(max_rank=2, include_kernel=False)
    assert [c["method"] for c in grid[:2]] == ["zero", "mean"]
    assert {c["log_margin"] for c in grid if c["representation"] == "log_ratio"} == {1., 3., 10.}
    assert {c["method"] for c in grid[2:]} == {"complete", "masked"}
