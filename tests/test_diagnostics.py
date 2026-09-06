import numpy as np

from hera_systematics_model.diagnostics import regional_losses


def test_regional_losses_keep_geometry_and_unavailable_denominators():
    truth = np.ones((4, 6))
    target = np.ones(truth.shape, bool)
    target[:, :2] = False
    prediction = np.zeros(truth.shape)
    prediction[~target] = np.nan
    regions = {"full": np.ones((2, 3), bool), "short": np.array([[True] * 3, [False] * 3]),
               "empty": np.zeros((2, 3), bool)}
    result = regional_losses(prediction, truth, np.ones_like(truth), target, np.ones_like(target), regions)
    assert result["full"]["scored_cells"] == 16
    assert result["full"]["eligible_cells"] == 24
    assert result["short"]["scored_cells"] == 4
    assert result["short"]["eligible_cells"] == 12
    assert result["full"]["window_loss"]["mean"] == 1
    assert result["empty"]["window_loss"]["mean"] is None
    assert result["empty"]["window_loss"]["reason"]
