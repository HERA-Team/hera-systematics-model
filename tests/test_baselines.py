from copy import deepcopy

import numpy as np
import pytest

from hera_systematics_model.baselines import baseline_inventory, cross_window_similarity, match_coordinates, mode_localization
from test_samples import paired


def test_localization_reports_actual_baseline_membership(paired):
    component = np.zeros((1, 8))
    component[0, 5] = 1
    result = mode_localization(paired, component, "linear")[0]
    assert result["groups"][0]["group_id"] == "long"
    assert result["groups"][0]["baseline_ids"] == paired.baseline_ids[paired.baseline_group == 1].tolist()
    assert result["effective_groups"] == 1
    inventory = baseline_inventory(paired)
    assert sum(len(g["baselines"]) for g in inventory) == len(paired.baseline_ids)


def test_cross_window_signed_and_squared_similarities_are_distinct(paired):
    component = np.arange(1., 9.)[None, :]
    mask = np.ones(8, bool)
    result = cross_window_similarity(paired, paired, component, -component, mask, mask)
    np.testing.assert_allclose(result["signed_similarity"], -1)
    np.testing.assert_allclose(result["squared_loading_similarity"], 1)
    assert result["matched_cells"] == 8
    changed = deepcopy(paired)
    changed.baseline_length_m *= 2
    with pytest.raises(ValueError, match="geometry"):
        match_coordinates(paired, changed)


def test_unmatched_delays_are_counted_in_coverage(paired):
    right = deepcopy(paired)
    right.delay_s[0] *= .5
    component = np.ones((1, 8))
    result = cross_window_similarity(paired, right, component, component, np.ones(8, bool), np.ones(8, bool))
    assert result["matched_cells"] == 6
    assert result["left_cells"] == result["right_cells"] == 8
