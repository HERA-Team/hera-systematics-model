from copy import deepcopy

import numpy as np
import pytest

from hera_systematics_model.rebinning import common_delay_bins, overlap_weights, rebin_profiles, rebinned_mode_similarity
from test_samples import paired


def test_conservative_rebin_preserves_integrated_profile_and_requires_full_support():
    source, target = np.arange(5.), np.array([0., 2., 4.])
    weights = overlap_weights(source, target)
    profile = np.array([[[1., -1., 2., 4.]]])
    actual, support = rebin_profiles(profile, np.ones((1, 4), bool), weights)
    np.testing.assert_allclose((actual * np.diff(target)).sum(), (profile * np.diff(source)).sum())
    energy, _ = rebin_profiles(profile ** 2, np.ones((1, 4), bool), weights)
    assert actual[0, 0, 0] == 0 and energy[0, 0, 0] == 1
    mask = np.array([[False, True, True, True]])
    actual, support = rebin_profiles(profile, mask, weights)
    assert np.isnan(actual[0, 0, 0]) and not support[0, 0] and support[0, 1]
    with pytest.raises(ValueError, match="outside source"):
        overlap_weights(source, [-1., 1.])


def test_different_delay_grids_have_explicit_common_coordinates(paired):
    other = deepcopy(paired)
    other.delay_s = np.array([1.1, 2.2, 3.3, 4.4]) * 1e-7
    component = np.ones((1, 8))
    arrays, metadata = rebinned_mode_similarity(paired, other, component, -component,
                                                np.ones(8, bool), np.ones(8, bool))
    np.testing.assert_allclose(arrays["signed_similarity"], -1.)
    np.testing.assert_allclose(arrays["squared_loading_similarity"], 1.)
    assert metadata["matched_cells"] == 6
    assert metadata["bin_width_s"] >= 1.1e-7 - 1e-20
    assert np.all(arrays["common_edges_s"] >= max(arrays["left_source_edges_s"][0], arrays["right_source_edges_s"][0]))
    mask = np.ones(8, bool)
    mask[1] = False
    _, incomplete = rebinned_mode_similarity(paired, other, component, component, mask, np.ones(8, bool))
    assert incomplete["matched_cells"] < metadata["matched_cells"]
    assert incomplete["common_grid_cells"] == metadata["common_grid_cells"]


def test_common_grid_never_extrapolates_or_creates_finer_resolution():
    a, b, edges, wa, wb = common_delay_bins([1., 2., 3., 4.], [1.2, 2.4, 3.6])
    assert np.min(np.diff(edges)) >= max(np.max(np.diff(a)), np.max(np.diff(b))) - 1e-14
    np.testing.assert_allclose(wa.sum(axis=1), 1.)
    np.testing.assert_allclose(wb.sum(axis=1), 1.)
    with pytest.raises(ValueError, match="no complete common"):
        common_delay_bins([1., 2.], [5., 6.])
