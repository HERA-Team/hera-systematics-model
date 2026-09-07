from dataclasses import replace

import numpy as np
import pytest

from hera_systematics_model.artifacts import read_artifact, write_artifact
from hera_systematics_model.cross_spw import cross_spw_diagnostics
from hera_systematics_model.fitting import select_final_fit
from hera_systematics_model.views import analysis_view
from test_samples import paired


def sample_and_fit(paired, spw, spacing=1., rank=1):
    ids = np.arange(100)
    rng = np.random.default_rng(72)
    power = (rng.normal(size=(100, 1)) @ rng.normal(size=(1, 30))).reshape(100, 2, 15)
    samples = replace(paired, corrupted=power, ideal=np.zeros_like(power), pn=np.ones_like(power),
        valid=np.ones(power.shape, bool), weights=np.full((100, 2, 15, 2), .5),
        window_ids=ids, time_jd=2450000. + (ids + .5) * 270. / 86400.,
        lst_rad=(6.1 + ids * .02) % (2 * np.pi), delay_s=np.arange(1, 16) * spacing * 1e-7,
        kparallel=np.arange(1, 16) * .05, metadata={**paired.metadata, "spw": spw})
    arrays, shape, identity = analysis_view(samples)
    fit = select_final_fit(arrays, ids, shape,
        [{"method": "complete", "representation": "linear", "rank": rank}], guard=3)
    assert fit.metadata["complete"]
    fit.metadata["identity"] = identity
    return samples, fit


def test_exact_and_rebinned_comparisons_save_separate_signed_and_squared_arrays(paired, tmp_path):
    left, lf = sample_and_fit(paired, 0)
    right, rf = sample_and_fit(paired, 6)
    arrays, metadata = cross_spw_diagnostics(left, right, lf, rf)
    assert metadata["spw_0_6_comparison"]
    assert metadata["mode_similarity"]["exact"]["matched_cells"] == 30
    for method in ("exact", "rebinned"):
        np.testing.assert_allclose(arrays[method + "_signed_similarity"], 1.)
        np.testing.assert_allclose(arrays[method + "_squared_loading_similarity"], 1.)
    assert metadata["left"]["physical_mode_energy"]["kparallel_threshold"] == .3
    path = tmp_path / "comparison.npz"
    write_artifact(path, "diagnostics", arrays, metadata)
    restored, meta = read_artifact(path, "diagnostics")
    assert meta["complete"]
    np.testing.assert_equal(restored["left_mode_high_k_window_energy"], arrays["left_mode_high_k_window_energy"])


def test_different_delay_grids_keep_exact_absence_and_common_bin_evidence(paired):
    left, lf = sample_and_fit(paired, 0)
    right, rf = sample_and_fit(paired, 6, spacing=1.123)
    arrays, metadata = cross_spw_diagnostics(left, right, lf, rf)
    assert not metadata["mode_similarity"]["exact"]["available"]
    assert metadata["mode_similarity"]["rebinned"]["available"]
    assert arrays["rebinned_common_edges_s"].size > 2
    assert "no common spectral window-function calculation" in metadata["mode_similarity"]["rebinned"]["limitations"]


def test_zero_rank_is_not_fabricated_as_a_saved_component(paired):
    left, lf = sample_and_fit(paired, 0, rank=0)
    right, rf = sample_and_fit(paired, 6)
    arrays, metadata = cross_spw_diagnostics(left, right, lf, rf)
    assert metadata["complete"] and not metadata["mode_similarity"]["available"]
    assert "left_components" not in arrays
    assert arrays["left_mode_high_k_window_energy"].shape == (0, 100)


def test_cross_comparison_rejects_mismatched_saved_coordinates(paired):
    left, lf = sample_and_fit(paired, 0)
    right, rf = sample_and_fit(paired, 6)
    right.delay_s *= 1.01
    with pytest.raises(ValueError, match="match a complete"):
        cross_spw_diagnostics(left, right, lf, rf)
