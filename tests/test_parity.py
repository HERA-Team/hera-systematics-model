import numpy as np
import h5py

from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.parity import compare_arrays, compare_spectra, exact_equal


def test_parity_requires_identical_support_and_measures_complex_error():
    reference = np.array([1 + 2j, 3 - 4j, np.nan])
    assert compare_arrays(reference, reference)["passed"]
    changed = reference.copy()
    changed[0] += 1j
    result = compare_arrays(reference, changed)
    assert not result["passed"]
    assert result["max_absolute_difference"] == 1
    changed[-1] = 0
    assert compare_arrays(reference, changed)["reason"] == "finite support mismatch"


def test_zero_reference_has_explicit_unavailable_relative_metric():
    result = compare_arrays(np.zeros(3), np.zeros(3))
    assert result["passed"]
    assert result["relative_l2"] is None
    canonical_json(result)
    assert not exact_equal(np.arange(3), np.arange(3)[::-1])
    assert not compare_arrays(np.arange(3), np.arange(4))["passed"]


def synthetic_spectrum(path):
    with h5py.File(path, "w") as f:
        g = f.create_group("stokespol/interleave_averaged")
        for name in ("spw_array", "polpair_array", "dly_array", "freq_array", "channel_width", "telescope_location", "scalar_array"):
            g.attrs[name] = np.arange(14)
        for name in ("vis_units", "norm_units", "cosmo", "taper", "norm", "folded"):
            g.attrs[name] = "value"
        for name in ("bl_array", "bl_vecs", "blpair_array", "spw_dly_array", "spw_freq_array", "time_1_array",
                     "time_2_array", "time_avg_array", "lst_1_array", "lst_2_array", "lst_avg_array"):
            g[name] = np.arange(14)
        for spw in range(14):
            for name in (f"data_spw{spw}", f"stats_P_N_{spw}", f"nsample_spw{spw}", f"integration_spw{spw}", f"wgt_spw{spw}"):
                g[name] = np.arange(4.)


def test_full_spectral_comparison_rejects_coordinate_changes_and_missing_spws(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    synthetic_spectrum(a)
    synthetic_spectrum(b)
    assert compare_spectra(a, b)["passed"]
    with h5py.File(b, "a") as f:
        f['stokespol/interleave_averaged/time_avg_array'][0] = 9
    result = compare_spectra(a, b)
    assert not result["passed"] and not result["coordinates"]["time_avg_array"]
    with h5py.File(b, "a") as f:
        del f['stokespol/interleave_averaged/data_spw13']
    result = compare_spectra(a, b)
    assert not result["spectral_windows"][-1]["passed"]
