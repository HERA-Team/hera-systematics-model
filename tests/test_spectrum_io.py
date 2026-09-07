import numpy as np
import pytest
from types import SimpleNamespace

from hera_systematics_model.spectrum_io import bind_memberships, pair_identity, read_records, spectral_measurements, selected_window_index
from hera_systematics_model.configuration import file_identity
from hera_systematics_model.records import WindowGrid
from test_window_membership import membership


def test_real_zero_and_negative_power_are_measurements_with_valid_contributors():
    data = np.array([[0., -2., 4.], [1., 2., 3.]]) + 0j
    power, pn, valid = spectral_measurements(data, np.ones_like(data), [1, 1], [300, 300], np.ones((2, 5, 2)), True)
    assert valid.all()
    np.testing.assert_equal(power, data.real)
    assert pair_identity(((6, 34), (6, 34))) == "6_34:6_34"


def test_empty_weight_support_is_not_a_measured_zero():
    weights = np.ones((2, 5, 2))
    weights[0, :, 1] = 0
    data = np.zeros((2, 3))
    _, _, valid = spectral_measurements(data, np.ones_like(data), [1, 1], [300, 300], weights, True)
    assert not valid[0].any() and valid[1].all()


def test_noise_sign_or_absence_cannot_be_silently_repaired():
    data = np.ones((2, 3))
    noise = data.copy()
    noise[0, 0] = -1
    _, _, valid = spectral_measurements(data, noise, [1, 1], [300, 300], np.ones((2, 5, 2)), True)
    assert not valid[0, 0]
    with pytest.raises(ValueError, match="lacks"):
        spectral_measurements(data, None, [1, 1], [300, 300], np.ones((2, 5, 2)), True)
    _, pn, valid = spectral_measurements(data, None, [1, 1], [300, 300], np.ones((2, 5, 2)), False)
    assert valid.all() and np.isnan(pn).all()
    with pytest.raises(ValueError, match="real-component"):
        spectral_measurements(data, data + 1j, [1, 1], [300, 300], np.ones((2, 5, 2)), True)


def test_reader_requires_an_export_before_loading_the_io_stack():
    with pytest.raises(ValueError, match="exact native averaging export"):
        read_records("absent.h5", 0, None, None, "ideal")


def selected_window(index=0):
    return SimpleNamespace(spw_array=np.array([index]), folded=False,
        freq_array=np.array([130e6, 131e6, 132e6]), dly_array=np.array([-1e-6, 0., 1e-6]),
        spw_freq_array=np.full(3, index), spw_dly_array=np.full(3, index))


@pytest.mark.parametrize("index", [0, 6, 13])
def test_selected_window_accepts_local_reindexing_only_with_exact_coordinates(index):
    uvp = selected_window(index)
    assert selected_window_index(uvp, uvp.freq_array, uvp.dly_array) == index
    with pytest.raises(ValueError, match="coordinates differ"):
        selected_window_index(uvp, uvp.freq_array + 1e6, uvp.dly_array)
    with pytest.raises(ValueError, match="coordinates differ"):
        selected_window_index(uvp, uvp.freq_array, uvp.dly_array * 2)


@pytest.mark.parametrize("name,value", [("folded", True), ("spw_array", np.array([0, 1])),
    ("spw_array", np.array([-1])), ("spw_freq_array", np.array([0, 0, 1])),
    ("spw_dly_array", np.array([], int))])
def test_ambiguous_selected_window_is_rejected(name, value):
    uvp = selected_window()
    setattr(uvp, name, value)
    with pytest.raises(ValueError):
        selected_window_index(uvp, uvp.freq_array, uvp.dly_array)


@pytest.mark.parametrize("coordinates", [[], [1., 1.], [np.nan], [[1.]], [2., 1.], [-1.]])
def test_invalid_requested_window_is_rejected(coordinates):
    uvp = selected_window()
    with pytest.raises(ValueError, match="coordinates are invalid"):
        selected_window_index(uvp, coordinates, uvp.dly_array)


def test_membership_export_is_bound_to_exact_spectral_content(tmp_path):
    raw, product = tmp_path / "spectrum.h5", tmp_path / "members.npz"
    raw.write_bytes(b"spectral-payload")
    exported = membership()
    exported.metadata["spectrum_source"] = file_identity(raw)
    exported.save(product)
    grid = WindowGrid(2450000., 50.)
    ids, native, provenance = bind_memberships(raw, product, exported.baseline_ids, exported.centroid_jd, grid)
    np.testing.assert_equal(ids, exported.window_ids)
    np.testing.assert_equal(native, exported.native_ids)
    assert provenance["native_grid_digest"] == exported.native_grid_digest
    with pytest.raises(ValueError, match="shared grid"):
        bind_memberships(raw, product, exported.baseline_ids, exported.centroid_jd, WindowGrid(2450000., 10.))
    raw.write_bytes(b"changed-payload")
    with pytest.raises(ValueError, match="different spectral product"):
        bind_memberships(raw, product, exported.baseline_ids, exported.centroid_jd, grid)
