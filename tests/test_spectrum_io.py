import numpy as np
import pytest

from hera_systematics_model.spectrum_io import bind_memberships, pair_identity, read_records, spectral_measurements
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
