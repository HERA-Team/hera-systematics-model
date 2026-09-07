from dataclasses import replace

import numpy as np
import pytest

from hera_systematics_model.records import SpectrumRecords, WindowGrid, matched_indices


@pytest.fixture
def records():
    grid = WindowGrid(2450000., 270.)
    return SpectrumRecords(
        power=np.arange(12.).reshape(4, 3), pn=np.ones((4, 3)), valid=np.ones((4, 3), bool),
        window_ids=np.array([0, 0, 2, 2]), baseline_ids=np.array(["a", "b", "a", "b"]),
        group_ids=np.array(["g", "g", "g", "g"]), time_jd=grid.centers([0, 0, 2, 2]),
        lst_rad=np.array([6.2, 6.2, .1, .1]), baseline_length_m=np.full(4, 10.),
        kperp=np.full(4, .1), delay_s=np.array([-1e-7, 0., 1e-7]),
        kparallel=np.array([-.1, 0., .1]),
        metadata={"spw": 0, "polarization": "pI", "power_units": "mK2 Mpc3 / h3",
                  "cosmology": {"name": "test"}, "sources": [{"path": "synthetic"}],
                  "window_anchor_jd": grid.anchor_jd, "window_seconds": grid.window_seconds,
                  "native_grid_digest": "0" * 64, "n_interleaves": 1,
                  "averaging_configuration": {"native_samples": 3}},
        native_ids=np.array([[0, 1, 2], [0, 1, 2], [6, 7, 8], [6, 7, 8]]),
    )


def reorder(record, order):
    names = ("power", "pn", "valid", "window_ids", "baseline_ids", "group_ids",
             "time_jd", "lst_rad", "baseline_length_m", "kperp", "native_ids")
    return replace(record, **{key: getattr(record, key)[order] for key in names})


def test_join_uses_identity_not_order(records):
    other = reorder(records, [3, 0, 2, 1])
    ci, ii, report = matched_indices(records, other)
    np.testing.assert_array_equal(records.power[ci], other.power[ii])
    assert report["matched_rows"] == 4
    assert report["ideal_unmatched_rows"] == 0


def test_unmatched_are_reported(records):
    ci, ii, report = matched_indices(records, reorder(records, [3, 0]))
    assert len(ci) == len(ii) == 2
    assert report["corrupted_unmatched_rows"] == 2


@pytest.mark.parametrize("key,value", [("spw", 1), ("power_units", "Jy"),
                                        ("cosmology", {"name": "different"}), ("n_interleaves", 2),
                                        ("averaging_configuration", {"native_samples": 2})])
def test_identity_metadata_mismatch(records, key, value):
    other = replace(records, metadata={**records.metadata, key: value})
    with pytest.raises(ValueError, match="metadata mismatch"):
        matched_indices(records, other)


def test_duplicates_and_delay_mismatch(records):
    with pytest.raises(ValueError, match="duplicate"):
        reorder(records, [0, 0])
    with pytest.raises(ValueError, match="coordinate mismatch"):
        matched_indices(records, replace(records, delay_s=records.delay_s * 2))


def test_reference_grid_does_not_reanchor_a_branch(records):
    times = records.time_jd + 20 / 86400
    shifted = replace(records, time_jd=times)
    assert matched_indices(records, shifted)[2]["matched_rows"] == 4
    with pytest.raises(ValueError, match="centroids"):
        replace(records, time_jd=times + 270 / 86400)


def test_same_centroid_bin_cannot_hide_different_native_windows(records):
    other = replace(records, native_ids=records.native_ids + 1)
    with pytest.raises(ValueError, match="native averaging membership mismatch"):
        matched_indices(records, other)
    old = replace(records, native_ids=None)
    with pytest.raises(ValueError, match="memberships are required"):
        matched_indices(records, old)
    other = replace(records, metadata={**records.metadata, "native_grid_digest": "1" * 64})
    with pytest.raises(ValueError, match="native_grid_digest"):
        matched_indices(records, other)


def test_native_padding_does_not_change_physical_membership(records):
    padded = np.pad(records.native_ids, ((0, 0), (0, 2)), constant_values=-1)
    other = replace(records, native_ids=padded)
    assert matched_indices(records, other)[2]["native_membership_verified"]
