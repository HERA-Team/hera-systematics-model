import numpy as np
import pytest

from hera_systematics_model.window_membership import WindowMemberships


def membership():
    times = 2450000. + np.arange(20) * 10 / 86400
    return WindowMemberships(np.array(["0_1:0_1", "0_2:0_2"]), times[[2, 7]], np.array([0, 1]),
        np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), times,
        {"spectrum_source": {"sha256": "abc"}, "native_time_source": {"sha256": "def"},
         "n_interleaves": 1, "averaging_configuration": {"native_samples": 5}})


def test_native_membership_lookup_is_identity_preserving_and_roundtrips(tmp_path):
    data = membership()
    ids, native = data.lookup(data.baseline_ids[::-1], data.centroid_jd[::-1])
    np.testing.assert_equal(ids, [1, 0])
    np.testing.assert_equal(native, data.native_ids[::-1])
    path = tmp_path / "membership.npz"
    data.save(path)
    restored = WindowMemberships.load(path)
    assert restored.native_grid_digest == data.native_grid_digest
    np.testing.assert_equal(restored.lookup(data.baseline_ids, data.centroid_jd)[1], data.native_ids)
    with pytest.raises(ValueError, match="exact native"):
        data.lookup(data.baseline_ids, data.centroid_jd + 1e-8)


def test_native_membership_rejects_false_or_ambiguous_support():
    for invalid in ([0, 1, 1, 3, 4], [0, -1, 2, 3, 4], [5, 6, 7, 8, 9]):
        data = membership()
        data.native_ids[0] = invalid
        with pytest.raises(ValueError):
            data.__post_init__()
