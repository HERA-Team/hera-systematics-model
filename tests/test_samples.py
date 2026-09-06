from dataclasses import replace

import numpy as np
import pytest

from hera_systematics_model.samples import PairedSamples


@pytest.fixture
def paired():
    anchor, seconds = 2450000., 270.
    shape = (3, 2, 4)
    return PairedSamples(
        corrupted=np.zeros(shape), ideal=np.ones(shape), pn=np.ones(shape),
        valid=np.ones(shape, bool), window_ids=np.array([0, 1, 3]),
        time_jd=anchor + (np.array([0, 1, 3]) + .5) * seconds / 86400,
        lst_rad=np.array([6.2, .1, .3]), group_ids=np.array(["short", "long"]),
        baseline_length_m=np.array([10., 20.]), delay_s=np.arange(1, 5) * 1e-7,
        kperp=np.array([.1, .2]), kparallel=np.arange(1, 5) * .1,
        baseline_ids=np.array(["0:1", "0:2"]), baseline_group=np.array([0, 1]),
        weights=np.full((3, 2, 4, 2), .5),
        metadata={"spw": 0, "polarization": "pI", "power_units": "mK2 Mpc3 / h3",
                  "cosmology": {"name": "test"}, "sources": [{"path": "synthetic"}],
                  "window_anchor_jd": anchor, "window_seconds": seconds,
                  "noise_model": "corrupted_diagonal_independent_delay_sides"},
    )


def test_zero_is_measurement_and_wrap_is_valid(paired, tmp_path):
    np.testing.assert_equal(paired.residual, -1.)
    np.testing.assert_equal(paired.contributor_counts, 1)
    path = tmp_path / "paired.npz"
    paired.save(path)
    other = PairedSamples.load(path)
    np.testing.assert_array_equal(other.residual, paired.residual)
    np.testing.assert_array_equal(other.window_ids, paired.window_ids)


@pytest.mark.parametrize("field,value", [
    ("window_ids", np.array([0, 1, 1])),
    ("time_jd", np.zeros(3)),
    ("group_ids", np.array(["same", "same"])),
    ("kperp", np.array([np.nan, .2])),
    ("kparallel", np.array([0., .1, .2, .3])),
    ("baseline_group", np.array([0, 2])),
    ("weights", np.ones((3, 2, 4, 2))),
    ("pn", np.zeros((3, 2, 4))),
    ("metadata", {}),
])
def test_invalid_sample_contract(paired, field, value):
    with pytest.raises(ValueError):
        replace(paired, **{field: value})


def test_missing_cell_has_no_weight(paired):
    valid = paired.valid.copy()
    valid[0, 0, 0] = False
    with pytest.raises(ValueError, match="weights"):
        replace(paired, valid=valid)
    weights = paired.weights.copy()
    weights[0, 0, 0] = 0
    sample = replace(paired, valid=valid, weights=weights)
    assert np.isnan(sample.residual[0, 0, 0])
