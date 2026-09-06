import numpy as np
import pytest

from hera_systematics_model.views import analysis_view, geometry_masks
from test_samples import paired


def test_both_slice_directions_preserve_physical_identity(paired):
    arrays, shape, identity = analysis_view(paired, group=1)
    assert shape == (1, 4)
    assert identity["group_ids"] == ["long"]
    np.testing.assert_equal(arrays[0], paired.corrupted[:, 1, :])
    arrays, shape, identity = analysis_view(paired, delay=2)
    assert shape == (2, 1)
    assert identity["kparallel"] == [paired.kparallel[2]]
    np.testing.assert_equal(arrays[0], paired.corrupted[:, :, 2])
    with pytest.raises(ValueError):
        analysis_view(paired, group=0, delay=0)


def test_geometric_masks_do_not_use_power(paired):
    masks = geometry_masks(paired, np.array([1e-7, 2e-7]), buffer_ns=100.)
    assert masks["full"].all()
    assert np.all(masks["horizon_buffer"] <= masks["horizon"])
    paired.corrupted[:] = 1e12
    other = geometry_masks(paired, np.array([1e-7, 2e-7]), buffer_ns=100.)
    for key in masks:
        np.testing.assert_equal(masks[key], other[key])
