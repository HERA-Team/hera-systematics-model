from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from hera_systematics_model.ideal import ideal_arrays, interpolate_supported, periodic_source_order, replace_reference_visibilities


def test_source_interpolation_crosses_wrap_without_extrapolation():
    knots = np.array([6.1, 6.2, .02, .12, .22])
    targets = np.array([6.24, .03, .10])
    source, target, order = periodic_source_order(knots, targets)
    values = (source - 2 * np.pi)[:, None] ** 3 + 1j * source[:, None]
    output, valid = interpolate_supported(source, values, np.ones(values.shape, bool), target)
    np.testing.assert_allclose(output[:, 0], (target - 2 * np.pi) ** 3 + 1j * target, atol=1e-12)
    assert valid.all()
    outside, supported = interpolate_supported(source, values, np.ones(values.shape, bool), [source[0] - .1])
    assert not supported.any() and np.isnan(outside).all()


def test_unsupported_knots_are_never_filled_as_observed_data():
    values = np.arange(10.).reshape(5, 2)
    supported = np.ones(values.shape, bool)
    supported[2, 1] = False
    first, valid = interpolate_supported(np.arange(5.), values, supported, [1.5, 2.5])
    values[2, 1] = 1e100
    second, other = interpolate_supported(np.arange(5.), values, supported, [1.5, 2.5])
    np.testing.assert_equal(first, second)
    assert valid[:, 0].all() and not valid[:, 1].any()


def test_ideal_flags_counts_and_reference_metadata_policy():
    class Reference(SimpleNamespace):
        def copy(self):
            return deepcopy(self)
    reference = Reference(data_array=np.ones((2, 3)), flag_array=np.ones((2, 3), bool),
        nsample_array=np.zeros((2, 3)), vis_units="Jy", history="reference")
    for name in ("time_array", "lst_array", "integration_time", "ant_1_array", "ant_2_array", "freq_array", "polarization_array"):
        setattr(reference, name, np.array([1., 2.]))
    values = np.array([[0., 2., np.nan], [1., 0., 3.]])
    supported = np.ones((2, 3), bool)
    supported[1, 2] = False
    result = replace_reference_visibilities(reference, values, supported, "Jy", "sky", "source-hash")
    assert not result.flag_array[0, 0] and result.nsample_array[0, 0] == 1
    assert result.flag_array[0, 2] and result.flag_array[1, 2]
    assert reference.flag_array.all()
    np.testing.assert_equal(result.integration_time, reference.integration_time)
    with pytest.raises(ValueError, match="units"):
        replace_reference_visibilities(reference, values, supported, "mK", "sky", "source-hash")
