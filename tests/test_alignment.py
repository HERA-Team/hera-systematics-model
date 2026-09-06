from dataclasses import replace

import numpy as np
import pytest

from hera_systematics_model.alignment import build_paired, delay_pairs
from test_records import records, reorder


def test_common_weights_ignore_ideal_noise_and_row_order(records):
    cp = np.array([[2., 0., 6.], [10., 0., 14.], [2., 0., 6.], [10., 0., 14.]])
    ip = cp - np.array([1., 3., 1., 3.])[:, None]
    noise = np.array([1., 2., 1., 2.])[:, None] * np.ones((4, 3))
    corrupted = replace(records, power=cp, pn=noise)
    ideal = reorder(replace(records, power=ip, pn=np.full((4, 3), np.nan)), [3, 2, 1, 0])
    result = build_paired(corrupted, ideal)
    # Each side has normalized baseline weights .8/.2, then receives half weight.
    np.testing.assert_allclose(result.weights[:, :, 0, 0], [[.4, .1], [.4, .1]])
    np.testing.assert_allclose(result.corrupted, 5.6)
    np.testing.assert_allclose(result.residual, 1.4)
    np.testing.assert_allclose(result.pn, np.sqrt(.4))


def test_invalid_baseline_does_not_contribute_fake_zero(records):
    valid = records.valid.copy()
    valid[1] = False
    source = replace(records, valid=valid)
    result = build_paired(source, records, quorum=.5)
    assert result.weights[0, 1].sum() == 0
    np.testing.assert_allclose(result.corrupted[0, 0], 1.)
    np.testing.assert_allclose(result.residual, 0.)
    strict = build_paired(source, records, quorum=1.)
    np.testing.assert_array_equal(strict.window_ids, [2])


def test_missing_side_invalidates_whole_folded_cell(records):
    valid = records.valid.copy()
    valid[:2, 0] = False
    result = build_paired(replace(records, valid=valid), records)
    assert not result.valid[0, 0, 0]
    assert result.weights[0].sum() == 0
    assert np.isnan(result.residual[0, 0, 0])


def test_delay_counterpart_checks():
    with pytest.raises(ValueError, match="counterpart"):
        delay_pairs(np.array([-2., 0., 1.]), np.array([-2., 0., 1.]))
    with pytest.raises(ValueError, match="kparallel"):
        delay_pairs(np.array([-1., 0., 1.]), np.array([-2., 0., 1.]))
    np.testing.assert_array_equal(delay_pairs(np.array([-2., -1., 0., 1.]),
                                            np.array([-2., -1., 0., 1.])), [[1, 3]])
