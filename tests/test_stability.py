import numpy as np
import pytest

from hera_systematics_model.scoring import CandidateFailure
from hera_systematics_model.stability import block_bootstrap_indices, bootstrap_stability, compare_components
from test_models import low_rank_data


def test_bootstrap_blocks_never_cross_gaps_or_wrap_arc():
    ids = np.r_[np.arange(25), np.arange(40, 66)]
    rows, starts, lengths = block_bootstrap_indices(ids, 12, np.random.default_rng(5))
    offset = 0
    for start, length in zip(starts, lengths):
        np.testing.assert_equal(rows[offset:offset + length], np.arange(start, start + length))
        assert np.all(np.diff(ids[rows[offset:offset + length]]) == 1)
        offset += length
    assert len(rows) == len(ids)
    assert np.count_nonzero(rows < 25) == 25
    with pytest.raises(CandidateFailure):
        block_bootstrap_indices([0, 1, 10, 11], 3, np.random.default_rng())


def test_sign_permutations_and_degenerate_rotations_are_distinguished():
    basis = np.eye(2, 10)
    trial = np.array([[0., -1.], [1., 0.]]) @ basis
    result = compare_components(basis, trial)
    np.testing.assert_allclose(np.abs(result["signed_cosines"]), 1)
    np.testing.assert_allclose(result["principal_angles"], 0)
    rotation = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2)
    result = compare_components(basis, rotation @ basis)
    np.testing.assert_allclose(result["principal_angles"], 0, atol=2e-8)
    assert np.all(np.abs(result["signed_cosines"]) < .8)


def test_low_rank_bootstrap_preserves_subspace_and_records_every_draw():
    arrays = low_rank_data()
    candidate = {"method": "complete", "representation": "linear", "rank": 2}
    result, meta = bootstrap_stability(arrays, np.arange(40), candidate, n_replicates=5, block_length=12)
    assert meta["complete"]
    np.testing.assert_allclose(result["principal_angles"], 0, atol=4e-8)
    assert result["sample_rows"].shape == (5, 40)
