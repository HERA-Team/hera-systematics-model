import numpy as np
import pytest

from hera_systematics_model.scoring import CandidateFailure
from hera_systematics_model.stability import block_bootstrap_indices, bootstrap_stability, compare_components, spectral_clusters
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
    np.testing.assert_equal(np.sort(result["assignments"], axis=1), np.tile([0, 1], (5, 1)))
    assert np.all(np.isin(result["signs"], [-1., 1.]))
    assert np.isfinite(result["cluster_principal_angles"]).all()
    assert sorted(sum(meta["spectral_clusters"], [])) == [0, 1]
    assert meta["cluster_crosses_rank_boundary"] is False


def test_spectral_clusters_report_close_modes_and_unresolved_rank_boundary():
    clusters, boundary = spectral_clusters([10, 9.5, 4, 3.9], 3)
    assert clusters == [[0, 1], [2]]
    assert boundary is True
    assert spectral_clusters([10, 9.5, 4], 3)[1] is None
    assert spectral_clusters([], 0) == ([], None)
    for energies, rank in [([1, 2], 2), ([1], -1), ([1], 1.5), ([np.nan], 1), ([-1], 1)]:
        with pytest.raises(ValueError, match="spectral energies"):
            spectral_clusters(energies, rank)


@pytest.mark.parametrize("method", ["complete", "masked", "kernel"])
def test_bootstrap_region_ignores_excluded_power(method):
    from hera_systematics_model.sensitivity import GeometricRegion

    arrays = low_rank_data()
    mask = np.arange(30) < 20
    region = GeometricRegion(mask, "horizon")
    candidate = {"method": method, "representation": "linear", "rank": 2}
    if method == "kernel":
        candidate.update(bandwidth=1., alpha=1.)
    result, meta = bootstrap_stability(arrays, np.arange(40), candidate,
        n_replicates=2, training_filter=region)
    changed = tuple(a.copy() for a in arrays)
    changed[0][:, ~mask] += np.arange(40)[:, None] * 1e12
    other, other_meta = bootstrap_stability(changed, np.arange(40), candidate,
        n_replicates=2, training_filter=region)
    assert meta["complete"] and other_meta["complete"]
    for key in result:
        np.testing.assert_allclose(result[key], other[key], atol=1e-7)
    assert not result["reference_feature_mask"][~mask].any()
    assert not result["replicate_feature_masks"][:, ~mask].any()
    assert meta["reference_training_filter"]["region"] == "horizon"
    assert all(r["training_filter"]["region"] == "horizon" for r in meta["records"])


def test_bootstrap_exclusions_use_each_sample_with_repeated_rows():
    from hera_systematics_model.sensitivity import GroupExclusion

    arrays = low_rank_data()
    exclusion = GroupExclusion((3, 10), ("a", "b", "c"), 1, "noise_weighted_energy")
    candidate = {"method": "complete", "representation": "linear", "rank": 2}
    result, meta = bootstrap_stability(arrays, np.arange(40), candidate,
        n_replicates=3, training_filter=exclusion)
    assert meta["complete"]
    for sampled, record in zip(result["sample_rows"], meta["records"]):
        expected = exclusion(arrays, sampled)[1]
        assert record["training_filter"] == expected
        assert len(np.unique(sampled)) < len(sampled)
