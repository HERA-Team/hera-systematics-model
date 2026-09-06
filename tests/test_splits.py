import numpy as np
import pytest

from hera_systematics_model.splits import continuous_segments, feature_partitions, time_folds


def test_time_wrap_and_gaps_keep_native_order_and_guards():
    ids = np.delete(np.arange(140), [12, 40, 99])
    folds = time_folds(ids)
    np.testing.assert_array_equal(np.concatenate([fold.test for fold in folds]), np.arange(len(ids)))
    for fold in folds:
        assert fold.supported
        assert np.min(np.abs(ids[fold.train, None] - ids[fold.test])) > 12
        for segment in continuous_segments(ids, fold.test):
            np.testing.assert_equal(np.diff(ids[segment]), 1)
        for inner in time_folds(ids, n_splits=3, pool=fold.train):
            assert not np.intersect1d(inner.train, fold.test).size
            assert not np.intersect1d(inner.test, fold.test).size
            assert not np.intersect1d(inner.train, fold.guard_rows).size


def test_insufficient_support_is_explicit():
    folds = time_folds(np.arange(12), guard=12)
    assert all(not fold.supported for fold in folds)
    assert all(fold.reason == "insufficient_training_windows" for fold in folds)


def test_time_indices_cannot_be_reordered_or_duplicated():
    for ids in (np.array([2, 0, 1]), np.array([0, 1, 1])):
        with pytest.raises(ValueError):
            time_folds(ids)


@pytest.mark.parametrize("axis,guard", [("delay", 2), ("group", 1)])
def test_target_cells_are_withheld_once_with_spatial_guard(axis, guard):
    shape = (12, 27)
    partitions = feature_partitions(shape, axis=axis, guard=guard)
    np.testing.assert_array_equal(np.sum([p.target for p in partitions], axis=0), 1)
    coordinate = np.indices(shape)[1 if axis == "delay" else 0].ravel()
    for p in partitions:
        assert not (p.target & p.predictor).any()
        assert not (p.guard & p.predictor).any()
        assert (p.target | p.predictor | p.guard).all()
        assert np.min(np.abs(coordinate[p.target, None] - coordinate[p.predictor])) > guard
