"""Time and feature partitions expressed in physical grid coordinates."""

from dataclasses import dataclass

import numpy as np


def continuous_segments(window_ids, rows):
    """Return index arrays without connecting missing native windows."""
    rows = np.asarray(rows, dtype=int)
    if not len(rows):
        return []
    return np.split(rows, np.flatnonzero(np.diff(np.asarray(window_ids)[rows]) != 1) + 1)


@dataclass(frozen=True)
class TimeFold:
    train: np.ndarray
    test: np.ndarray
    guard_rows: np.ndarray
    interval: tuple
    supported: bool
    reason: str | None


def time_folds(window_ids, n_splits=4, guard=12, pool=None, min_train=3, min_test=2):
    """Partition the physical span, then intersect with an optional training pool.

    Guard widths count native windows, including absent windows. Inner splits
    retain original identities and never renumber the outer training pool.
    """
    ids = np.asarray(window_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0 or np.any(np.diff(ids) <= 0):
        raise ValueError("strictly increasing integer window identities required")
    if not isinstance(n_splits, int) or n_splits < 2 or not isinstance(guard, int) or guard < 0:
        raise ValueError("invalid fold count or guard")
    rows = np.arange(len(ids)) if pool is None else np.asarray(pool)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
            or np.any((rows < 0) | (rows >= len(ids))) or len(np.unique(rows)) != len(rows)):
        raise ValueError("invalid time-split pool")
    rows = np.sort(rows)
    edges = np.linspace(ids[rows[0]], ids[rows[-1]] + 1, n_splits + 1)
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        # Integer half-open boundaries make every native window belong once.
        low, high = int(np.ceil(low)), int(np.ceil(high))
        test = rows[(ids[rows] >= low) & (ids[rows] < high)]
        train = rows[(ids[rows] < low - guard) | (ids[rows] >= high + guard)]
        guard_rows = np.setdiff1d(rows, np.concatenate([test, train]))
        reason = None
        if len(train) < min_train:
            reason = "insufficient_training_windows"
        elif len(test) < min_test:
            reason = "insufficient_test_windows"
        result.append(TimeFold(train, test, guard_rows, (low, high), reason is None, reason))
    return result


@dataclass(frozen=True)
class FeaturePartition:
    target: np.ndarray
    predictor: np.ndarray
    guard: np.ndarray


def feature_partitions(shape, axis="delay", n_splits=5, guard=2):
    """Create contiguous target bands and adjacent predictor exclusions."""
    if len(shape) != 2 or min(shape) <= 0 or axis not in ("delay", "group"):
        raise ValueError("positive group/delay dimensions and a valid axis required")
    if n_splits < 2 or guard < 0:
        raise ValueError("invalid feature partition settings")
    size = shape[1] if axis == "delay" else shape[0]
    coordinate = np.broadcast_to(np.arange(size)[None, :] if axis == "delay"
                                 else np.arange(size)[:, None], shape)
    partitions = []
    for block in np.array_split(np.arange(size), n_splits):
        target = np.isin(coordinate, block)
        unavailable = (np.zeros(shape, bool) if not len(block) else
                       (coordinate >= block[0] - guard) & (coordinate <= block[-1] + guard))
        partitions.append(FeaturePartition(target.ravel(), (~unavailable).ravel(),
                                           (unavailable & ~target).ravel()))
    return partitions
