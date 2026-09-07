"""Training-only group exclusions with fixed physical feature coordinates."""

from dataclasses import dataclass

import numpy as np

from .models import fit_complete, measured_arrays
from .scoring import CandidateFailure


@dataclass(frozen=True)
class GeometricRegion:
    """Apply a fixed physical mask while retaining the original denominator."""

    feature_mask: np.ndarray
    name: str

    def __post_init__(self):
        mask = np.asarray(self.feature_mask)
        if mask.ndim != 1 or mask.dtype != bool or not mask.size:
            raise ValueError("region requires a nonempty one-dimensional boolean mask")
        if self.name not in ("full", "horizon", "horizon_buffer"):
            raise ValueError("unknown geometric region")
        mask = mask.copy()
        mask.flags.writeable = False
        object.__setattr__(self, "feature_mask", mask)

    def __call__(self, arrays, training_rows):
        arrays = measured_arrays(*arrays)
        if arrays[0].shape[1] != len(self.feature_mask):
            raise ValueError("region feature dimensions disagree")
        if not self.feature_mask.any():
            raise CandidateFailure("geometric region contains no feature cells")
        filtered = (*arrays[:3], arrays[3] & self.feature_mask)
        return filtered, {"region": self.name, "data_dependent": False,
            "excluded_feature_indices": np.flatnonzero(~self.feature_mask).tolist(),
            "original_features": len(self.feature_mask),
            "retained_features": int(self.feature_mask.sum())}


@dataclass(frozen=True)
class CombinedFilters:
    """Apply each filter to the preceding support using the same training rows."""

    filters: tuple

    def __call__(self, arrays, training_rows):
        reports = []
        for function in self.filters:
            arrays, report = function(arrays, training_rows)
            reports.append(report)
        return arrays, {"ordered_filters": reports}


@dataclass(frozen=True)
class GroupExclusion:
    shape: tuple
    group_ids: tuple
    count: int
    metric: str = "leading_linear_loading"

    def __post_init__(self):
        if (len(self.shape) != 2 or len(self.group_ids) != self.shape[0]
                or min(self.shape) < 1 or not 0 < self.count < self.shape[0]
                or len(set(self.group_ids)) != len(self.group_ids)
                or self.metric not in ("leading_linear_loading", "noise_weighted_energy")):
            raise ValueError("invalid physical group exclusion configuration")

    def __call__(self, arrays, training_rows):
        arrays = measured_arrays(*arrays)
        if arrays[0].shape[1] != np.prod(self.shape):
            raise ValueError("exclusion feature identity dimensions disagree")
        power, ideal, pn, valid = (array[training_rows] for array in arrays)
        if self.metric == "leading_linear_loading":
            model = fit_complete(power, ideal, pn, valid, 1, representation="linear")
            contributions = (model.components[0].reshape(self.shape) ** 2).sum(axis=1)
            supported = model.feature_mask.reshape(self.shape).any(axis=1)
        else:
            standardized = np.divide(power - ideal, pn, out=np.zeros_like(power), where=valid)
            contributions = (standardized ** 2).reshape((-1, *self.shape)).sum(axis=(0, 2))
            supported = valid.reshape((-1, *self.shape)).any(axis=(0, 2))
        if not np.isfinite(contributions).all() or not contributions.sum() > 0:
            raise CandidateFailure("group contribution measure has no finite positive support")
        order = sorted(np.flatnonzero(supported), key=lambda i: (-contributions[i], self.group_ids[i]))
        if len(order) <= self.count:
            raise CandidateFailure("too few supported groups for exclusion sensitivity")
        excluded = order[:self.count]
        feature_mask = np.ones(self.shape, bool)
        feature_mask[excluded] = False
        filtered = (*arrays[:3], arrays[3] & feature_mask.ravel())
        return filtered, {"metric": self.metric, "training_rows": np.asarray(training_rows).tolist(),
            "excluded_group_ids": [self.group_ids[index] for index in excluded],
            "excluded_feature_indices": np.flatnonzero(~feature_mask.ravel()).tolist(),
            "contribution_shares": (contributions / contributions.sum()).tolist(),
            "group_ids": list(self.group_ids), "training_supported_groups": supported.tolist(),
            "original_features": arrays[0].shape[1], "retained_features": int(feature_mask.sum())}
