"""Physical window identities and keyed joins before redundant averaging."""

from dataclasses import dataclass, fields

import numpy as np

from .artifacts import canonical_json, read_artifact, write_artifact


@dataclass(frozen=True)
class WindowGrid:
    anchor_jd: float
    window_seconds: float

    def __post_init__(self):
        if not np.isfinite(self.anchor_jd) or not 0 < self.window_seconds < np.inf:
            raise ValueError("a finite explicit reference grid is required")

    def assign(self, times):
        times = np.asarray(times, dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("nonfinite time centroid")
        return np.floor((times - self.anchor_jd) * 86400 / self.window_seconds).astype(np.int64)

    def centers(self, ids):
        return self.anchor_jd + (np.asarray(ids) + .5) * self.window_seconds / 86400


@dataclass
class SpectrumRecords:
    """One spectrum per baseline and physical window; row order is arbitrary."""

    power: np.ndarray
    pn: np.ndarray
    valid: np.ndarray
    window_ids: np.ndarray
    baseline_ids: np.ndarray
    group_ids: np.ndarray
    time_jd: np.ndarray
    lst_rad: np.ndarray
    baseline_length_m: np.ndarray
    kperp: np.ndarray
    delay_s: np.ndarray
    kparallel: np.ndarray
    metadata: dict
    native_ids: np.ndarray | None = None

    def __post_init__(self):
        for field in fields(self):
            if field.name != "metadata" and getattr(self, field.name) is not None:
                setattr(self, field.name, np.asarray(getattr(self, field.name)))
        if self.power.ndim != 2 or min(self.power.shape) == 0:
            raise ValueError("records must have nonempty row and delay axes")
        nr, nd = self.power.shape
        if self.native_ids is None:
            self.native_ids = np.empty((nr, 0), dtype=np.int64)
        if (self.native_ids.ndim != 2 or self.native_ids.shape[0] != nr
                or self.native_ids.dtype.kind not in "iu" or np.any(self.native_ids < -1)):
            raise ValueError("invalid native averaging member axes")
        if self.native_ids.shape[1]:
            digest = self.metadata.get("native_grid_digest", "")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(value not in "0123456789abcdef" for value in digest)):
                raise ValueError("native reference grid identity is required")
            if (type(self.metadata.get("n_interleaves")) is not int or self.metadata["n_interleaves"] <= 0
                    or not isinstance(self.metadata.get("averaging_configuration"), dict)
                    or not self.metadata["averaging_configuration"]):
                raise ValueError("native averaging configuration is required")
            for row in self.native_ids:
                present = row[row >= 0]
                if (not len(present) or np.any(np.diff(present) <= 0)
                        or not np.array_equal(row[:len(present)], present)):
                    raise ValueError("invalid native averaging member order or padding")
        if self.pn.shape != (nr, nd) or self.valid.shape != (nr, nd):
            raise ValueError("power, noise and validity shapes disagree")
        if (self.valid.dtype.kind != "b" or self.power.dtype.kind not in "fiu"
                or self.pn.dtype.kind not in "fiu" or not np.isfinite(self.power[self.valid]).all()):
            raise ValueError("invalid measured power or validity")
        for name in ("window_ids", "baseline_ids", "group_ids", "time_jd", "lst_rad",
                     "baseline_length_m", "kperp"):
            if getattr(self, name).shape != (nr,):
                raise ValueError(f"{name} row shape mismatch")
        if self.window_ids.dtype.kind not in "iu":
            raise ValueError("window identities must be integers")
        for name in ("baseline_ids", "group_ids"):
            if getattr(self, name).dtype.kind not in "US" or np.any(getattr(self, name) == ""):
                raise ValueError("nonempty string baseline/group identities are required")
        for name in ("time_jd", "lst_rad", "baseline_length_m", "kperp"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"nonfinite {name}")
        if np.any(self.baseline_length_m <= 0) or np.any(self.kperp <= 0):
            raise ValueError("cross-baseline lengths and kperp must be positive")
        if np.any((self.lst_rad < 0) | (self.lst_rad >= 2 * np.pi)):
            raise ValueError("LST outside [0, 2pi)")
        for name in ("delay_s", "kparallel"):
            value = getattr(self, name)
            if (value.shape != (nd,) or not np.isfinite(value).all()
                    or np.any(np.diff(value) <= 0)):
                raise ValueError(f"invalid ordered {name} coordinate")
        required = {"spw", "polarization", "power_units", "cosmology", "sources",
                    "window_anchor_jd", "window_seconds"}
        if not required.issubset(self.metadata):
            raise ValueError("missing record metadata")
        canonical_json(self.metadata)
        if "frequency_hz" in self.metadata:
            frequency = np.asarray(self.metadata["frequency_hz"], dtype=float)
            if (frequency.ndim != 1 or not len(frequency) or not np.isfinite(frequency).all()
                    or np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0)):
                raise ValueError("invalid frequency coordinates")
        if (type(self.metadata["spw"]) is not int or self.metadata["spw"] < 0
                or not self.metadata["sources"] or not self.metadata["cosmology"]
                or any(not isinstance(self.metadata[k], str) or not self.metadata[k].strip()
                       for k in ("polarization", "power_units"))):
            raise ValueError("invalid physical record metadata")
        grid = WindowGrid(self.metadata["window_anchor_jd"], self.metadata["window_seconds"])
        if not np.array_equal(grid.assign(self.time_jd), self.window_ids):
            raise ValueError("time centroids disagree with reference window identities")
        if len(set(self.keys)) != nr:
            raise ValueError("duplicate baseline/window identity")
        for baseline in np.unique(self.baseline_ids):
            rows = self.baseline_ids == baseline
            for name in ("group_ids", "baseline_length_m", "kperp"):
                if len(np.unique(getattr(self, name)[rows])) != 1:
                    raise ValueError("baseline geometry changes between windows")

    @property
    def keys(self):
        return list(zip(self.window_ids.tolist(), self.baseline_ids.tolist()))

    def save(self, path):
        self.__post_init__()
        arrays = {field.name: getattr(self, field.name) for field in fields(self)
                  if field.name != "metadata"}
        return write_artifact(path, "spectrum-records", arrays, self.metadata)

    @classmethod
    def load(cls, path):
        arrays, metadata = read_artifact(path, "spectrum-records")
        return cls(**arrays, metadata=metadata)


def matched_indices(corrupted, ideal):
    """Return a one-to-one physical join and explicit unmatched row counts."""
    if not corrupted.native_ids.shape[1] or not ideal.native_ids.shape[1]:
        raise ValueError("native averaging memberships are required for residual matching")
    if canonical_json(corrupted.metadata.get("frequency_hz")) != canonical_json(ideal.metadata.get("frequency_hz")):
        raise ValueError("branch frequency coordinates are absent or different")
    for key in ("spw", "polarization", "power_units", "cosmology",
                "window_anchor_jd", "window_seconds", "native_grid_digest", "n_interleaves", "averaging_configuration"):
        if canonical_json(corrupted.metadata[key]) != canonical_json(ideal.metadata[key]):
            raise ValueError(f"branch metadata mismatch: {key}")
    for key, atol in (("delay_s", 1e-15), ("kparallel", 1e-12)):
        left, right = getattr(corrupted, key), getattr(ideal, key)
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-10, atol=atol):
            raise ValueError(f"branch coordinate mismatch: {key}")
    cm = {key: row for row, key in enumerate(corrupted.keys)}
    im = {key: row for row, key in enumerate(ideal.keys)}
    common = sorted(cm.keys() & im.keys())
    if not common:
        raise ValueError("no common physical samples")
    ci = np.array([cm[key] for key in common], dtype=int)
    ii = np.array([im[key] for key in common], dtype=int)
    for left, right in zip(corrupted.native_ids[ci], ideal.native_ids[ii]):
        if not np.array_equal(left[left >= 0], right[right >= 0]):
            raise ValueError("branch native averaging membership mismatch")
    if not np.array_equal(corrupted.group_ids[ci], ideal.group_ids[ii]):
        raise ValueError("branch group identity mismatch")
    for key in ("baseline_length_m", "kperp"):
        if not np.allclose(getattr(corrupted, key)[ci], getattr(ideal, key)[ii],
                           rtol=1e-10, atol=1e-12):
            raise ValueError(f"branch geometry mismatch: {key}")
    return ci, ii, {"matched_rows": len(common), "native_membership_verified": True,
                    "corrupted_unmatched_rows": len(cm) - len(common),
                    "ideal_unmatched_rows": len(im) - len(common)}
