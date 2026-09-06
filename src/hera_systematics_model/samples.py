"""Coordinate-explicit paired, folded power-spectrum samples."""

from dataclasses import dataclass, fields

import numpy as np

from .artifacts import canonical_json, read_artifact, write_artifact


@dataclass
class PairedSamples:
    """Arrays use (time, group, positive delay), with no implicit row identity.

    Contributor weights use (time, baseline, positive delay, side), where side
    0 is negative delay and side 1 is positive delay. Each valid folded cell
    has total weight one, half from each side. Both branches use these weights.
    """

    corrupted: np.ndarray
    ideal: np.ndarray
    pn: np.ndarray
    valid: np.ndarray
    window_ids: np.ndarray
    time_jd: np.ndarray
    lst_rad: np.ndarray
    group_ids: np.ndarray
    baseline_length_m: np.ndarray
    delay_s: np.ndarray
    kperp: np.ndarray
    kparallel: np.ndarray
    baseline_ids: np.ndarray
    baseline_group: np.ndarray
    weights: np.ndarray
    metadata: dict

    def __post_init__(self):
        for field in fields(self):
            if field.name != "metadata":
                setattr(self, field.name, np.asarray(getattr(self, field.name)))
        self.validate()

    def validate(self):
        shape = self.corrupted.shape
        if len(shape) != 3 or min(shape) == 0:
            raise ValueError("power must have nonempty time/group/delay axes")
        nt, ng, nd = shape
        for name in ("ideal", "pn", "valid"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape mismatch")
        if self.valid.dtype.kind != "b":
            raise ValueError("validity must be boolean")
        for name in ("corrupted", "ideal", "pn"):
            value = getattr(self, name)
            if value.dtype.kind not in "fi" or not np.isfinite(value[self.valid]).all():
                raise ValueError(f"invalid measured {name}")
        if np.any(self.pn[self.valid] <= 0):
            raise ValueError("valid samples need positive noise")
        for name, size in (("window_ids", nt), ("time_jd", nt), ("lst_rad", nt),
                           ("group_ids", ng), ("baseline_length_m", ng),
                           ("kperp", ng), ("delay_s", nd), ("kparallel", nd)):
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} coordinate shape mismatch")
        if self.window_ids.dtype.kind not in "iu":
            raise ValueError("window identities must be integers")
        for name in ("window_ids", "time_jd", "delay_s", "kparallel"):
            value = getattr(self, name)
            if not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
                raise ValueError(f"{name} must be finite and strictly increasing")
        for name in ("baseline_length_m", "kperp", "delay_s", "kparallel"):
            value = getattr(self, name)
            if not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if (not np.isfinite(self.lst_rad).all()
                or np.any((self.lst_rad < 0) | (self.lst_rad >= 2 * np.pi))):
            raise ValueError("LST must be finite radians in [0, 2pi)")
        for name in ("group_ids", "baseline_ids"):
            value = getattr(self, name)
            if (value.ndim != 1 or value.dtype.kind not in "US"
                    or len(np.unique(value)) != len(value) or np.any(value == "")):
                raise ValueError(f"{name} must contain unique string identities")
        nb = len(self.baseline_ids)
        if (self.baseline_group.shape != (nb,) or self.baseline_group.dtype.kind not in "iu"
                or np.any((self.baseline_group < 0) | (self.baseline_group >= ng))):
            raise ValueError("invalid baseline membership")
        if (self.weights.shape != (nt, nb, nd, 2)
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)):
            raise ValueError("invalid contributor weights")
        totals = np.zeros((nt, ng, nd, 2))
        for group in range(ng):
            totals[:, group] = self.weights[:, self.baseline_group == group].sum(axis=1)
        if not np.allclose(totals, self.valid[..., None] * 0.5, rtol=1e-12, atol=1e-14):
            raise ValueError("folded contributor weights disagree with validity")
        required = {"spw", "polarization", "power_units", "cosmology", "sources",
                    "window_anchor_jd", "window_seconds", "noise_model"}
        if not isinstance(self.metadata, dict) or not required.issubset(self.metadata):
            raise ValueError("missing physical or source metadata")
        canonical_json(self.metadata)
        if (type(self.metadata["spw"]) is not int or self.metadata["spw"] < 0
                or self.metadata["window_seconds"] <= 0
                or not self.metadata["sources"] or not self.metadata["cosmology"]
                or any(not isinstance(self.metadata[k], str) or not self.metadata[k].strip()
                       for k in ("polarization", "power_units", "noise_model"))):
            raise ValueError("invalid spectral window, time grid or sources")
        if np.any(np.diff(self.baseline_length_m) < 0) or np.any(np.diff(self.kperp) < 0):
            raise ValueError("baseline groups must be ordered by physical length")
        centers = (self.metadata["window_anchor_jd"]
                   + (self.window_ids + 0.5) * self.metadata["window_seconds"] / 86400)
        if not np.allclose(centers, self.time_jd, rtol=0, atol=1e-9):
            raise ValueError("time centers disagree with window identities")

    @property
    def residual(self):
        return np.where(self.valid, self.corrupted - self.ideal, np.nan)

    @property
    def contributor_counts(self):
        return np.stack([
            (self.weights[:, self.baseline_group == group] > 0).sum(axis=1)
            for group in range(len(self.group_ids))
        ], axis=1)

    def save(self, path):
        self.validate()
        arrays = {field.name: getattr(self, field.name) for field in fields(self)
                  if field.name != "metadata"}
        return write_artifact(path, "paired-samples", arrays, self.metadata)

    @classmethod
    def load(cls, path):
        arrays, metadata = read_artifact(path, "paired-samples")
        return cls(**arrays, metadata=metadata)
