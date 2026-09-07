"""Exact native-sample membership exported by spectral averaging."""

from dataclasses import dataclass

import numpy as np

from .artifacts import canonical_json, read_artifact, write_artifact
from .configuration import digest_json


@dataclass
class WindowMemberships:
    baseline_ids: np.ndarray
    centroid_jd: np.ndarray
    window_ids: np.ndarray
    native_ids: np.ndarray
    native_time_jd: np.ndarray
    metadata: dict

    def __post_init__(self):
        for name in ("baseline_ids", "centroid_jd", "window_ids", "native_ids", "native_time_jd"):
            setattr(self, name, np.asarray(getattr(self, name)))
        n = len(self.baseline_ids)
        if (not n or self.baseline_ids.shape != (n,) or self.baseline_ids.dtype.kind not in "US"
                or np.any(self.baseline_ids == "") or self.centroid_jd.shape != (n,)
                or not np.isfinite(self.centroid_jd).all() or self.window_ids.shape != (n,)
                or self.window_ids.dtype.kind not in "iu"):
            raise ValueError("invalid averaging row identities")
        if (self.native_time_jd.ndim != 1 or not len(self.native_time_jd)
                or not np.isfinite(self.native_time_jd).all() or np.any(np.diff(self.native_time_jd) <= 0)):
            raise ValueError("native reference times must be strictly increasing")
        if (self.native_ids.ndim != 2 or self.native_ids.shape[0] != n or not self.native_ids.shape[1]
                or self.native_ids.dtype.kind not in "iu" or np.any(self.native_ids < -1)
                or np.any(self.native_ids >= len(self.native_time_jd))):
            raise ValueError("invalid native averaging membership")
        for centroid, members in zip(self.centroid_jd, self.native_ids):
            present = members[members >= 0]
            if (not len(present) or np.any(np.diff(present) <= 0)
                    or not np.array_equal(members[:len(present)], present)):
                raise ValueError("native members must be ordered, unique and followed only by padding")
            times = self.native_time_jd[present]
            if not times[0] <= centroid <= times[-1]:
                raise ValueError("spectrum centroid lies outside its native samples")
        if len(set(self.keys)) != n:
            raise ValueError("duplicate baseline and centroid identity")
        if len(set(zip(self.baseline_ids.tolist(), self.window_ids.tolist()))) != n:
            raise ValueError("duplicate baseline and window identity")
        required = {"spectrum_source", "native_time_source", "n_interleaves", "averaging_configuration"}
        if (not required.issubset(self.metadata) or not self.metadata["spectrum_source"]
                or not self.metadata["native_time_source"] or type(self.metadata["n_interleaves"]) is not int
                or self.metadata["n_interleaves"] <= 0
                or not isinstance(self.metadata["averaging_configuration"], dict)
                or not self.metadata["averaging_configuration"]):
            raise ValueError("averaging provenance is required")
        canonical_json(self.metadata)

    @property
    def keys(self):
        return list(zip(self.baseline_ids.tolist(), self.centroid_jd.tolist()))

    @property
    def native_grid_digest(self):
        return digest_json(self.native_time_jd.tolist())

    def lookup(self, baseline_ids, centroid_jd):
        """Match exported physical row identities exactly; never round centroids."""
        if len(baseline_ids) != len(centroid_jd):
            raise ValueError("averaging lookup row lengths differ")
        positions = {key: row for row, key in enumerate(self.keys)}
        requested = list(zip(np.asarray(baseline_ids).tolist(), np.asarray(centroid_jd).tolist()))
        if any(key not in positions for key in requested):
            raise ValueError("spectrum row lacks exact native averaging membership")
        indices = np.asarray([positions[key] for key in requested], int)
        return self.window_ids[indices], self.native_ids[indices]

    def save(self, path):
        self.__post_init__()
        arrays = {name: getattr(self, name) for name in ("baseline_ids", "centroid_jd", "window_ids",
                                                        "native_ids", "native_time_jd")}
        return write_artifact(path, "window-memberships", arrays,
                              {**self.metadata, "native_grid_digest": self.native_grid_digest})

    @classmethod
    def load(cls, path):
        arrays, metadata = read_artifact(path, "window-memberships")
        result = cls(**arrays, metadata=metadata)
        if metadata.get("native_grid_digest") != result.native_grid_digest:
            raise ValueError("native reference grid identity mismatch")
        return result
