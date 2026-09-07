"""Explicit HDF5 spectral selection for comparisons with merged libraries."""

import numpy as np


def baseline_pair_identity(code):
    """Decode the stored pair of six-digit, 100-offset antenna-pair codes."""
    if not isinstance(code, (int, np.integer)) or code <= 0:
        raise ValueError("positive integer baseline-pair code required")
    baselines = [int(code) // 1000000, int(code) % 1000000]
    pairs = [(value // 1000 - 100, value % 1000 - 100) for value in baselines]
    if any(not 0 <= antenna < 900 for pair in pairs for antenna in pair):
        raise ValueError("invalid encoded antenna identity")
    return ":".join(f"{a}_{b}" for a, b in pairs)


class SpectralSelection:
    """Read selected baseline-pair rows and polarization columns without averaging.

    UVPSpec baseline-pair codes concatenate two six-digit baseline codes.
    Row order and all physical time coordinates are preserved.
    """

    row_coordinates = {"blpair_array", "time_1_array", "time_2_array", "time_avg_array",
                       "lst_1_array", "lst_2_array", "lst_avg_array"}

    def __init__(self, group, baseline_pair_code=None, polarization_code=None):
        self.group = group
        self.rows = slice(None)
        self.baselines = slice(None)
        self.polarizations = slice(None)
        if baseline_pair_code is not None:
            if type(baseline_pair_code) is not int or baseline_pair_code <= 0:
                raise ValueError("positive integer baseline-pair code required")
            codes = group["blpair_array"][()]
            self.rows = np.flatnonzero(codes == baseline_pair_code)
            if not len(self.rows):
                raise ValueError("requested baseline pair absent from spectral product")
            members = np.unique([baseline_pair_code // 1000000, baseline_pair_code % 1000000])
            available = group["bl_array"][()]
            if not np.isin(members, available).all():
                raise ValueError("baseline pair and baseline geometry disagree")
            self.baselines = np.flatnonzero(np.isin(available, members))
        if polarization_code is not None:
            if type(polarization_code) is not int:
                raise ValueError("integer polarization-pair code required")
            self.polarizations = np.flatnonzero(group.attrs["polpair_array"] == polarization_code)
            if len(self.polarizations) != 1:
                raise ValueError("requested polarization absent or ambiguous")

    def attribute(self, name):
        value = self.group.attrs[name]
        if name == "polpair_array":
            return value[self.polarizations]
        if name == "scalar_array" and not isinstance(self.polarizations, slice):
            return value[:, self.polarizations]
        return value

    def dataset(self, name):
        dataset = self.group[name]
        if name in self.row_coordinates:
            return dataset[self.rows]
        if name in ("bl_array", "bl_vecs"):
            return dataset[self.baselines]
        if name.startswith(("data_spw", "stats_P_N_", "nsample_spw", "integration_spw", "wgt_spw")):
            values = dataset[self.rows]
            return values[..., self.polarizations]
        return dataset[()]

    def record(self):
        return {"row_indices": None if isinstance(self.rows, slice) else self.rows.tolist(),
                "baseline_indices": None if isinstance(self.baselines, slice) else self.baselines.tolist(),
                "polarization_indices": None if isinstance(self.polarizations, slice) else self.polarizations.tolist()}
