"""Validate physical UVPSpec axes before combining spectral products."""

import numpy as np

from .parity import exact_equal
from .spectrum_selection import SpectralSelection


ROW_COORDINATES = SpectralSelection.row_coordinates
BASELINE_DATASETS = {"bl_array", "bl_vecs"}
CONSTANT_DATASETS = {"OmegaP", "OmegaPP", "spw_dly_array", "spw_freq_array"}
LABEL_DATASETS = {"label_1_array", "label_2_array"}
VARIABLE_ATTRIBUTES = {"Nbls", "Nblpairs", "Nbltpairs", "Ntimes", "Ntpairs", "history"}
REQUIRED_ATTRIBUTES = {"spw_array", "polpair_array", "dly_array", "freq_array", "channel_width",
    "telescope_location", "vis_units", "norm_units", "cosmo", "taper", "norm", "folded", "scalar_array",
    "Nbls", "Nblpairs", "Nbltpairs", "Ntimes", "Ntpairs", "Nspws", "Npols", "history"}


def dataset_axis(name, spws):
    """Return the axis carrying physical rows, baselines, or constant metadata."""
    if name in ROW_COORDINATES:
        return "row", 0
    if name in BASELINE_DATASETS:
        return "baseline", 0
    if name in CONSTANT_DATASETS:
        return "constant", None
    if name in LABEL_DATASETS:
        return "row", 1
    for spw in spws:
        if name in {f"{prefix}{spw}" for prefix in
                    ("data_spw", "wgt_spw", "integration_spw", "nsample_spw", "stats_P_N_", "stats_P_SN_")}:
            return "row", 0
    raise ValueError(f"unrecognized spectral dataset: {name}")


def require_equal(left, right, label):
    if not exact_equal(left, right):
        raise ValueError(f"spectral metadata mismatch: {label}")


def derived_scalars_equivalent(left, right):
    """Return whether positive float64 scalars differ by at most two steps."""
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or left.dtype != np.dtype(np.float64)
            or right.dtype != np.dtype(np.float64)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or np.any(left <= 0) or np.any(right <= 0)):
        return False
    lower, upper = np.minimum(left, right), np.maximum(left, right)
    first_step = np.nextafter(lower, upper)
    second_step = np.nextafter(first_step, upper)
    return bool(np.equal(second_step, upper).all())


def inspect_spectrum(group, expected_spws=tuple(range(14))):
    """Reject absent bands, inconsistent axes, and duplicate physical samples.

    Data and noise can contain nonfinite measurements; this structural check
    does not manufacture validity or discard them. Coordinates must be finite.
    """
    if not REQUIRED_ATTRIBUTES <= set(group.attrs):
        raise ValueError("required spectral attributes are absent")
    spws = tuple(int(value) for value in group.attrs["spw_array"])
    if spws != tuple(expected_spws) or len(set(spws)) != len(spws):
        raise ValueError("required spectral windows are absent or reordered")
    required = ROW_COORDINATES | BASELINE_DATASETS | CONSTANT_DATASETS | LABEL_DATASETS
    required |= {f"{prefix}{spw}" for spw in spws for prefix in
                 ("data_spw", "wgt_spw", "integration_spw", "nsample_spw", "stats_P_N_")}
    if not required <= set(group):
        raise ValueError("required spectral datasets are absent")
    axes = {name: dataset_axis(name, spws) for name in group}
    rows = np.asarray(group["blpair_array"])
    baselines = np.asarray(group["bl_array"])
    if (rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu"
            or baselines.ndim != 1 or not len(baselines) or baselines.dtype.kind not in "iu"
            or len(np.unique(baselines)) != len(baselines) or np.any(baselines <= 0)):
        raise ValueError("invalid physical baseline identities")
    if not np.isin(np.concatenate([rows // 1000000, rows % 1000000]), baselines).all():
        raise ValueError("baseline pairs have no matching geometry")
    if not np.array_equal(np.unique(np.concatenate([rows // 1000000, rows % 1000000])), np.sort(baselines)):
        raise ValueError("baseline geometry contains unused identities")
    for name in ROW_COORDINATES:
        values = group[name][()]
        if values.shape != rows.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid physical row coordinates: {name}")
    if group["bl_vecs"].shape != (len(baselines), 3) or not np.isfinite(group["bl_vecs"][()]).all():
        raise ValueError("invalid physical baseline vectors")
    time1, time2 = group["time_1_array"][()], group["time_2_array"][()]
    physical = list(zip(rows.tolist(), time1.tolist(), time2.tolist()))
    if len(set(physical)) != len(rows):
        raise ValueError("duplicate physical spectral samples")
    pols = np.asarray(group.attrs["polpair_array"])
    if pols.ndim != 1 or not len(pols) or len(np.unique(pols)) != len(pols):
        raise ValueError("invalid polarization identities")
    counts = {"Nbls": len(baselines), "Nblpairs": len(np.unique(rows)), "Nbltpairs": len(rows),
              "Ntimes": len(np.unique(np.concatenate([time1, time2]))),
              "Ntpairs": len(set(zip(time1.tolist(), time2.tolist()))), "Nspws": len(spws), "Npols": len(pols)}
    for name, expected in counts.items():
        if group.attrs[name] != expected:
            raise ValueError(f"spectral dimension attribute disagrees: {name}")
    for name in LABEL_DATASETS:
        if group[name].shape != (len(spws), len(rows), len(pols)):
            raise ValueError(f"spectral label dimensions disagree: {name}")
    if np.shape(group.attrs["scalar_array"]) != (len(spws), len(pols)):
        raise ValueError("spectral normalization dimensions disagree")
    for axis in ("dly", "freq"):
        ids, coordinates = group[f"spw_{axis}_array"][()], np.asarray(group.attrs[f"{axis}_array"])
        if ids.ndim != 1 or coordinates.shape != ids.shape or not np.isfinite(coordinates).all() or not np.isin(ids, spws).all():
            raise ValueError(f"invalid spectral {axis} coordinates")
        for spw in spws:
            values = coordinates[ids == spw]
            if not len(values) or np.any(np.diff(values) <= 0):
                raise ValueError(f"spectral {axis} coordinates are empty or unordered")
    for spw in spws:
        ndelay = int(np.count_nonzero(group["spw_dly_array"][()] == spw))
        nfreq = int(np.count_nonzero(group["spw_freq_array"][()] == spw))
        shapes = {f"{prefix}{spw}": (len(rows), ndelay, len(pols))
                  for prefix in ("data_spw", "stats_P_N_", "stats_P_SN_")}
        shapes[f"wgt_spw{spw}"] = (len(rows), nfreq, 2, len(pols))
        shapes.update({f"{prefix}{spw}": (len(rows), len(pols)) for prefix in ("nsample_spw", "integration_spw")})
        for name, shape in shapes.items():
            if name in group and group[name].shape != shape:
                raise ValueError(f"spectral payload dimensions disagree: {name}")
    return {"spws": spws, "axes": axes, "baselines": baselines, "rows": rows,
            "physical_samples": physical, "counts": counts}


def _require_compatible(reference, candidate, allow_scalar_roundoff):
    if set(reference.attrs) != set(candidate.attrs) or set(reference) != set(candidate):
        raise ValueError("spectral schema differs across inputs")
    for name in set(reference.attrs) - VARIABLE_ATTRIBUTES:
        if name == "scalar_array" and allow_scalar_roundoff:
            if not derived_scalars_equivalent(reference.attrs[name], candidate.attrs[name]):
                raise ValueError("spectral metadata mismatch: scalar_array")
        else:
            require_equal(reference.attrs[name], candidate.attrs[name], name)
    for name in CONSTANT_DATASETS:
        require_equal(reference[name][()], candidate[name][()], name)
    for name in reference:
        if reference[name].dtype != candidate[name].dtype:
            raise ValueError(f"spectral dataset type differs: {name}")


def require_compatible(reference, candidate):
    """Require exact numerical conventions and ordered spectral coordinates."""
    _require_compatible(reference, candidate, allow_scalar_roundoff=False)


def require_merge_compatible(reference, candidate):
    """Allow only two-step roundoff in the derived scalar metadata."""
    _require_compatible(reference, candidate, allow_scalar_roundoff=True)
