"""Source-supported interpolation and explicit ideal visibility validity."""

import numpy as np
from scipy.interpolate import interp1d


def periodic_source_order(source_lsts, target_lsts):
    """Place one short physical-time target chunk and its sources across LST zero."""
    source, target = np.asarray(source_lsts, float), np.asarray(target_lsts, float)
    if (source.ndim != 1 or target.ndim != 1 or not len(target)
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("finite source and target LST vectors required")
    unwrapped = np.unwrap(target)
    if np.any(np.diff(unwrapped) <= 0) or np.ptp(unwrapped) >= np.pi:
        raise ValueError("target chunk must follow a short continuous physical-time arc")
    center = unwrapped.mean()
    shifted = center + (source - center + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(shifted, kind="stable")
    if np.any(np.diff(shifted[order]) <= 0):
        raise ValueError("source LST knots must be unique")
    return shifted[order], unwrapped, order


def interpolate_supported(source_lsts, values, source_valid, target_lsts):
    """Use cubic interpolation only on fully supported source curves.

    The cubic spline uses all supplied knots. Any invalid knot therefore makes
    that curve unavailable throughout this chunk; invalid samples are never
    inserted as numerical observations. Extrapolation is always unavailable.
    """
    source_lsts, target_lsts = np.asarray(source_lsts, float), np.asarray(target_lsts, float)
    values, source_valid = np.asarray(values), np.asarray(source_valid)
    if (source_lsts.ndim != 1 or len(source_lsts) < 4 or np.any(np.diff(source_lsts) <= 0)
            or target_lsts.ndim != 1 or not np.isfinite(source_lsts).all()
            or not np.isfinite(target_lsts).all() or values.ndim < 2
            or len(values) != len(source_lsts) or source_valid.shape != values.shape
            or source_valid.dtype.kind != "b" or values.dtype.kind not in "fci"):
        raise ValueError("invalid cubic interpolation inputs")
    shape = (len(target_lsts),) + values.shape[1:]
    output = np.full(shape, np.nan, dtype=np.result_type(values.dtype, np.float64))
    valid = np.zeros(shape, bool)
    supported_curves = (source_valid & np.isfinite(values)).all(axis=0).ravel()
    source = values.reshape(len(source_lsts), -1)
    interpolated = output.reshape(len(target_lsts), -1)
    if supported_curves.any():
        interpolated[:, supported_curves] = interp1d(source_lsts, source[:, supported_curves], kind="cubic",
            axis=0, bounds_error=False, fill_value=np.nan, assume_sorted=True)(target_lsts)
    valid[:] = np.isfinite(output)
    return output, valid


def ideal_arrays(values, supported):
    """Finite supported values, including genuine zeros, get unit counts."""
    values, supported = np.asarray(values), np.asarray(supported)
    if values.shape != supported.shape or supported.dtype.kind != "b" or values.dtype.kind not in "fci":
        raise ValueError("visibility and source support shapes disagree")
    valid = supported & np.isfinite(values)
    return np.where(valid, values, 0), ~valid, valid.astype(float)


def replace_reference_visibilities(reference, values, supported, source_units, source_history, provenance):
    """Preserve reference coordinates and integration metadata while replacing data."""
    if source_units != reference.vis_units:
        raise ValueError("source and reference visibility units differ")
    data, flags, counts = ideal_arrays(values, supported)
    if data.shape != reference.data_array.shape:
        raise ValueError("interpolated data do not match reference axes")
    result = reference.copy()
    result.data_array = data
    result.flag_array = flags
    result.nsample_array = counts
    result.history = (source_history + "\nIdeal sky samples interpolated onto reference coordinates. "
        "Finite source-supported samples have cleared flags and unit counts; unsupported samples are invalid.\n"
        + provenance + "\nReference metadata history, with visibility samples replaced:\n" + reference.history)
    for name in ("time_array", "lst_array", "integration_time", "ant_1_array", "ant_2_array", "freq_array", "polarization_array"):
        if not np.array_equal(getattr(reference, name), getattr(result, name)):
            raise ValueError("reference metadata changed during ideal construction")
    return result
