#!/usr/bin/env python
"""Candidate residual representations for a matched (systematic, ideal) pair.

Every transform maps a pair of cylindrical power spectra to one residual array
of the same shape and marks cells it cannot define with NaN. Nothing here
silently substitutes a value for an undefined one: the choice among these
representations is supposed to be settled by held-out reconstruction, and a
transform that quietly fabricates cells would win that comparison dishonestly.

Cross-power spectra are real but not positive -- noise-dominated cells go
negative routinely. That is why `log_ratio` needs a floor and still refuses
cells the floor cannot rescue, and why `signed_asinh` exists: asinh is defined
and monotonic over the whole real line, so it compresses dynamic range without
a positivity assumption.

The transforms share one signature:

    f(p_sys, p_ideal, pn=None, **params) -> ndarray

`pn` is the effective per-cell noise written by build_aligned_samples.py.
"""
import numpy as np


def _resolve_scale(p_ideal, pn, scale):
    """asinh transition scale, shared by the forward and inverse transforms.

    Defaults to the per-cell noise when available so the transform is linear
    below the noise floor and logarithmic above it.
    """
    if scale is None:
        if pn is not None:
            scale = np.asarray(pn, float)
        else:
            finite = np.abs(p_ideal[np.isfinite(p_ideal)])
            scale = float(np.median(finite)) if finite.size else 1.0
    return np.where(np.isfinite(scale) & (np.asarray(scale) > 0), scale, np.nan)


def linear(p_sys, p_ideal, pn=None, **_):
    """P_sys - P_ideal. Direct excess power, no dynamic-range control."""
    return np.asarray(p_sys, float) - np.asarray(p_ideal, float)


def linear_inverse(residual, p_ideal, pn=None, **_):
    return np.asarray(p_ideal, float) + np.asarray(residual, float)


def signed_asinh(p_sys, p_ideal, pn=None, scale=None, **_):
    """asinh(P_sys/s) - asinh(P_ideal/s).

    Linear for |P| << s and logarithmic for |P| >> s, through zero and into
    negative power without a special case.

    This is a reweighting, not a shape-preserving map: to first order the
    residual is dP / sqrt(s^2 + P^2), so a contamination shape is multiplied by
    a factor that varies across the plane. A basis learned in this space is a
    reweighted contamination shape, not the physical one.
    """
    p_sys = np.asarray(p_sys, float)
    p_ideal = np.asarray(p_ideal, float)
    scale = _resolve_scale(p_ideal, pn, scale)
    return np.arcsinh(p_sys / scale) - np.arcsinh(p_ideal / scale)


def signed_asinh_inverse(residual, p_ideal, pn=None, scale=None, **_):
    p_ideal = np.asarray(p_ideal, float)
    scale = _resolve_scale(p_ideal, pn, scale)
    return scale * np.sinh(np.arcsinh(p_ideal / scale)
                           + np.asarray(residual, float))


def log_ratio(p_sys, p_ideal, pn=None, floor=0.0, **_):
    """log(P_sys + floor) - log(P_ideal + floor).

    Cells where either argument is non-positive after the floor are NaN. They
    are genuinely undefined: taking the log of a negative cross-power is the
    failure this guard exists to prevent.
    """
    p_sys, p_ideal = np.broadcast_arrays(np.asarray(p_sys, float) + floor,
                                         np.asarray(p_ideal, float) + floor)
    ok = (p_sys > 0) & (p_ideal > 0)
    out = np.full(p_sys.shape, np.nan)
    out[ok] = np.log(p_sys[ok]) - np.log(p_ideal[ok])
    return out


def log_ratio_inverse(residual, p_ideal, pn=None, floor=0.0, **_):
    base = np.asarray(p_ideal, float) + floor
    ok = base > 0
    out = np.full(np.asarray(residual, float).shape, np.nan)
    out[ok] = np.exp(np.asarray(residual, float)[ok] + np.log(base[ok])) - floor
    return out


def noise_weighted(p_sys, p_ideal, pn=None, **_):
    """(P_sys - P_ideal) / P_N -- the residual in units of its own noise."""
    if pn is None:
        raise ValueError("noise_weighted needs the effective noise pn")
    pn = np.asarray(pn, float)
    ok = np.isfinite(pn) & (pn > 0)
    diff = np.asarray(p_sys, float) - np.asarray(p_ideal, float)
    return np.where(ok, diff / np.where(ok, pn, 1.0), np.nan)


def noise_weighted_inverse(residual, p_ideal, pn=None, **_):
    if pn is None:
        raise ValueError("noise_weighted needs the effective noise pn")
    pn = np.asarray(pn, float)
    ok = np.isfinite(pn) & (pn > 0)
    return np.where(ok, np.asarray(p_ideal, float)
                    + np.asarray(residual, float) * np.where(ok, pn, 0.0), np.nan)


TRANSFORMS = {
    "linear": linear,
    "signed_asinh": signed_asinh,
    "log_ratio": log_ratio,
    "noise_weighted": noise_weighted,
}

# Inverses matter because reconstruction error is only comparable across
# representations when it is measured in one common space. A log-ratio MSE is
# in log units and a linear MSE is in (mK^2 h^-3 Mpc^3)^2; scoring them against
# each other directly would rank the units, not the representations.
INVERSES = {
    "linear": linear_inverse,
    "signed_asinh": signed_asinh_inverse,
    "log_ratio": log_ratio_inverse,
    "noise_weighted": noise_weighted_inverse,
}


def apply_transform(name, p_sys, p_ideal, pn=None, **params):
    """Run one transform and report how much of the grid it could define."""
    if name not in TRANSFORMS:
        raise KeyError(f"unknown residual representation {name!r}; "
                       f"available: {sorted(TRANSFORMS)}")
    res = TRANSFORMS[name](p_sys, p_ideal, pn=pn, **params)
    undefined = ~np.isfinite(res)
    finite = res[~undefined]
    info = {
        "representation": name,
        "params": {k: (v if np.isscalar(v) else "array") for k, v in params.items()},
        "n_cells": int(res.size),
        "n_undefined": int(undefined.sum()),
        "undefined_fraction": float(undefined.mean()),
        "dynamic_range_orders": (
            float(np.log10(np.abs(finite).max() / np.abs(finite[finite != 0]).min()))
            if finite.size and np.any(finite != 0) else float("nan")),
    }
    return res, info
