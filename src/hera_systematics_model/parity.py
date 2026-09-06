"""Structural and numerical parity against retained spectral products."""

import argparse
from pathlib import Path

import numpy as np

from .configuration import file_identity
from .production import write_json_exclusive
from .spectrum_selection import SpectralSelection


def compare_arrays(reference, candidate, rtol=1e-8, atol=0.):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {"passed": False, "reason": "shape mismatch", "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape)}
    finite = np.isfinite(reference)
    same_support = np.array_equal(finite, np.isfinite(candidate))
    if not same_support:
        return {"passed": False, "reason": "finite support mismatch"}
    left, right = reference[finite], candidate[finite]
    if not len(left):
        return {"passed": True, "finite_count": 0, "relative_l2": None, "reason": "no finite values"}
    difference = np.abs(right - left)
    scale = float(np.max(np.abs(left)))
    relative = None
    if scale > 0:
        relative = float(np.linalg.norm((right - left) / scale) / np.linalg.norm(left / scale))
    return {"passed": bool(np.allclose(left, right, rtol=rtol, atol=atol)), "finite_count": int(len(left)),
            "relative_l2": relative, "relative_unavailable_reason": "zero reference norm" if relative is None else None,
            "max_absolute_difference": float(np.max(difference)), "reference_max_absolute": scale}


def exact_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare_power(reference, candidate, noise, rtol=1e-8, atol=0., noise_atol=0.):
    """Report strict parity and an optional absolute tolerance in noise units.

    The additional allowance applies only where the reference noise is finite,
    real and positive. It cannot hide different finite support or coordinates.
    """
    if not np.isfinite(noise_atol) or noise_atol < 0:
        raise ValueError("nonnegative finite noise tolerance required")
    result = compare_arrays(reference, candidate, rtol, atol)
    result["strict_passed"] = result["passed"]
    result["noise_atol"] = noise_atol
    a, b, pn = map(np.asarray, (reference, candidate, noise))
    if "finite_count" not in result or a.shape != pn.shape:
        return {**result, "passed": False, "reason": result.get("reason", "noise shape mismatch")}
    finite = np.isfinite(a)
    usable = finite & np.isfinite(pn) & (pn.imag == 0) & (pn.real > 0)
    allowance = np.where(usable, pn.real, 0.) * noise_atol
    difference = np.abs(b[finite] - a[finite])
    strict_allowance = atol + rtol * np.abs(a[finite])
    result["passed"] = bool(np.all(difference <= strict_allowance + allowance[finite]))
    excess = np.maximum(np.abs(b[usable] - a[usable]) - atol - rtol * np.abs(a[usable]), 0)
    result["maximum_excess_in_noise_units"] = float(np.max(excess / pn.real[usable])) if usable.any() else None
    result["noise_support_cells"] = int(usable.sum())
    result["noise_unavailable_reason"] = None if usable.any() else "no finite positive reference noise"
    return result


def compare_spectra(reference, candidate, group="stokespol/interleave_averaged", rtol=1e-8, atol=0.,
                    baseline_pair_code=None, polarization_code=None, power_noise_atol=0.):
    """Require identical coordinates and compare every declared SPW product."""
    import h5py

    coordinate_attributes = ("spw_array", "polpair_array", "dly_array", "freq_array", "channel_width",
                             "telescope_location", "vis_units", "norm_units", "cosmo", "taper", "norm", "folded")
    coordinate_datasets = ("bl_array", "bl_vecs", "blpair_array", "spw_dly_array", "spw_freq_array",
                            "time_1_array", "time_2_array", "time_avg_array", "lst_1_array", "lst_2_array", "lst_avg_array")
    report = {"reference": file_identity(reference), "candidate": file_identity(candidate), "group": group,
              "rtol": rtol, "atol": atol, "power_noise_atol": power_noise_atol,
              "baseline_pair_code": baseline_pair_code, "polarization_code": polarization_code,
              "coordinates": {}, "spectral_windows": []}
    with h5py.File(reference, "r") as ref, h5py.File(candidate, "r") as new:
        a, b = ref[group], new[group]
        left = SpectralSelection(a, baseline_pair_code, polarization_code)
        right = SpectralSelection(b, baseline_pair_code, polarization_code)
        report["selection"] = {"reference": left.record(), "candidate": right.record()}
        for name in coordinate_attributes:
            report["coordinates"][name] = name in a.attrs and name in b.attrs and exact_equal(left.attribute(name), right.attribute(name))
        for name in coordinate_datasets:
            report["coordinates"][name] = name in a and name in b and exact_equal(left.dataset(name), right.dataset(name))
        spws = list(range(14))
        report["all_requested_spws"] = exact_equal(a.attrs["spw_array"], spws) and exact_equal(b.attrs["spw_array"], spws)
        report["normalization"] = compare_arrays(left.attribute("scalar_array"), right.attribute("scalar_array"), rtol, atol)
        for spw in spws:
            comparisons = {}
            for name in (f"data_spw{spw}", f"stats_P_N_{spw}", f"nsample_spw{spw}",
                         f"integration_spw{spw}", f"wgt_spw{spw}"):
                if name not in a or name not in b:
                    comparisons[name] = {"passed": False, "reason": "missing required spectral product"}
                elif name.startswith("data_spw") and power_noise_atol and f"stats_P_N_{spw}" in a:
                    comparisons[name] = compare_power(left.dataset(name), right.dataset(name),
                        left.dataset(f"stats_P_N_{spw}"), rtol, atol, power_noise_atol)
                else:
                    comparisons[name] = compare_arrays(left.dataset(name), right.dataset(name), rtol, atol)
            report["spectral_windows"].append({"spw": spw, "arrays": comparisons,
                "passed": all(item["passed"] for item in comparisons.values())})
    report["passed"] = (all(report["coordinates"].values()) and report["all_requested_spws"]
                         and report["normalization"]["passed"] and all(w["passed"] for w in report["spectral_windows"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare all spectral windows to retained products")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--group", default="stokespol/interleave_averaged")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--baseline-pair-code", type=int)
    parser.add_argument("--polarization-code", type=int)
    parser.add_argument("--power-noise-atol", type=float, default=0.)
    args = parser.parse_args(argv)
    report = compare_spectra(args.reference, args.candidate, args.group, args.rtol, args.atol,
        args.baseline_pair_code, args.polarization_code, args.power_noise_atol)
    write_json_exclusive(Path(args.output), report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
