#!/usr/bin/env python
"""Create explicit row-exclusion copies of historical aligned sample files.

This descriptive utility preserves the original physical times and records
all removed source rows. It requires explicit validity and never identifies
missing data from a zero power value. Amplitude-based row exclusions are
optional sensitivity products; they do not implement training-only selection
or nested predictive validation.

Multiple branches must contain every requested spectral window and share exact
physical time grids. The union of explicit exclusions is applied to all of them.
An existing output directory or an already trimmed input is rejected.
"""
import argparse
import glob
import json
import os
import re
import warnings

import numpy as np


ROW_KEYS = ("matrix", "pn_eff", "valid", "time_grid", "lst_grid",
            "window_counts")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", action="append", required=True,
                   metavar="DIR:LABEL",
                   help="aligned-sample directory and its label, repeatable; "
                        "rows to drop are unioned over all inputs and the "
                        "union is applied to every input")
    p.add_argument("--outdir", required=True,
                   help="one subdirectory per label is created here")
    p.add_argument("--drop-rows", default="",
                   help="comma-separated row indices to drop in every spw")
    p.add_argument("--drop-empty", action="store_true",
                   help="drop rows with no data inside the mask")
    p.add_argument("--max-noise-inflation", type=float, default=0.0,
                   help="if > 0, drop rows whose median in-mask cell noise "
                        "exceeds this multiple of the per-feature median "
                        "noise")
    p.add_argument("--mask", choices=("none", "above-wedge"),
                   default="above-wedge",
                   help="cell mask used by --drop-empty and "
                        "--max-noise-inflation; 'above-wedge' matches the "
                        "region the above-wedge PCA uses")
    p.add_argument("--wedge-buffer-ns", type=float, default=500.0)
    p.add_argument("--spws", default="", help="comma-separated subset (default: all)")
    return p.parse_args()


def spw_files(d, label):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f"{label}.aligned.spw*.npz"))):
        mm = re.search(r"spw(\d+)\.npz$", fn)
        if mm:
            out[int(mm.group(1))] = fn
    return out


def cell_mask(z, kind, buffer_ns):
    """Flat boolean feature mask for the empty/inflation tests."""
    n_feat = int(z["cube_shape"][1]) * int(z["cube_shape"][2])
    if kind == "none":
        return np.ones(n_feat, dtype=bool)
    from cylindrical import coords_from_npz, wedge_mask

    coords = coords_from_npz(z)
    geom = wedge_mask(coords["kperp"], coords["kpara"], coords["slope"],
                      kpara_per_delay=coords["kpara_per_delay"],
                      buffer_ns=buffer_ns)
    return geom.ravel()


def measured_cells(z, mask):
    """Use explicit validity, finite power and finite positive noise."""
    if "valid" not in z or "pn_eff" not in z:
        raise ValueError("explicit validity and noise required for row exclusion")
    power, valid, pn = (np.asarray(z[key]) for key in ("matrix", "valid", "pn_eff"))
    if (power.ndim != 2 or valid.dtype.kind != "b" or valid.shape != power.shape
            or pn.shape != power.shape):
        raise ValueError("aligned power, validity and noise axes disagree")
    return (valid & np.isfinite(power) & np.isfinite(pn) & (pn > 0))[:, mask]


def empty_rows(z, mask):
    """Rows without measured cells; a valid all-zero row is retained."""
    return np.flatnonzero(~measured_cells(z, mask).any(axis=1))


def inflated_rows(z, mask, threshold):
    """Rows whose median in-mask noise exceeds threshold x the feature median."""
    pn = np.asarray(z["pn_eff"], float)[:, mask]
    good = measured_cells(z, mask)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanmedian(np.where(good, pn, np.nan), axis=0)
        inflation = np.nanmedian(np.where(good, pn, np.nan) / scale, axis=1)
    return np.where(np.isfinite(inflation) & (inflation > threshold))[0], inflation


def unwrapped_lst_hours(lst_rad):
    """LST in hours, unwrapped past a single 24 -> 0 crossing."""
    lst = np.asarray(lst_rad, float) * 12.0 / np.pi
    out = lst.copy()
    for i in range(1, len(out)):
        while out[i] < out[i - 1] - 12.0:
            out[i] += 24.0
    return out


def main():
    args = parse_args()
    inputs = []
    for spec in args.samples:
        d, _, label = spec.rpartition(":")
        if not d or not label:
            raise SystemExit(f"--samples wants DIR:LABEL, got {spec!r}")
        files = spw_files(d, label)
        if not files:
            raise SystemExit(f"no aligned NPZ for label {label} in {d}")
        inputs.append((d, label, files))
    labels = [label for _, label, _ in inputs]
    if len(set(labels)) != len(labels):
        raise SystemExit("every --samples entry needs its own label; got "
                         + ", ".join(labels))
    explicit = sorted({int(s) for s in args.drop_rows.split(",") if s.strip()})

    requested = {int(s) for s in args.spws.split(",")} if args.spws else set.union(*(set(f) for _, _, f in inputs))
    if not requested or any(not requested <= set(files) for _, _, files in inputs):
        raise ValueError("every branch must contain every requested spectral window")
    spws = sorted(requested)
    if not np.isfinite(args.max_noise_inflation) or args.max_noise_inflation < 0:
        raise ValueError("nonnegative finite noise threshold required")
    os.makedirs(args.outdir, exist_ok=False)
    provenance = {
        "inputs": [{"dir": d, "label": label} for d, label, _ in inputs],
        "rules": {
            "drop_rows": explicit,
            "drop_empty": bool(args.drop_empty),
            "max_noise_inflation": args.max_noise_inflation or None,
            "mask": args.mask,
            "wedge_buffer_ns": args.wedge_buffer_ns,
        },
        "row_union": "rows flagged by any rule in any input branch are "
                     "dropped from every branch, so all outputs keep "
                     "identical time grids",
        "spws": [],
    }

    for spw in spws:
        reasons = {}   # original row index -> list of "rule (label)" strings

        def flag(rows, why):
            for r in rows:
                reasons.setdefault(int(r), []).append(why)

        # load every branch first, and check that dropping by row index is
        # even meaningful: a row index only names the same time window in
        # every branch when the branches share one time grid. An input that
        # was already trimmed is refused, because a second trim would record
        # its dropped rows in the wrong (already-shifted) row space.
        loaded = {}
        for d, label, files in inputs:
            z = np.load(files[spw])
            if "trimmed_rows" in z.files:
                raise SystemExit(f"spw {spw} ({label}): input is already a "
                                 f"trimmed file; trim from the original "
                                 f"samples so all dropped rows are recorded "
                                 f"in one row space")
            loaded[label] = z
        first = next(iter(loaded.values()))
        times = first["time_grid"]
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("finite monotonic physical time grid required")
        for label, z in loaded.items():
            if not np.array_equal(z["time_grid"], first["time_grid"]):
                raise SystemExit(f"spw {spw}: branch {label!r} is on a "
                                 f"different time grid than the first "
                                 f"input; rows cannot be dropped by shared "
                                 f"index across branches with different "
                                 f"grids")

        flag(explicit, "explicit drop-rows")
        need_mask = args.drop_empty or args.max_noise_inflation > 0
        for d, label, files in inputs:
            z = loaded[label]
            if not need_mask:
                continue
            mask = cell_mask(z, args.mask, args.wedge_buffer_ns)
            if not mask.any():
                raise SystemExit(f"spw {spw} ({label}): the cell mask keeps "
                                 f"no cells, so the empty/inflation rules "
                                 f"would flag every row; check --mask and "
                                 f"--wedge-buffer-ns")
            if args.drop_empty:
                flag(empty_rows(z, mask), f"empty in mask ({label})")
            if args.max_noise_inflation > 0:
                rows, _ = inflated_rows(z, mask, args.max_noise_inflation)
                flag(rows, f"noise inflation > "
                           f"{args.max_noise_inflation:g}x ({label})")

        drop = np.array(sorted(reasons), dtype=int)
        n_rows = int(first["cube_shape"][0])
        if drop.size and (drop.min() < 0 or drop.max() >= n_rows):
            raise SystemExit(f"spw {spw}: drop rows outside 0..{n_rows - 1}")
        keep = np.ones(n_rows, dtype=bool)
        keep[drop] = False
        if not keep.any():
            raise SystemExit(f"spw {spw}: the rules drop every row")
        lst_hours = unwrapped_lst_hours(first["lst_grid"])

        for d, label, files in inputs:
            z = loaded[label]
            out = {}
            for key in z.files:
                out[key] = z[key][keep] if key in ROW_KEYS else z[key]
            cs = np.array(z["cube_shape"], dtype=int).copy()
            cs[0] = int(keep.sum())
            out["cube_shape"] = cs
            out["trimmed_rows"] = drop
            out["source_rows"] = np.flatnonzero(keep)
            odir = os.path.join(args.outdir, label)
            os.makedirs(odir, exist_ok=True)
            np.savez_compressed(
                os.path.join(odir, f"{label}.aligned.spw{spw:02d}.npz"), **out)

        provenance["spws"].append({
            "spw": spw,
            "n_rows_before": n_rows,
            "n_rows_after": int(keep.sum()),
            "dropped": [{
                "row": r,
                "lst_hours": round(float(lst_hours[r]), 2),
                "time_jd": float(first["time_grid"][r]),
                "reasons": reasons[r],
            } for r in sorted(reasons)],
        })
        print(f"spw {spw:2d}: dropped {drop.size} of {n_rows} rows "
              f"({', '.join(str(r) for r in drop) if drop.size else 'none'})",
              flush=True)

    fn = os.path.join(args.outdir, "trim_provenance.json")
    with open(fn, "w") as fh:
        json.dump(provenance, fh, indent=2, allow_nan=False)
    print(f"wrote {fn} ({len(provenance['spws'])} spws, "
          f"{len(inputs)} branches)")


if __name__ == "__main__":
    main()
