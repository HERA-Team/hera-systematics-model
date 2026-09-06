#!/usr/bin/env python
"""Descriptive single-branch PCA of historical aligned sample matrices.

Use versioned paired-sample artifacts with the package fit/evaluate commands
for residual prediction. Historical sample matrices do not establish common
baseline contributors or averaging weights for subtraction between branches.
All stored component counts describe available saved directions; variance
thresholds are descriptive statistics, not estimates of physical dimension.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--subtract-dir", default="")
    p.add_argument("--subtract-label", default="")
    p.add_argument("--variance-threshold", type=float, default=0.99)
    p.add_argument("--missing", choices=("drop-features",), default="drop-features",
                   help="fit only features valid in every input row")
    p.add_argument("--mask", choices=("none", "min-delay", "above-wedge"),
                   default="none",
                   help="restrict the PCA to part of the cylindrical plane. "
                        "'min-delay' drops delay bins below --min-delay-ns; "
                        "'above-wedge' keeps only cells above the horizon plus "
                        "--wedge-buffer-ns. Components are still stored on the "
                        "full grid, zero outside the mask")
    p.add_argument("--min-delay-ns", type=float, default=300.0)
    p.add_argument("--wedge-buffer-ns", type=float, default=500.0)
    p.add_argument("--whiten", choices=("none", "pn-median", "pn-cell"), default="none",
                   help="divide power by no scale, median per-feature noise, or each cell's noise; "
                        "scaling does not identify the physical origin of a component")
    args = p.parse_args()
    if args.subtract_dir or args.subtract_label:
        p.error("branch subtraction requires paired records with common contributors; use hera-systematics pair")
    if not 0 < args.variance_threshold <= 1:
        p.error("variance threshold must be in (0, 1]")
    return args


def spw_files(d, label):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f"{label}.aligned.spw*.npz"))):
        mm = re.search(r"spw(\d+)\.npz$", fn)
        if mm:
            out[int(mm.group(1))] = fn
    return out


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=False)
    out_label = args.label

    files = spw_files(args.samples_dir, args.label)
    if not files:
        raise SystemExit(f"no aligned NPZ files for label {args.label} "
                         f"in {args.samples_dir}")

    summary = []
    for spw, fn in files.items():
        z = np.load(fn)
        X = z["matrix"]
        time_grid = z["time_grid"]
        lst_grid = z["lst_grid"]
        if "valid" not in z or "pn_eff" not in z:
            raise ValueError("explicit validity and noise are required for descriptive PCA")
        valid, pn = np.asarray(z["valid"]), np.asarray(z["pn_eff"])
        if (X.ndim != 2 or min(X.shape) == 0 or X.dtype.kind not in "fiu"
                or valid.dtype.kind != "b" or valid.shape != X.shape or pn.shape != X.shape):
            raise ValueError("power, validity and noise axes disagree")
        if (time_grid.shape != (len(X),) or not np.isfinite(time_grid).all()
                or np.any(np.diff(time_grid) <= 0)):
            raise ValueError("finite monotonic physical times required")
        if (lst_grid.shape != time_grid.shape or not np.isfinite(lst_grid).all()
                or np.any((lst_grid < 0) | (lst_grid >= 2 * np.pi))
                or len(z["cube_shape"]) != 3 or z["cube_shape"][0] != len(X)
                or np.prod(z["cube_shape"][1:]) != X.shape[1]):
            raise ValueError("physical coordinate dimensions disagree")
        valid = valid & np.isfinite(X) & np.isfinite(pn) & (pn > 0)
        invalid_frac = float(1.0 - valid.mean())
        n_full = X.shape[1]
        feature_mask = valid.all(axis=0)

        # Optional geometric support for this descriptive fit.
        if args.mask != "none":
            from cylindrical import coords_from_npz, wedge_mask

            cs = z["cube_shape"]
            n_g, n_d = int(cs[1]), int(cs[2])
            coords = coords_from_npz(z)
            if args.mask == "min-delay":
                geom = np.broadcast_to(
                    (np.asarray(z["dlys"], float) * 1e9 >= args.min_delay_ns),
                    (n_g, n_d))
            else:
                geom = wedge_mask(coords["kperp"], coords["kpara"],
                                  coords["slope"],
                                  kpara_per_delay=coords["kpara_per_delay"],
                                  buffer_ns=args.wedge_buffer_ns)
            print(f"spw {spw}: mask '{args.mask}' keeps "
                  f"{int(geom.sum())} of {geom.size} cells "
                  f"(z={coords['redshift']:.2f})", flush=True)
            feature_mask &= geom.ravel()

        feature_scale = None
        cell_pn = None
        if args.whiten == "pn-median":
            feature_scale = np.median(pn, axis=0)
        elif args.whiten == "pn-cell":
            cell_pn = pn

        if not feature_mask.any():
            raise AssertionError(f"spw {spw}: no features survive the masks")
        if not feature_mask.all():
            X = X[:, feature_mask]
        if feature_scale is not None:
            X = X / feature_scale[feature_mask]
        if cell_pn is not None:
            pnm = cell_pn[:, feature_mask]
            X = X / pnm

        mean = X.mean(axis=0)
        Xc = X - mean
        # economy SVD; rank <= Ntimes - 1
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S**2
        evr = var / var.sum() if var.sum() > 0 else var
        cum = np.cumsum(evr)
        n99 = min(len(S), int(np.searchsorted(cum, args.variance_threshold) + 1)) if var.sum() > 0 else 0
        ntop = len(S)
        scores = U[:, :ntop] * S[:ntop]

        # scatter back onto the full feature axis so that cube_shape and the
        # coordinate arrays keep describing the stored mean and components
        if feature_mask.all():
            mean_full, comps_full = mean, Vt[:ntop]
        else:
            mean_full = np.full(n_full, np.nan)
            mean_full[feature_mask] = mean
            comps_full = np.zeros((ntop, n_full))
            comps_full[:, feature_mask] = Vt[:ntop]

        out_fn = os.path.join(args.outdir, f"{out_label}.pca.spw{spw:02d}.npz")
        np.savez_compressed(
            out_fn,
            mean=mean_full, components=comps_full, singular_values=S,
            explained_variance_ratio=evr, scores=scores,
            n_for_threshold=np.array([n99]),
            variance_threshold=np.array([args.variance_threshold]),
            feature_mask=feature_mask,
            whiten=np.array(args.whiten),
            time_grid=time_grid, lst_grid=lst_grid,
            cube_shape=z["cube_shape"], blp_lens=z["blp_lens"],
            dlys=z["dlys"], kperps=z["kperps"], kparas=z["kparas"],
            valid=valid, pn_eff=pn, source_rows=z["source_rows"] if "source_rows" in z else np.arange(len(time_grid)),
            **({"feature_scale": np.where(feature_mask, feature_scale, np.nan)}
               if feature_scale is not None else {}))
        summary.append({
            "spw": spw, "file": os.path.basename(out_fn),
            "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
            "n_features_total": int(n_full),
            "missing_policy": args.missing,
            "mask": args.mask,
            "whiten": args.whiten,
            "invalid_cell_fraction": invalid_frac,
            "n_components_for_threshold": n99,
            "top5_evr": [float(v) for v in evr[:5]],
            "analysis_type": "single-branch descriptive PCA",
            "input_path": os.path.abspath(fn),
            "saved_components": ntop,
        })
        print(f"spw {spw:2d}: {X.shape[0]} x {X.shape[1]}  "
              f"{n99} comps for {args.variance_threshold:.0%}  "
              f"top5 evr {np.round(evr[:5], 4)}"
              + ("" if invalid_frac is None else f"  invalid={invalid_frac:.4f}"),
              flush=True)

    with open(os.path.join(args.outdir, f"{out_label}.pca_summary.json"), "w") as fh:
        json.dump({"label": out_label, "spws": summary}, fh, indent=2, allow_nan=False)
    print(f"wrote {out_label}.pca_summary.json ({len(summary)} spws)")


if __name__ == "__main__":
    main()
