#!/usr/bin/env python
"""Per-spw PCA on LST-aligned sample matrices from build_aligned_samples.py.

Modes:
  single branch:  run_pca_aligned.py --samples-dir DIR --label sum --outdir OUT
  contrast:       add --subtract-dir DIR2 --subtract-label eor-only
                  -> PCA on (branch - subtracted) matched on the common LST
                  grid. This is a labeled first-look contrast, NOT the proper
                  ideal residual (the subtracted branch is not the agreed
                  ideal reference).

Uses plain numpy SVD (no scikit-learn dependency). Saves per-spw NPZ with
mean, top components, scores, explained-variance ratios, and coordinates,
plus one JSON summary across spws.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

TOP_COMPONENTS = 40  # stored, not a truncation choice for science


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--subtract-dir", default="")
    p.add_argument("--subtract-label", default="")
    p.add_argument("--time-tol-sec", type=float, default=60.0)
    p.add_argument("--variance-threshold", type=float, default=0.99)
    p.add_argument("--missing", choices=("zero", "drop-features"), default="zero",
                   help="how to treat cells with no contributing baseline. "
                        "'zero' keeps the historical behaviour (they enter the "
                        "PCA as exact zeros, which asserts zero power rather "
                        "than absence); 'drop-features' excludes any feature "
                        "invalid in some row and scatters the components back "
                        "into the full grid afterwards")
    return p.parse_args()


def spw_files(d, label):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f"{label}.aligned.spw*.npz"))):
        mm = re.search(r"spw(\d+)\.npz$", fn)
        if mm:
            out[int(mm.group(1))] = fn
    return out


def match_times(t1, t2, tol_days):
    """Indices (i1, i2) pairing rows of two time grids within tolerance."""
    i1, i2 = [], []
    j = 0
    for i, t in enumerate(t1):
        while j < len(t2) and t2[j] < t - tol_days:
            j += 1
        if j < len(t2) and abs(t2[j] - t) <= tol_days:
            i1.append(i)
            i2.append(j)
            j += 1
    return np.asarray(i1, dtype=int), np.asarray(i2, dtype=int)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tol_days = args.time_tol_sec / 86400.0
    contrast = bool(args.subtract_dir)
    out_label = (f"{args.label}_minus_{args.subtract_label}"
                 if contrast else args.label)

    files = spw_files(args.samples_dir, args.label)
    if not files:
        raise SystemExit(f"no aligned NPZ files for label {args.label} "
                         f"in {args.samples_dir}")
    sub_files = spw_files(args.subtract_dir, args.subtract_label) if contrast else {}

    summary = []
    for spw, fn in files.items():
        z = np.load(fn)
        X = z["matrix"]
        time_grid = z["time_grid"]
        lst_grid = z["lst_grid"]
        note = ""
        if contrast:
            if spw not in sub_files:
                print(f"spw {spw}: no counterpart in subtract dir, skipping")
                continue
            z2 = np.load(sub_files[spw])
            if z2["matrix"].shape[1] != X.shape[1]:
                raise AssertionError(
                    f"spw {spw}: feature dimension mismatch "
                    f"{X.shape[1]} vs {z2['matrix'].shape[1]}")
            i1, i2 = match_times(time_grid, z2["time_grid"], tol_days)
            if len(i1) == 0:
                raise AssertionError(f"spw {spw}: no overlapping times")
            X = X[i1] - z2["matrix"][i2]
            time_grid = time_grid[i1]
            lst_grid = lst_grid[i1]
            note = (f"contrast {args.label}-{args.subtract_label}; "
                    f"{len(i1)} matched times; NOT the proper ideal residual")

        nonfinite = ~np.isfinite(X)
        if np.any(nonfinite):
            print(f"spw {spw}: WARNING {nonfinite.sum()} non-finite cells "
                  f"set to 0 before PCA", flush=True)
            X = np.where(nonfinite, 0.0, X)

        # validity mask, when the sample builder recorded one. In contrast mode
        # a cell is usable only if both branches measured it.
        valid = z["valid"] if "valid" in z.files else None
        if contrast and valid is not None and "valid" in z2.files:
            valid = valid[i1] & z2["valid"][i2]
        elif contrast and valid is not None:
            valid = valid[i1]
        invalid_frac = float(1.0 - valid.mean()) if valid is not None else None

        n_full = X.shape[1]
        feature_mask = np.ones(n_full, dtype=bool)
        if args.missing == "drop-features" and valid is not None:
            feature_mask = valid.all(axis=0)
            if not feature_mask.any():
                raise AssertionError(
                    f"spw {spw}: every feature is invalid in some row")
            X = X[:, feature_mask]
            print(f"spw {spw}: dropped {n_full - int(feature_mask.sum())} of "
                  f"{n_full} features invalid in at least one row", flush=True)

        mean = X.mean(axis=0)
        Xc = X - mean
        # economy SVD; rank <= Ntimes - 1
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S**2
        evr = var / var.sum() if var.sum() > 0 else var
        cum = np.cumsum(evr)
        n99 = int(np.searchsorted(cum, args.variance_threshold) + 1)
        ntop = min(TOP_COMPONENTS, len(S))
        scores = U[:, :ntop] * S[:ntop]

        # scatter back onto the full feature axis so that cube_shape and the
        # coordinate arrays keep describing the stored mean and components
        if feature_mask.all():
            mean_full, comps_full = mean, Vt[:ntop]
        else:
            mean_full = np.zeros(n_full)
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
            time_grid=time_grid, lst_grid=lst_grid,
            cube_shape=z["cube_shape"], blp_lens=z["blp_lens"],
            dlys=z["dlys"], kperps=z["kperps"], kparas=z["kparas"],
            **({"valid": valid} if valid is not None else {}))
        summary.append({
            "spw": spw, "file": os.path.basename(out_fn),
            "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
            "n_features_total": int(n_full),
            "missing_policy": args.missing,
            "invalid_cell_fraction": invalid_frac,
            "n_components_for_threshold": n99,
            "top5_evr": [float(v) for v in evr[:5]],
            "note": note,
        })
        print(f"spw {spw:2d}: {X.shape[0]} x {X.shape[1]}  "
              f"{n99} comps for {args.variance_threshold:.0%}  "
              f"top5 evr {np.round(evr[:5], 4)}"
              + ("" if invalid_frac is None else f"  invalid={invalid_frac:.4f}"),
              flush=True)

    with open(os.path.join(args.outdir, f"{out_label}.pca_summary.json"), "w") as fh:
        json.dump({"label": out_label, "spws": summary}, fh, indent=2)
    print(f"wrote {out_label}.pca_summary.json ({len(summary)} spws)")


if __name__ == "__main__":
    main()
