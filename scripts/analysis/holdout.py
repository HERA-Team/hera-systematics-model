#!/usr/bin/env python
"""Held-out reconstruction comparison of residual representations.

For each spectral window and each candidate representation: transform the
matched pair to residual space, fit a PCA on the training rows only, project
the held-out rows onto the leading k components, invert back to linear power,
and score the reconstruction.

Two choices in here are deliberate and worth stating.

Splitting is by contiguous LST block, never at random. The retained windows
form one unbroken sky track a few hours long, so neighbouring rows are strongly
correlated; a random split would put near-duplicates on both sides and report a
reconstruction error that flatters every representation equally.

Scoring happens in linear power, after inverting the transform. A log-ratio
residual's error is in log units and a linear residual's is in
(mK^2 h^-3 Mpc^3)^2; comparing those directly would rank the units rather than
the representations. The metric is an inverse-P_N-weighted mean squared error
restricted to cells above the horizon wedge, since that is the region inference
will actually use. The basis itself is fitted on the whole plane, so the score
measures what a wedge-dominated basis does to the window it is not describing.

Without --ideal-dir there is no matched ideal branch to subtract, so the
training-set mean stands in as the reference. It is computed from training rows
only. That is a provisional stand-in for a real ideal reference, not a
substitute for one, and the choice is recorded in the output.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from cylindrical import coords_from_npz, wedge_mask
from residuals import INVERSES, TRANSFORMS

DEFAULT_REPRS = "linear,signed_asinh,noise_weighted"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ideal-dir", default="",
                   help="aligned samples of the matched ideal branch; without "
                        "it the training mean is used as a stand-in reference")
    p.add_argument("--ideal-label", default="")
    p.add_argument("--representations", default=DEFAULT_REPRS)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--max-components", type=int, default=20)
    p.add_argument("--wedge-buffer-ns", type=float, default=500.0)
    p.add_argument("--log-floor", type=float, default=0.0,
                   help="floor for the log_ratio representation")
    p.add_argument("--spws", default="")
    return p.parse_args()


def spw_files(d, label):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f"{label}.aligned.spw*.npz"))):
        mm = re.search(r"spw(\d+)\.npz$", fn)
        if mm:
            out[int(mm.group(1))] = fn
    return out


def lst_block_folds(lst, n_folds):
    """Contiguous blocks along the LST track, each held out in turn."""
    order = np.argsort(np.asarray(lst, float))
    return [blk for blk in np.array_split(order, n_folds) if len(blk)]


def weighted_mse(truth, recon, pn, mask):
    """Inverse-variance-weighted MSE over the masked cells."""
    ok = mask & np.isfinite(truth) & np.isfinite(recon) & np.isfinite(pn) & (pn > 0)
    if not np.any(ok):
        return float("nan"), 0
    w = 1.0 / pn[ok] ** 2
    err = (truth[ok] - recon[ok]) ** 2
    return float(np.sum(w * err) / np.sum(w)), int(ok.sum())


def evaluate(p_sys, p_ideal, pn, feature_ok, metric_mask, train, test,
             name, params, max_k):
    """Held-out weighted MSE against component count for one representation."""
    fwd, inv = TRANSFORMS[name], INVERSES[name]
    res = fwd(p_sys, p_ideal, pn=pn, **params)

    usable = feature_ok & np.all(np.isfinite(res), axis=0)
    if not usable.any():
        return None
    flat = res.reshape(res.shape[0], -1)[:, usable.ravel()]

    mu = flat[train].mean(axis=0)
    U, S, Vt = np.linalg.svd(flat[train] - mu, full_matrices=False)
    k_max = min(max_k, Vt.shape[0], len(train) - 1)
    if k_max < 1:
        return None

    # Score only cells the basis actually reconstructed. Features dropped from
    # the fit keep their input values, so leaving them in would compare truth
    # against truth and credit the representation with a perfect cell.
    scored_cells = metric_mask & usable

    curve = []
    centred_test = flat[test] - mu
    for k in range(1, k_max + 1):
        basis = Vt[:k]
        approx = mu + (centred_test @ basis.T) @ basis
        full = np.array(res[test], dtype=float)
        flat_full = full.reshape(len(test), -1)
        flat_full[:, usable.ravel()] = approx
        recon_power = inv(flat_full.reshape(full.shape), p_ideal[test],
                          pn=pn[test], **params)
        mse, n_cells = weighted_mse(p_sys[test], recon_power, pn[test],
                                    scored_cells[None] & np.isfinite(p_sys[test]))
        curve.append({"k": k, "weighted_mse": mse, "n_metric_cells": n_cells})
    return {"representation": name, "params": params,
            "n_features_used": int(usable.sum()),
            "n_features_total": int(usable.size), "curve": curve}


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    reprs = [r.strip() for r in args.representations.split(",") if r.strip()]
    for r in reprs:
        if r not in TRANSFORMS:
            raise SystemExit(f"unknown representation {r!r}; "
                             f"available: {sorted(TRANSFORMS)}")

    files = spw_files(args.samples_dir, args.label)
    if not files:
        raise SystemExit(f"no aligned NPZ for {args.label} in {args.samples_dir}")
    if args.spws:
        keep = {int(s) for s in args.spws.split(",")}
        files = {k: v for k, v in files.items() if k in keep}
    ideal_files = (spw_files(args.ideal_dir, args.ideal_label)
                   if args.ideal_dir else {})
    reference = "matched ideal branch" if ideal_files else "training-set mean"
    print(f"[{args.label}] reference: {reference}; representations: {reprs}",
          flush=True)

    results = []
    for spw in sorted(files):
        z = np.load(files[spw])
        cs = z["cube_shape"]
        n_s, n_g, n_d = int(cs[0]), int(cs[1]), int(cs[2])
        p_sys = z["matrix"].reshape(n_s, n_g, n_d)
        pn = (z["pn_eff"].reshape(n_s, n_g, n_d) if "pn_eff" in z.files
              else np.ones_like(p_sys))
        valid = (z["valid"].reshape(n_s, n_g, n_d) if "valid" in z.files
                 else np.isfinite(p_sys))
        if "pn_eff" not in z.files:
            print(f"  spw {spw}: no pn_eff stored, weighting uniformly "
                  f"(rebuild the samples to score properly)", flush=True)

        coords = coords_from_npz(z)
        metric_mask = wedge_mask(coords["kperp"], coords["kpara"],
                                 coords["slope"],
                                 kpara_per_delay=coords["kpara_per_delay"],
                                 buffer_ns=args.wedge_buffer_ns)
        feature_ok = np.all(valid, axis=0) & np.all(np.isfinite(p_sys), axis=0)

        folds = lst_block_folds(np.asarray(z["lst_grid"], float), args.n_folds)
        per_repr = {r: [] for r in reprs}
        for fi, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n_s), test)
            if len(train) < 3 or len(test) < 1:
                continue
            if ideal_files.get(spw):
                zi = np.load(ideal_files[spw])
                p_ideal = zi["matrix"].reshape(n_s, n_g, n_d)
            else:
                # train rows only: the reference must not see the held-out data
                p_ideal = np.repeat(p_sys[train].mean(axis=0)[None], n_s, axis=0)

            for name in reprs:
                params = {}
                if name == "log_ratio":
                    params["floor"] = args.log_floor
                out = evaluate(p_sys, p_ideal, pn, feature_ok, metric_mask,
                               train, test, name, params, args.max_components)
                if out is not None:
                    out["fold"] = fi
                    per_repr[name].append(out)

        summary = {}
        for name, runs in per_repr.items():
            if not runs:
                summary[name] = None
                continue
            n_k = min(len(r["curve"]) for r in runs)
            mean_curve = [
                {"k": k + 1,
                 "weighted_mse": float(np.nanmean([r["curve"][k]["weighted_mse"]
                                                   for r in runs]))}
                for k in range(n_k)]
            best = min(mean_curve, key=lambda c: (np.inf
                                                  if not np.isfinite(c["weighted_mse"])
                                                  else c["weighted_mse"]))
            summary[name] = {"folds": len(runs),
                             "n_features_used": runs[0]["n_features_used"],
                             "n_features_total": runs[0]["n_features_total"],
                             "mean_curve": mean_curve,
                             "best_k": best["k"],
                             "best_weighted_mse": best["weighted_mse"]}
            print(f"  spw {spw:2d} {name:14s} best k={best['k']:2d} "
                  f"mse={best['weighted_mse']:.4e} "
                  f"({runs[0]['n_features_used']}/{runs[0]['n_features_total']} "
                  f"features)", flush=True)

        results.append({"spw": spw, "redshift": float(coords["redshift"]),
                        "n_samples": n_s, "n_folds": len(folds),
                        "wedge_slope": float(coords["slope"]),
                        "metric_cells_above_wedge": int(metric_mask.sum()),
                        "metric_cells_total": int(metric_mask.size),
                        "representations": summary})

    out = {"label": args.label, "reference": reference,
           "metric": "inverse-P_N-weighted MSE in linear power, above the "
                     f"horizon wedge + {args.wedge_buffer_ns:.0f} ns",
           "split": "contiguous LST blocks",
           "n_folds": args.n_folds, "spws": results}
    fn = os.path.join(args.outdir, f"{args.label}.holdout.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[{args.label}] wrote {os.path.basename(fn)}")


if __name__ == "__main__":
    main()
