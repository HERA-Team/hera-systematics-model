#!/usr/bin/env python
"""Build LST/JD-aligned, redundantly averaged cylindrical-PS sample matrices.

Samples are joined on physical time (time_avg_array) rather than on
per-baseline row order.

Alignment model (measured on the merged products): the upstream pipeline
averaged every ~28 integrations (9.66 s) into ~270.59 s windows on a grid of
window EDGES common to all baselines, but per-baseline flagging shifts each
window's time CENTROID by up to +-half a window. Centroid clustering therefore
cannot work; instead each centroid is assigned to its window index
floor((t - anchor)/window), where the window length is inferred from
per-baseline time differences (or forced with --window-sec). Rows that share a
window within one baseline-pair (centroids straddling a window edge) are
discarded; a sample row is retained only when every baseline-pair has exactly
one measurement in that window.

Pipeline per run (one merged PSpecContainer, e.g. baselines_merged.pspec.h5):
  1. load UVPSpec (group/spectrum selectable)
  2. drop all-zero baseline-pairs
  3. build the common time grid across all baseline-pairs (hard assertion)
  4. incoherent redundant-group average (P_N weights) preserving the time axis
  5. fold delay spectra
  6. write one NPZ per spectral window: matrix (Ntimes x Ngroups*Ndly),
     coordinates (blp_lens, dlys, kperp, kpara, lst_grid, time_grid)
     plus a provenance JSON.

Run with the validation_env python on NRAO (hera_pspec 0.4.3.dev89).
"""
import argparse
import json
import os
import sys
import time as time_mod

import numpy as np

import hera_pspec as hp
import hera_pspec.uvpspec_utils as uvputils


def blp_to_int(b):
    """Normalize a blpair (int code or ((a1,a2),(a3,a4)) tuple) to its int code."""
    if isinstance(b, (int, np.integer)):
        return int(b)
    return int(uvputils._antnums_to_blpair(b))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pspec_file", help="path to merged .pspec.h5 (PSpecContainer)")
    p.add_argument("--group", default="stokespol")
    p.add_argument("--spectrum", default="interleave_averaged")
    p.add_argument("--outdir", required=True)
    p.add_argument("--label", required=True, help="branch label, e.g. sum / eor-only")
    p.add_argument("--window-sec", type=float, default=-1.0,
                   help="averaging-window length in seconds; -1 = infer from "
                        "per-baseline time spacing; 0 = single window (fully "
                        "time-averaged inputs)")
    p.add_argument("--polpair", default="pI",
                   help="polarization (both halves of the pair), default pI")
    p.add_argument("--max-blpairs", type=int, default=0,
                   help="if >0, truncate to this many blpairs (smoke tests)")
    p.add_argument("--spws", default="",
                   help="comma-separated spw indices to process (default: all)")
    p.add_argument("--bl-error-tol", type=float, default=1.0)
    return p.parse_args()


class WindowGrid:
    """Assign time centroids to common averaging-window indices."""

    def __init__(self, times, per_blp_times, window_sec):
        uniq = np.unique(times)
        if window_sec == 0 or len(uniq) < 2:
            self.window_days = np.inf
        elif window_sec > 0:
            self.window_days = window_sec / 86400.0
        else:
            diffs = []
            for t in per_blp_times:
                t = np.sort(t)
                if len(t) > 1:
                    diffs.append(np.diff(t))
            if not diffs:
                self.window_days = np.inf
            else:
                self.window_days = float(np.median(np.concatenate(diffs)))
        # anchor half an integration before the earliest centroid, so that a
        # fully-sampled first window sits mid-window and shifted centroids
        # stay inside their window
        gaps = np.diff(uniq) * 86400.0
        small = gaps[gaps < 30.0]
        integ_days = (np.median(small) / 86400.0) if len(small) else 0.0
        self.anchor = uniq[0] - integ_days / 2 - 1e-8
        print(f"  window grid: window="
              f"{'inf' if np.isinf(self.window_days) else f'{self.window_days*86400:.2f}s'}"
              f" integration~{integ_days*86400:.2f}s anchor={self.anchor:.6f}",
              flush=True)

    def assign(self, vals):
        if np.isinf(self.window_days):
            return np.zeros(len(vals), dtype=int)
        return np.floor((np.asarray(vals) - self.anchor)
                        / self.window_days).astype(int)

    def center(self, ids):
        if np.isinf(self.window_days):
            return np.zeros(len(np.atleast_1d(ids)))
        return self.anchor + (np.asarray(ids) + 0.5) * self.window_days


def main():
    args = parse_args()
    t0 = time_mod.time()
    polpair = (args.polpair, args.polpair)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[{args.label}] loading {args.pspec_file}", flush=True)
    psc = hp.container.PSpecContainer(args.pspec_file, mode="r", keep_open=False)
    uvp = psc.get_pspec(args.group, args.spectrum)
    print(f"  Nspws={uvp.Nspws} Nblpairs={uvp.Nblpairs} "
          f"Nbltpairs={uvp.Nbltpairs} Ntimes={uvp.Ntimes}", flush=True)

    if args.spws:
        keep_spws = [int(s) for s in args.spws.split(",")]
        uvp.select(spws=keep_spws, inplace=True)
        print(f"  selected spws {keep_spws}", flush=True)

    blpairs = list(np.unique(uvp.blpair_array))
    if args.max_blpairs > 0:
        blpairs = blpairs[: args.max_blpairs]
        uvp.select(blpairs=blpairs, inplace=True)
        print(f"  truncated to {len(blpairs)} blpairs (smoke test)", flush=True)

    # ---- drop all-zero blpairs (e.g. dead baselines) ----
    spw0 = int(uvp.spw_array[0])
    dead = []
    for blp in blpairs:
        d = uvp.get_data((spw0, blp, polpair))
        if not np.any(d):
            dead.append(blp)
    if dead:
        print(f"  dropping {len(dead)} all-zero blpairs: {dead}", flush=True)
        keep = [b for b in blpairs if b not in set(dead)]
        uvp.select(blpairs=keep, inplace=True)
        blpairs = keep

    # ---- common window grid across blpairs ----
    idx_cache = {blp: np.asarray(uvp.blpair_to_indices(blp)) for blp in blpairs}
    grid = WindowGrid(
        uvp.time_avg_array,
        [uvp.time_avg_array[idx_cache[b]] for b in blpairs[:50]],
        args.window_sec)

    per_blp_ids = {}
    n_dup_rows = 0
    for blp in blpairs:
        inds = idx_cache[blp]
        ids = grid.assign(uvp.time_avg_array[inds])
        uniq, counts = np.unique(ids, return_counts=True)
        dup = set(int(u) for u in uniq[counts > 1])
        if dup:
            keep = np.array([int(i) not in dup for i in ids])
            n_dup_rows += int(np.sum(~keep))
            inds, ids = inds[keep], ids[keep]
            idx_cache[blp] = inds
        per_blp_ids[blp] = ids
    if n_dup_rows:
        print(f"  discarded {n_dup_rows} rows sharing a window within one "
              f"blpair (centroids straddling a window edge)", flush=True)

    common = None
    for blp in blpairs:
        s = set(int(i) for i in per_blp_ids[blp])
        common = s if common is None else (common & s)
    common = np.array(sorted(common), dtype=int)
    print(f"  {len(common)} windows common to all {len(blpairs)} blpairs",
          flush=True)
    if len(common) == 0:
        raise AssertionError("no common windows across baseline-pairs")
    common_pos = {c: i for i, c in enumerate(common)}

    # per-blpair row index into the common grid, in window order
    row_index = {}
    for blp in blpairs:
        inds, ids = idx_cache[blp], per_blp_ids[blp]
        sel = np.array([int(c) in common_pos for c in ids])
        inds, ids = inds[sel], ids[sel]
        order = np.argsort([common_pos[int(c)] for c in ids])
        row_index[blp] = inds[order]
        assert len(row_index[blp]) == len(common), f"blpair {blp} misaligned"

    # sanity: cross-blpair centroid spread inside each window < window length
    stack = np.stack([uvp.time_avg_array[row_index[b]] for b in blpairs])
    spread_sec = (stack.max(axis=0) - stack.min(axis=0)) * 86400
    max_spread = float(spread_sec.max()) if len(common) else 0.0
    print(f"  max cross-blpair centroid spread inside a window: "
          f"{max_spread:.1f}s", flush=True)
    if np.isfinite(grid.window_days) and max_spread > grid.window_days * 86400:
        raise AssertionError("centroid spread exceeds window length")

    time_grid = grid.center(common)
    lst_grid = uvp.lst_avg_array[row_index[blpairs[0]]]

    # ---- redundant length groups ----
    blp_groups, blp_lens, blp_angs, _ = hp.utils.get_blvec_reds(
        uvp, bl_error_tol=args.bl_error_tol, match_bl_lens=True)
    kept = set(int(b) for b in blpairs)
    groups, group_lens = [], []
    for g, glen in zip(blp_groups, blp_lens):
        g2 = [blp_to_int(b) for b in g if blp_to_int(b) in kept]
        if g2:
            groups.append(g2)
            group_lens.append(float(np.atleast_1d(glen)[0]))
    n_groups = len(groups)
    print(f"  {n_groups} redundant length groups after drops", flush=True)

    # ---- explicit P_N-weighted incoherent average + fold, per spw ----
    # (average_spectra/fold_spectra pair rows ordinally, which is exactly the
    # fuzziness this script exists to remove, so the averaging is done here
    # on window-aligned rows instead.)
    polpair_int = uvputils.polpair_tuple2int(polpair)
    ipol = int(np.where(uvp.polpair_array == polpair_int)[0][0])
    have_pn = hasattr(uvp, "stats_array") and "P_N" in getattr(uvp, "stats_array", {})
    if not have_pn:
        print("  WARNING: no P_N stats found; using uniform weights", flush=True)

    manifest = []
    n_bad_weights = 0
    for spw in list(uvp.spw_array):
        spw = int(spw)
        dlys = uvp.get_dlys(spw)
        data = uvp.data_array[spw][:, :, ipol]          # (Nbltpairs, Ndlys)
        if have_pn:
            pn = np.abs(uvp.stats_array["P_N"][spw][:, :, ipol])
            w_all = np.zeros_like(pn)
            good = np.isfinite(pn) & (pn > 0)
            w_all[good] = 1.0 / pn[good] ** 2
        else:
            w_all = np.ones(data.shape, dtype=float)

        cube = np.zeros((len(common), n_groups, len(dlys)))
        for gi, g in enumerate(groups):
            num = np.zeros((len(common), len(dlys)))
            den = np.zeros((len(common), len(dlys)))
            for blp in g:
                rows = row_index[blp]
                d = data[rows, :].real
                finite = np.isfinite(d)
                w = w_all[rows, :] * finite
                num += w * np.where(finite, d, 0.0)
                den += w
            ok = den > 0
            n_bad_weights += int(np.sum(~ok))
            cube[:, gi, :] = np.where(ok, num / np.where(ok, den, 1.0), 0.0)

        # fold: average P(+dly) with P(-dly); keep positive delays
        pos = np.where(dlys > 0)[0]
        pos = pos[np.argsort(dlys[pos])]
        folded = np.zeros((len(common), n_groups, len(pos)))
        for j, ip in enumerate(pos):
            im = np.argmin(np.abs(dlys + dlys[ip]))
            folded[:, :, j] = 0.5 * (cube[:, :, ip] + cube[:, :, im])
        fold_dlys = dlys[pos]

        try:
            kparas = uvp.get_kparas(spw)
            kparas = np.asarray(kparas)[pos]
        except Exception:
            kparas = np.full(len(pos), np.nan)

        fn = os.path.join(args.outdir,
                          f"{args.label}.aligned.spw{spw:02d}.npz")
        np.savez_compressed(
            fn,
            matrix=folded.reshape(len(common), n_groups * len(pos)),
            cube_shape=np.array(folded.shape),
            time_grid=time_grid, lst_grid=lst_grid,
            blp_lens=np.asarray(group_lens), dlys=fold_dlys,
            kperps=np.full(n_groups, np.nan), kparas=kparas,
            group_reps=np.asarray([int(g[0]) for g in groups], dtype=np.int64))
        manifest.append({"spw": spw, "file": os.path.basename(fn),
                         "shape": list(folded.shape)})
        print(f"  wrote {fn} shape={folded.shape}", flush=True)
    if n_bad_weights:
        print(f"  WARNING: {n_bad_weights} (time,delay,group) cells had zero "
              f"total weight and were set to 0", flush=True)

    prov = {
        "input": args.pspec_file,
        "input_mtime": os.path.getmtime(args.pspec_file),
        "input_size": os.path.getsize(args.pspec_file),
        "group": args.group, "spectrum": args.spectrum,
        "polpair": args.polpair,
        "window_sec": (None if np.isinf(grid.window_days)
                       else grid.window_days * 86400.0),
        "max_centroid_spread_sec": max_spread,
        "bl_error_tol": args.bl_error_tol,
        "n_blpairs": len(blpairs), "n_dropped_dead": len(dead),
        "n_dropped_duplicate_window_rows": int(n_dup_rows),
        "n_zero_weight_cells": int(n_bad_weights),
        "weights": "1/P_N^2" if have_pn else "uniform",
        "n_common_times": int(len(common)), "n_groups": n_groups,
        "spws": manifest,
        "hera_pspec_version": hp.__version__,
        "command": " ".join(sys.argv),
        "runtime_sec": time_mod.time() - t0,
    }
    with open(os.path.join(args.outdir, f"{args.label}.provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"[{args.label}] done in {prov['runtime_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
