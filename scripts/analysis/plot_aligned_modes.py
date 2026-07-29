#!/usr/bin/env python
"""Render PCA modes from run_pca_aligned.py in cylindrical (kperp, kpara) space.

Each stored component is a flattened (n_groups x n_delays) map, so it can be
reshaped back to the cylindrical plane using cube_shape and plotted against
physical coordinates.

Coordinates: kparas is always written by build_aligned_samples.py, but kperps
is only populated on runs that read it from the UVPSpec object. When kperps is
absent or all-NaN it is derived here instead: the ratio kpara/delay is a
constant per spectral window,

    kpara = 2 pi tau nu21 H(z) / (c (1+z)^2),

so inverting that ratio gives the window's redshift, and then

    kperp = 2 pi |b| nu / (c D_M(z)).

The derived redshifts are printed so they can be checked against the stored
kperps whenever a run carries both.

The horizon wedge kpara = [H(z) D_M(z) / (c (1+z))] kperp is overlaid on every
map, together with a 500 ns buffer line matching the inpainting delay floor
used upstream.

Modes are drawn as the pure signed component vector on a symmetric diverging
scale. They are not offset by the mean and not scaled by a singular value.

Usage:
  plot_aligned_modes.py --pca-dir DIR --label sum --outdir FIGDIR
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from cylindrical import NU21_HZ, cylindrical_coords, redshift_from_kpara_ratio

BUFFER_NS = 500.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pca-dir", required=True, help="directory of *.pca.spw*.npz")
    p.add_argument("--label", required=True, help="branch label, e.g. sum")
    p.add_argument("--outdir", required=True)
    p.add_argument("--spws", default="", help="comma-separated subset (default: all)")
    p.add_argument("--n-modes", type=int, default=5)
    p.add_argument("--clip-percentile", type=float, default=99.5,
                   help="symmetric colour limit for mode maps")
    return p.parse_args()


def spw_files(d, label):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f"{label}.pca.spw*.npz"))):
        mm = re.search(r"spw(\d+)\.npz$", fn)
        if mm:
            out[int(mm.group(1))] = fn
    return out


def geometric_edges(x):
    """Cell edges for pcolormesh on a log axis; positivity preserved."""
    x = np.asarray(x, float)
    if len(x) == 1:
        return np.array([x[0] * 0.9, x[0] * 1.1])
    e = np.empty(len(x) + 1)
    e[1:-1] = np.sqrt(x[1:] * x[:-1])
    e[0] = x[0] ** 2 / e[1]
    e[-1] = x[-1] ** 2 / e[-2]
    return e


def draw_wedge(ax, kperp, slope, kpara_per_delay):
    """Horizon line, plus the same line offset by the delay floor.

    The buffer is a constant kpara offset rather than a change of slope: the
    500 ns floor is a delay, and kpara/delay is the constant already recovered
    from the stored coordinates.
    """
    kp = np.array([kperp.min(), kperp.max()])
    buf = BUFFER_NS * 1e-9 * kpara_per_delay
    ax.plot(kp, slope * kp, color="k", lw=1.2, ls="-", label="horizon")
    ax.plot(kp, slope * kp + buf, color="k", lw=1.0, ls="--",
            label=f"horizon + {BUFFER_NS:.0f} ns")


def mesh(ax, kperp, kpara, image, norm=None, cmap="RdBu_r", **kw):
    xe, ye = geometric_edges(kperp), geometric_edges(kpara)
    pc = ax.pcolormesh(xe, ye, image.T, norm=norm, cmap=cmap, shading="flat", **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return pc


def plot_spw(fn, spw, args, report):
    z_npz = np.load(fn)
    cube_shape = z_npz["cube_shape"]
    n_groups, n_dly = int(cube_shape[1]), int(cube_shape[2])
    blp_lens = z_npz["blp_lens"]
    kparas = z_npz["kparas"]
    dlys = z_npz["dlys"]

    zred, ratio, is_const = redshift_from_kpara_ratio(dlys, kparas)
    if not np.isfinite(zred):
        print(f"  spw {spw}: cannot derive redshift, skipping", flush=True)
        return
    kperp_derived, slope = cylindrical_coords(zred, blp_lens)

    stored = z_npz["kperps"]
    stored_ok = np.any(np.isfinite(stored))
    kperp = np.asarray(stored, float) if stored_ok else kperp_derived
    if stored_ok:
        rel = np.nanmax(np.abs(stored - kperp_derived)
                        / np.where(kperp_derived != 0, kperp_derived, np.nan))
        kperp_source = f"stored (max rel. dev. from derived {rel:.2%})"
    else:
        kperp_source = "derived"

    kpara = np.asarray(kparas, float)
    above = float(np.mean(kpara[None, :] > slope * kperp[:, None]))

    mean_map = z_npz["mean"].reshape(n_groups, n_dly)
    comps = z_npz["components"].reshape(-1, n_groups, n_dly)
    evr = z_npz["explained_variance_ratio"]
    scores = z_npz["scores"]
    lst_hours = np.asarray(z_npz["lst_grid"], float) * 12.0 / np.pi

    report.append({
        "spw": spw, "redshift": round(float(zred), 4),
        "nu_center_mhz": round(float(NU21_HZ / (1 + zred) / 1e6), 3),
        "kpara_over_delay_constant": is_const,
        "kperp_source": kperp_source,
        "kperp_min": float(kperp.min()), "kperp_max": float(kperp.max()),
        "kpara_min": float(kpara.min()), "kpara_max": float(kpara.max()),
        "wedge_slope": round(float(slope), 4),
        "fraction_cells_above_wedge": round(above, 4),
        "n_components_for_threshold": int(z_npz["n_for_threshold"][0]),
    })

    n_modes = min(args.n_modes, comps.shape[0])

    # ---- figure A: mean map + leading modes ----
    ncol = 3
    nrow = int(np.ceil((n_modes + 1) / ncol))
    figa, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.2 * nrow),
                              squeeze=False)
    finite_abs = np.abs(mean_map[np.isfinite(mean_map) & (mean_map != 0)])
    linthresh = float(np.percentile(finite_abs, 10)) if finite_abs.size else 1.0
    ax = axes[0][0]
    pc = mesh(ax, kperp, kpara, mean_map,
              norm=SymLogNorm(linthresh=linthresh,
                              vmin=-np.nanmax(np.abs(mean_map)),
                              vmax=np.nanmax(np.abs(mean_map))))
    draw_wedge(ax, kperp, slope, ratio)
    ax.set_title(f"mean  (symlog, linthresh={linthresh:.1e})")
    figa.colorbar(pc, ax=ax, label=r"mK$^2$ $h^{-3}$ Mpc$^3$")
    ax.legend(loc="lower right", fontsize=7)

    for i in range(n_modes):
        ax = axes[(i + 1) // ncol][(i + 1) % ncol]
        v = comps[i]
        lim = float(np.percentile(np.abs(v), args.clip_percentile)) or 1.0
        pc = mesh(ax, kperp, kpara, v, vmin=-lim, vmax=lim)
        draw_wedge(ax, kperp, slope, ratio)
        ax.set_title(f"mode {i}   EVR {evr[i]:.3f}")
        figa.colorbar(pc, ax=ax)

    for r in range(nrow):
        for c in range(ncol):
            a = axes[r][c]
            if not a.has_data():
                a.axis("off")
                continue
            a.set_xlabel(r"$k_\perp$ [$h$ Mpc$^{-1}$]")
            a.set_ylabel(r"$k_\parallel$ [$h$ Mpc$^{-1}$]")

    figa.suptitle(
        f"{args.label}  spw {spw}   z = {zred:.2f}   "
        f"nu = {NU21_HZ / (1 + zred) / 1e6:.1f} MHz   "
        f"wedge slope {slope:.2f}   {above:.0%} of cells above horizon",
        fontsize=11)
    figa.tight_layout(rect=(0, 0, 1, 0.97))
    fna = os.path.join(args.outdir, f"{args.label}.modes.spw{spw:02d}.png")
    figa.savefig(fna, dpi=130)
    plt.close(figa)

    # ---- figure B: scree + scores against LST ----
    figb, (axs, axl) = plt.subplots(1, 2, figsize=(12, 4.2))
    axs.semilogy(np.arange(1, len(evr) + 1), evr, "o-", ms=3)
    axs.axvline(int(z_npz["n_for_threshold"][0]), color="r", ls="--", lw=1,
                label=f"{float(z_npz['variance_threshold'][0]):.0%} threshold")
    axs.set_xlabel("component")
    axs.set_ylabel("explained variance ratio")
    axs.set_title("scree")
    axs.legend(fontsize=8)
    axs.grid(alpha=0.3)

    order = np.argsort(lst_hours)
    for i in range(n_modes):
        s = scores[:, i]
        axl.plot(lst_hours[order], s[order] / (np.abs(s).max() or 1.0),
                 "o-", ms=3, lw=1, label=f"mode {i}")
    axl.set_xlabel("LST [hours]")
    axl.set_ylabel("score (normalised)")
    axl.set_title("scores against LST — smooth trend implies sky, not instrument")
    axl.legend(fontsize=8)
    axl.grid(alpha=0.3)
    figb.suptitle(f"{args.label}  spw {spw}   z = {zred:.2f}", fontsize=11)
    figb.tight_layout(rect=(0, 0, 1, 0.94))
    fnb = os.path.join(args.outdir, f"{args.label}.diagnostics.spw{spw:02d}.png")
    figb.savefig(fnb, dpi=130)
    plt.close(figb)

    # ---- figure C: validity map, only when the run carried one ----
    if "valid" in z_npz.files:
        valid = np.asarray(z_npz["valid"])
        frac = valid.reshape(-1, n_groups, n_dly).mean(axis=0) \
            if valid.ndim > 1 else valid.reshape(n_groups, n_dly).astype(float)
        figc, ax = plt.subplots(figsize=(5.6, 4.4))
        pc = mesh(ax, kperp, kpara, frac, vmin=0, vmax=1, cmap="viridis")
        draw_wedge(ax, kperp, slope, ratio)
        ax.set_xlabel(r"$k_\perp$ [$h$ Mpc$^{-1}$]")
        ax.set_ylabel(r"$k_\parallel$ [$h$ Mpc$^{-1}$]")
        ax.set_title(f"{args.label} spw {spw}: fraction of rows with weight")
        figc.colorbar(pc, ax=ax)
        figc.tight_layout()
        figc.savefig(os.path.join(
            args.outdir, f"{args.label}.validity.spw{spw:02d}.png"), dpi=130)
        plt.close(figc)

    print(f"  spw {spw:2d}: z={zred:6.3f} nu={NU21_HZ / (1 + zred) / 1e6:7.2f}MHz "
          f"kperp[{kperp.min():.4f},{kperp.max():.4f}] slope={slope:.2f} "
          f"above={above:.1%} kperp={kperp_source}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    files = spw_files(args.pca_dir, args.label)
    if not files:
        raise SystemExit(f"no PCA NPZ files for label {args.label} in {args.pca_dir}")
    if args.spws:
        keep = {int(s) for s in args.spws.split(",")}
        files = {k: v for k, v in files.items() if k in keep}

    print(f"[{args.label}] {len(files)} spectral windows -> {args.outdir}", flush=True)
    report = []
    for spw in sorted(files):
        plot_spw(files[spw], spw, args, report)

    zs = [r["redshift"] for r in report]
    monotonic = all(a > b for a, b in zip(zs, zs[1:]))
    out = {"label": args.label, "cosmology": "Planck15",
           "redshift_monotonic_decreasing": monotonic, "spws": report}
    with open(os.path.join(args.outdir, f"{args.label}.coords_report.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[{args.label}] redshift monotonic across spws: {monotonic}")
    print(f"[{args.label}] wrote {args.label}.coords_report.json")


if __name__ == "__main__":
    main()
