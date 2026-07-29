#!/usr/bin/env python
"""Self-checking exercise of residuals.py on synthetic matched pairs.

Run directly:  python check_residuals.py

The synthetic pairs reproduce the two properties of the real cylindrical
spectra that break naive transforms: a dynamic range of many orders of
magnitude between the lowest delay bin and the rest of the plane, and
noise-dominated cells whose cross-power is negative.
"""
import sys

import numpy as np

from residuals import INVERSES, TRANSFORMS, apply_transform

RNG = np.random.default_rng(20260726)
FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def synthetic_pair(n_samples=40, n_groups=12, n_dly=20, negatives=True,
                   amp=1e12):
    """Matched ideal/systematic pair with a realistic delay dynamic range.

    Power falls steeply with delay, so the first bin dominates exactly as it
    does in the merged products. The systematic adds one fixed contamination
    shape with a per-sample amplitude; `amp` sets whether the perturbation is
    large or small compared with the local power.
    """
    delay_falloff = 10.0 ** (-0.6 * np.arange(n_dly))
    base = (1e14 * delay_falloff)[None, :] * (1.0 + 0.1 * RNG.standard_normal(
        (n_groups, 1)))
    ideal = np.repeat(base[None], n_samples, axis=0)
    ideal *= 1.0 + 0.02 * RNG.standard_normal(ideal.shape)

    mode = np.zeros((n_groups, n_dly))
    mode[:, 2:8] = np.outer(np.linspace(1.0, 0.2, n_groups), np.ones(6))
    mode /= np.linalg.norm(mode)
    amps = amp * RNG.standard_normal(n_samples)
    sys_ = ideal + amps[:, None, None] * mode[None]

    if negatives:
        # noise floor at the high-delay end drives some cells negative
        noise = 1e9 * RNG.standard_normal(sys_.shape)
        sys_[:, :, n_dly // 2:] += noise[:, :, n_dly // 2:]
        ideal[:, :, n_dly // 2:] += noise[:, :, n_dly // 2:] * 0.5
    pn = np.full(sys_.shape, 1e9)
    return ideal, sys_, mode, amps, pn


def bin0_variance_share(residual):
    """Fraction of feature variance sitting in the first delay bin."""
    r = np.where(np.isfinite(residual), residual, 0.0)
    var = r.reshape(r.shape[0], -1).var(axis=0).reshape(r.shape[1:])
    total = var.sum()
    return float(var[:, 0].sum() / total) if total > 0 else float("nan")


def main():
    ideal, sys_, mode, amps, pn = synthetic_pair()

    print("linear")
    lin = TRANSFORMS["linear"](sys_, ideal)
    check("recovers the injected difference exactly",
          np.allclose(lin, sys_ - ideal, rtol=0, atol=0))
    check("leaves no undefined cells", np.all(np.isfinite(lin)))

    print("log_ratio on data containing negative cross-powers")
    n_neg = int(np.sum((sys_ <= 0) | (ideal <= 0)))
    check("the synthetic pair really does contain non-positive cells",
          n_neg > 0, f"{n_neg} cells")
    lr = TRANSFORMS["log_ratio"](sys_, ideal, floor=0.0)
    check("returns NaN rather than raising or inventing a value",
          np.isnan(lr).any() and not np.isnan(lr).all())
    check("every non-positive cell is marked undefined",
          np.all(np.isnan(lr[(sys_ <= 0) | (ideal <= 0)])))
    check("every positive cell is defined",
          np.all(np.isfinite(lr[(sys_ > 0) & (ideal > 0)])))
    lr_floor = TRANSFORMS["log_ratio"](sys_, ideal, floor=1e10)
    check("a floor rescues cells the unfloored transform rejected",
          np.isfinite(lr_floor).sum() > np.isfinite(lr).sum(),
          f"{int(np.isfinite(lr).sum())} -> {int(np.isfinite(lr_floor).sum())}")

    print("signed_asinh")
    sa = TRANSFORMS["signed_asinh"](sys_, ideal, pn=pn)
    check("defined everywhere despite the negative cells",
          np.all(np.isfinite(sa)))
    check("preserves the sign of the linear residual",
          np.all(np.sign(sa[lin != 0]) == np.sign(lin[lin != 0])))
    check("is monotonic in the systematic power", _monotonic_in_p_sys())

    # Dynamic-range compression has to be measured on the raw field. In a
    # residual the bright ideal has already cancelled, so both linear and asinh
    # residuals are free of the first-bin dominance and the comparison is
    # vacuous. On raw power the first delay bin carries almost all the variance,
    # which is what makes a PCA of raw spectra describe that one bin.
    scale_raw = float(np.median(np.abs(ideal)))
    share_raw = bin0_variance_share(sys_)
    share_raw_sa = bin0_variance_share(np.arcsinh(sys_ / scale_raw))
    check("first delay bin dominates the raw field",
          share_raw > 0.5, f"share={share_raw:.4f}")
    check("asinh removes the first-bin dominance of the raw field",
          share_raw_sa < 0.1 * share_raw,
          f"raw {share_raw:.4f} -> asinh {share_raw_sa:.4f}")

    print("noise_weighted")
    nw = TRANSFORMS["noise_weighted"](sys_, ideal, pn=pn)
    check("equals the linear residual divided by the noise",
          np.allclose(nw, (sys_ - ideal) / pn))
    bad_pn = pn.copy()
    bad_pn[0, 0, 0] = np.inf
    bad_pn[0, 0, 1] = 0.0
    nw_bad = TRANSFORMS["noise_weighted"](sys_, ideal, pn=bad_pn)
    check("marks cells with infinite or zero noise undefined",
          np.isnan(nw_bad[0, 0, 0]) and np.isnan(nw_bad[0, 0, 1]))
    try:
        TRANSFORMS["noise_weighted"](sys_, ideal, pn=None)
        check("refuses to run without noise", False)
    except ValueError:
        check("refuses to run without noise", True)

    print("injected-mode recovery")
    unit_mode = mode.ravel() / np.linalg.norm(mode)

    ov_lin = _leading_overlap(TRANSFORMS["linear"](sys_, ideal), unit_mode)
    check("linear: leading component recovers the injected shape exactly",
          ov_lin > 0.95, f"|overlap|={ov_lin:.4f}")

    # asinh is a nonlinear reweighting, not a shape-preserving map. To first
    # order d asinh(P/s) = dP / sqrt(s^2 + P^2), so the injected shape is
    # multiplied by a factor that varies across the plane. In the small-signal
    # regime that factor is exactly predictable and can be divided out; the
    # basis learned in asinh space is therefore a reweighted contamination
    # shape rather than the physical one.
    # negatives=False so the pair carries only the injected mode; the noise
    # mismatch used elsewhere is orders of magnitude above a small injection
    # and would hide it entirely.
    s_ideal, s_sys, s_mode, _, s_pn = synthetic_pair(amp=1e6, negatives=False)
    s_lin = TRANSFORMS["linear"](s_sys, s_ideal)
    s_sa = TRANSFORMS["signed_asinh"](s_sys, s_ideal, pn=s_pn)
    weight = 1.0 / np.sqrt(s_pn ** 2 + s_ideal ** 2)
    good = np.isfinite(s_sa) & (np.abs(s_lin * weight) > 0)
    rel = np.abs((s_sa[good] - (s_lin * weight)[good]) / (s_lin * weight)[good])
    check("small-signal asinh matches dP/sqrt(s^2+P^2)",
          float(np.median(rel)) < 1e-5, f"median rel dev={np.median(rel):.2e}")

    s_unit = s_mode.ravel() / np.linalg.norm(s_mode)
    ov_sa = _leading_overlap(s_sa, s_unit)
    ov_unweighted = _leading_overlap(s_sa / weight, s_unit)
    check("asinh reweights the shape rather than preserving it",
          ov_sa < 0.9, f"|overlap|={ov_sa:.4f} (linear gives {ov_lin:.4f})")
    check("dividing the predicted weight out restores the injected shape",
          ov_unweighted > 0.99 and ov_unweighted > ov_sa,
          f"asinh {ov_sa:.4f} -> unweighted {ov_unweighted:.4f}")

    print("round trip through each inverse")
    for name in TRANSFORMS:
        kw = {"pn": pn} if name in ("noise_weighted", "signed_asinh") else {}
        if name == "log_ratio":
            kw = {"floor": 1e12}
        fwd = TRANSFORMS[name](sys_, ideal, **kw)
        back = INVERSES[name](fwd, ideal, **kw)
        both = np.isfinite(fwd) & np.isfinite(back)
        check(f"{name}: inverse recovers P_sys where defined",
              np.allclose(back[both], sys_[both], rtol=1e-8, atol=0),
              f"max rel dev="
              f"{np.max(np.abs((back[both] - sys_[both]) / sys_[both])):.2e}")

    print("apply_transform reporting")
    _, info = apply_transform("log_ratio", sys_, ideal, floor=0.0)
    check("counts undefined cells",
          info["n_undefined"] == int(np.isnan(lr).sum()),
          f"{info['n_undefined']} undefined of {info['n_cells']}")
    try:
        apply_transform("does_not_exist", sys_, ideal)
        check("rejects an unknown representation", False)
    except KeyError:
        check("rejects an unknown representation", True)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print(f"all {len(TRANSFORMS)} representations checked, no failures")
    return 0


def _leading_overlap(residual, unit_mode):
    """|<first principal component, injected mode>| for a residual stack."""
    flat = np.where(np.isfinite(residual), residual, 0.0)
    flat = flat.reshape(flat.shape[0], -1)
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    return abs(float(np.dot(vt[0], unit_mode)))


def _monotonic_in_p_sys():
    """asinh residual must increase with P_sys at fixed P_ideal and scale."""
    grid = np.array([-1e12, -1e6, 0.0, 1e6, 1e12, 1e18])
    out = TRANSFORMS["signed_asinh"](grid, np.zeros_like(grid),
                                     pn=np.full(grid.shape, 1e9))
    return bool(np.all(np.diff(out) > 0))


if __name__ == "__main__":
    sys.exit(main())
