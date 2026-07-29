#!/usr/bin/env python
"""Cylindrical (kperp, kpara) coordinates and the horizon wedge.

Shared by the plotting and validation scripts so both describe the same
geometry. No plotting dependencies.

Sample files written by build_aligned_samples.py always carry kparas; kperps is
only populated when the run could read a cosmology off the UVPSpec object.
`coords_from_npz` prefers the stored values and falls back to deriving them,
which is possible because kpara/delay is a constant per spectral window:

    kpara = 2 pi tau nu21 H(z) / (c (1+z)^2)

so inverting that constant gives the window's redshift, and then

    kperp = 2 pi |b| nu / (c D_M(z)).
"""
import numpy as np

import astropy.units as u
from astropy.cosmology import Planck15
from scipy.optimize import brentq

NU21_HZ = 1420405751.7667
C_M_S = 299792458.0


def redshift_from_kpara_ratio(dlys, kparas, cosmo=Planck15):
    """Invert the kpara/delay constant for the window's redshift.

    Returns (z, ratio, is_constant); z is NaN when it cannot be solved.
    """
    dlys = np.asarray(dlys, float)
    kparas = np.asarray(kparas, float)
    ok = np.isfinite(dlys) & np.isfinite(kparas) & (dlys != 0)
    if not np.any(ok):
        return np.nan, np.nan, False
    ratio = kparas[ok] / dlys[ok]
    is_constant = bool(np.allclose(ratio, ratio[0], rtol=1e-6))
    target = float(np.median(ratio))

    def residual(z):
        h_per_s = cosmo.H(z).to(u.s**-1).value / cosmo.h
        return 2 * np.pi * NU21_HZ * h_per_s / (C_M_S * (1 + z) ** 2) \
            * u.Mpc.to(u.m) - target

    try:
        z = brentq(residual, 0.5, 60.0)
    except ValueError:
        return np.nan, target, is_constant
    return z, target, is_constant


def cylindrical_coords(z, blp_lens, cosmo=Planck15):
    """kperp [h/Mpc] per baseline length, and the horizon wedge slope.

    D_M enters kperp in Mpc/h so kperp comes out in h/Mpc, and enters the slope
    in plain Mpc because the slope relates two h/Mpc quantities and is
    therefore dimensionless.
    """
    d_m_mpc = cosmo.comoving_transverse_distance(z).to(u.Mpc).value
    d_m_h = d_m_mpc * cosmo.h
    nu_obs = NU21_HZ / (1 + z)
    kperp = 2 * np.pi * np.asarray(blp_lens, float) * nu_obs / (C_M_S * d_m_h)
    h_km_s_mpc = cosmo.H(z).to(u.km / u.s / u.Mpc).value
    slope = h_km_s_mpc * d_m_mpc / ((C_M_S / 1e3) * (1 + z))
    return kperp, slope


def coords_from_npz(z_npz, cosmo=Planck15):
    """Coordinates for one stored spectral window.

    Returns a dict with redshift, kperp, kpara, wedge slope, the kpara/delay
    constant, and which source kperp came from.
    """
    kparas = np.asarray(z_npz["kparas"], float)
    dlys = np.asarray(z_npz["dlys"], float)
    zred, ratio, is_constant = redshift_from_kpara_ratio(dlys, kparas, cosmo)
    if not np.isfinite(zred):
        raise ValueError("cannot determine redshift from the stored kparas")
    kperp_derived, slope = cylindrical_coords(zred, z_npz["blp_lens"], cosmo)

    stored = np.asarray(z_npz["kperps"], float)
    if np.any(np.isfinite(stored)):
        kperp, source = stored, "stored"
    else:
        kperp, source = kperp_derived, "derived"
    return {"redshift": zred, "kperp": kperp, "kpara": kparas, "slope": slope,
            "kpara_per_delay": ratio, "kpara_over_delay_constant": is_constant,
            "kperp_source": source, "kperp_derived": kperp_derived}


def wedge_mask(kperp, kpara, slope, kpara_per_delay=None, buffer_ns=0.0):
    """Boolean (n_kperp, n_kpara): True strictly above the horizon.

    The buffer is a constant kpara offset, since a delay floor is a delay --
    it shifts the line rather than tilting it.
    """
    offset = 0.0
    if buffer_ns and kpara_per_delay is not None:
        offset = buffer_ns * 1e-9 * kpara_per_delay
    return np.asarray(kpara)[None, :] > (slope * np.asarray(kperp)[:, None] + offset)
