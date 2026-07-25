"""Read-only metadata extraction for H6C-IDR2 inventory.

Run remotely with the validation_env Python:
    ssh ... '<validation_env>/bin/python -' < extract_metadata.py
Prints one JSON object per file (JSON lines) to stdout.
Uses h5py only (no full-data reads); pspec files via hera_pspec metadata.
"""
import json
import sys

import h5py
import numpy as np

BASE = "/lustre/aoc/projects/hera/Validation/DataReleases/H6C-IDR2"

FILES = [
    ("chunked_ideal_eor", BASE + "/chunked-ideal-sims/eor-grf-1024/zen.LST.0.0000002.eor-grf-1024.uvh5"),
    ("chunked_ideal_gsm", BASE + "/chunked-ideal-sims/gsm_nside1024/zen.LST.0.0000002.gsm_nside1024.uvh5"),
    ("chunked_ideal_ptsrc", BASE + "/chunked-ideal-sims/ptsrc1024/zen.LST.0.0000002.ptsrc1024.uvh5"),
    ("sim_data_eor", BASE + "/sim_data/eor/zen.LST.0.0000002.eor.uvh5"),
    ("sim_data_diffuse", BASE + "/sim_data/diffuse/zen.LST.0.0000002.diffuse.uvh5"),
    ("sim_data_foregrounds", BASE + "/sim_data/foregrounds/zen.LST.0.0000002.foregrounds.uvh5"),
    ("sim_data_sum", BASE + "/sim_data/sum/zen.LST.0.0000002.sum.uvh5"),
    ("lstbin_LST", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/zen.LST.5.40794.000.sum.uvh5"),
    ("lstbin_STD", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/zen.STD.5.40794.000.sum.uvh5"),
    ("lstbin_HIGHZ", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/zen.HIGHZ.5.40794.000.sum.uvh5"),
    ("lstbin_METRICS", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/zen.LSTBIN-METRICS.5.40794.000.sum.uvh5"),
    ("single_baseline_uvh5", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/single_baseline_files/zen.LST.baseline.0_1.sum.uvh5"),
    ("single_baseline_pspec", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/single_baseline_files/zen.LST.baseline.0_1.sum.pspec.h5"),
    ("mini_dataset", BASE + "/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/mini_dataset/zen.LST.0.53_hours.mini_dataset.sum.uvh5"),
    ("sky_model_eor_chunk", BASE + "/sky_models/Canonical/eor-grf-1024/fch0000.skyh5"),
    ("sky_model_ptsrc_full", BASE + "/sky_models/Canonical/ptsrc1024/full.skyh5"),
]


def _scalar(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def uvh5_meta(path):
    out = {}
    with h5py.File(path, "r") as f:
        if "Header" not in f:
            out["root_keys"] = sorted(f.keys())
            return out
        h = f["Header"]
        for key in ("Ntimes", "Nfreqs", "Nbls", "Nblts", "Npols", "Nants_data",
                    "telescope_name", "instrument", "vis_units", "phase_type"):
            if key in h:
                val = h[key][()]
                out[key] = _scalar(val)
        for key, ds in (("freq_range_hz", "freq_array"),
                        ("time_range_jd", "time_array"),
                        ("lst_range_rad", "lst_array")):
            if ds in h:
                arr = h[ds][()]
                out[key] = [float(np.min(arr)), float(np.max(arr))]
        if "integration_time" in h:
            out["integration_time_s"] = float(np.median(h["integration_time"][()]))
        if "polarization_array" in h:
            out["polarization_array"] = [int(p) for p in h["polarization_array"][()]]
        if "channel_width" in h:
            cw = h["channel_width"][()]
            out["channel_width_hz"] = float(np.median(cw)) if np.ndim(cw) else float(cw)
        if "history" in h:
            hist = _scalar(h["history"][()])
            out["history_head"] = hist[:600]
        if "Data" in f:
            out["data_datasets"] = {k: list(f["Data"][k].shape) for k in f["Data"]}
    return out


def pspec_meta(path):
    import hera_pspec as hp
    out = {}
    psc_meta = {}
    with h5py.File(path, "r") as f:
        out["root_keys"] = sorted(f.keys())
    # hera_pspec PSpecContainer layout: groups of UVPSpec
    psc = hp.container.PSpecContainer(path, mode="r", keep_open=False)
    groups = psc.groups()
    psc_meta["groups"] = groups
    for grp in groups[:1]:
        names = psc.spectra(grp)
        psc_meta["spectra"] = names
        for name in names[:1]:
            uvp = psc.get_pspec(grp, name)
            psc_meta["uvpspec"] = {
                "Nspws": int(uvp.Nspws),
                "Npols": int(uvp.Npols) if hasattr(uvp, "Npols") else None,
                "polpair_array": [int(p) for p in uvp.polpair_array],
                "Nblpairs": int(uvp.Nblpairs),
                "Nbltpairs": int(uvp.Nblpairts) if hasattr(uvp, "Nblpairts") else int(uvp.Nbltpairs),
                "Ntimes": int(uvp.Ntimes),
                "Ndlys_spw0": int(uvp.get_dlys(0).size),
                "spw_freq_ranges_hz": [
                    [float(uvp.freq_array[uvp.spw_to_freq_indices(i)].min()),
                     float(uvp.freq_array[uvp.spw_to_freq_indices(i)].max())]
                    for i in range(uvp.Nspws)
                ],
                "units": str(getattr(uvp, "units", "")),
                "vis_units": str(getattr(uvp, "vis_units", "")),
                "taper": str(getattr(uvp, "taper", "")),
                "norm": str(getattr(uvp, "norm", "")),
                "history_head": str(getattr(uvp, "history", ""))[:600],
            }
    out["pspec"] = psc_meta
    return out


def skyh5_meta(path):
    out = {}
    with h5py.File(path, "r") as f:
        out["root_keys"] = sorted(f.keys())
        h = f.get("Header")
        if h is not None:
            for key in ("Ncomponents", "Nfreqs", "component_type", "nside",
                        "units", "spectral_type"):
                if key in h:
                    out[key] = _scalar(h[key][()])
            if "freq_array" in h:
                arr = h["freq_array"][()]
                out["freq_range_hz"] = [float(np.min(arr)), float(np.max(arr))]
        if "Data" in f:
            out["data_datasets"] = {k: list(f["Data"][k].shape) for k in f["Data"]}
    return out


for label, path in FILES:
    rec = {"label": label, "path": path}
    try:
        if path.endswith(".pspec.h5"):
            rec.update(pspec_meta(path))
        elif path.endswith(".skyh5"):
            rec.update(skyh5_meta(path))
        else:
            rec.update(uvh5_meta(path))
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - inventory must survive odd files
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(rec))
    sys.stdout.flush()
