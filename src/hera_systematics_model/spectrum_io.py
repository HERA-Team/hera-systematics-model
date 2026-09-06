"""Convert HERA spectra to explicit baseline/window records without ordinal joins."""

import ast
from pathlib import Path

import numpy as np

from .configuration import file_identity
from .records import SpectrumRecords
from .statistics import summary


def pair_identity(pair):
    return ":".join(f"{int(a)}_{int(b)}" for a, b in pair)


def spectral_measurements(data, noise, counts, integration, weights, corrupted):
    """Take real cross-power with measured support and real-component noise."""
    data, counts, integration, weights = map(np.asarray, (data, counts, integration, weights))
    if (data.ndim != 2 or counts.shape != (len(data),) or integration.shape != (len(data),)
            or weights.ndim != 3 or weights.shape[0] != len(data) or weights.shape[2] != 2):
        raise ValueError("spectral measurement axes disagree")
    rows = (np.isfinite(counts) & (counts > 0) & np.isfinite(integration) & (integration > 0)
            & np.isfinite(weights).all(axis=(1, 2)) & (weights > 0).any(axis=1).all(axis=1))
    valid = np.isfinite(data) & rows[:, None]
    if noise is None:
        if corrupted:
            raise ValueError("corrupted spectrum lacks required thermal noise")
        pn = np.full(data.shape, np.nan)
    else:
        noise = np.asarray(noise)
        if noise.shape != data.shape or np.any(noise.imag[np.isfinite(noise)] != 0):
            raise ValueError("noise is not a real-component uncertainty")
        pn = noise.real.astype(float)
        if corrupted:
            valid &= np.isfinite(pn) & (pn > 0)
    return data.real.astype(float), pn, valid


def reference_groups(path, group="stokespol", spectrum="interleave_averaged", tolerance_m=1.):
    """Construct the length-group inventory once from corrupted reference geometry."""
    import hera_pspec as hp
    from hera_pspec import uvpspec_utils as utils

    container = hp.PSpecContainer(path, mode="r", keep_open=False)
    uvp = container.get_pspec(group, spectrum, just_meta=True)
    groups, lengths, _, _ = hp.utils.get_blvec_reds(uvp, bl_error_tol=tolerance_m, match_bl_lens=True)
    vectors = {tuple(utils._bl_to_antnums(int(code))): vector for code, vector in zip(uvp.bl_array, uvp.bl_vecs)}
    ordered = sorted(zip(groups, lengths), key=lambda item: (float(np.atleast_1d(item[1])[0]), str(sorted(item[0]))))
    mapping = {}
    for gi, (members, _) in enumerate(ordered):
        for member in members:
            pair = utils._blpair_to_antnums(int(member)) if isinstance(member, (int, np.integer)) else member
            if tuple(pair[0]) != tuple(pair[1]) or pair[0][0] == pair[0][1]:
                raise ValueError("single cross-baseline spectra required")
            vector = np.asarray(vectors[tuple(pair[0])])
            mapping[pair_identity(pair)] = {"group_id": f"length-{gi:03d}", "baseline_pair": [[int(a), int(b)] for a, b in pair],
                "length_m": float(np.linalg.norm(vector)), "vector_m": vector.tolist()}
    return {"input": file_identity(path), "group": group, "spectrum": spectrum,
            "length_tolerance_m": tolerance_m, "baselines": mapping}


def read_records(path, spw, grid, grouping, role, polarization="pI", group="stokespol", spectrum="interleave_averaged"):
    """Read only the requested window and polarization; reject absent products."""
    import h5py
    import hera_pspec as hp
    from hera_pspec import uvpspec_utils as utils

    if role not in ("corrupted", "ideal"):
        raise ValueError("spectral branch role must be explicit")
    with h5py.File(path, "r") as file:
        metadata = file[f"{group}/{spectrum}"].attrs
        if spw not in metadata["spw_array"]:
            raise ValueError("required spectral window is absent")
        cosmology = ast.literal_eval(metadata["cosmo"])
        units = f"{metadata['vis_units']}^2 {metadata['norm_units']}"
    container = hp.PSpecContainer(path, mode="r", keep_open=False)
    uvp = container.get_pspec(group, spectrum, spws=[spw], polpairs=[(polarization, polarization)])
    if uvp.folded or list(uvp.spw_array) != [spw]:
        raise ValueError("unfolded spectra with preserved window identity required")
    code = utils.polpair_tuple2int((polarization, polarization))
    if list(uvp.polpair_array) != [code]:
        raise ValueError("requested polarization identity is absent or ambiguous")
    mapping = grouping["baselines"]
    vectors = {tuple(utils._bl_to_antnums(int(code))): vector for code, vector in zip(uvp.bl_array, uvp.bl_vecs)}
    baseline_ids, group_ids, lengths, keep, omitted = [], [], [], [], set()
    for row, value in enumerate(uvp.blpair_array):
        pair = utils._blpair_to_antnums(int(value))
        identity = pair_identity(pair)
        if identity not in mapping:
            if role == "corrupted":
                raise ValueError("corrupted baseline missing from reference grouping")
            omitted.add(identity)
            continue
        entry = mapping[identity]
        if tuple(pair[0]) != tuple(pair[1]):
            raise ValueError("unsupported cross-baseline-pair spectrum")
        vector = np.asarray(vectors[tuple(pair[0])])
        if not np.allclose(vector, entry["vector_m"], rtol=1e-10, atol=1e-8):
            raise ValueError("physical baseline vector differs from reference")
        keep.append(row)
        baseline_ids.append(identity)
        group_ids.append(entry["group_id"])
        lengths.append(float(np.linalg.norm(vector)))
    if not keep:
        raise ValueError("no reference baselines in spectral product")
    indices = np.asarray(keep)
    noise = (getattr(uvp, "stats_array", None) or {}).get("P_N", {}).get(spw)
    power, pn, valid = spectral_measurements(uvp.data_array[spw][:, :, 0],
        noise[:, :, 0] if noise is not None else None, uvp.nsample_array[spw][:, 0],
        uvp.integration_array[spw][:, 0], uvp.wgt_array[spw][:, :, :, 0], role == "corrupted")
    times = uvp.time_avg_array[indices]
    z = uvp.cosmo.f2z(np.mean(uvp.freq_array[uvp.spw_to_freq_indices(spw)]))
    return SpectrumRecords(power[indices], pn[indices], valid[indices], grid.assign(times),
        np.asarray(baseline_ids), np.asarray(group_ids), times, uvp.lst_avg_array[indices], np.asarray(lengths),
        np.asarray(lengths) * uvp.cosmo.bl_to_kperp(z, little_h=True), uvp.get_dlys(spw), uvp.get_kparas(spw),
        {"spw": int(spw), "polarization": polarization, "power_units": units, "cosmology": cosmology,
         "sources": [file_identity(path)], "window_anchor_jd": grid.anchor_jd, "window_seconds": grid.window_seconds,
         "role": role, "power_component": "real", "noise_component": "real", "unmapped_baseline_ids": sorted(omitted),
         "grouping_input": grouping["input"], "length_tolerance_m": grouping["length_tolerance_m"],
         "sample_counts": summary(uvp.nsample_array[spw][indices, 0])})
