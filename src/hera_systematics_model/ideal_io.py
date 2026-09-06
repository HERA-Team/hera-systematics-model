"""HERA I/O adapter for deterministic ideal visibility chunks."""

from pathlib import Path

import numpy as np

from .configuration import file_identity
from .ideal import interpolate_supported, periodic_source_order, replace_reference_visibilities
from .production import write_json_exclusive


def baseline_key(pair):
    return f"{int(pair[0])}_{int(pair[1])}"


def redundant_baseline_map(reference, source):
    """Map each reference pair uniquely, recording absent source coverage."""
    from hera_cal.red_groups import RedundantGroups

    positions = reference.telescope.get_enu_antpos()
    antennas = reference.telescope.antenna_numbers
    groups = RedundantGroups.from_antpos(antpos=dict(zip(antennas, positions)))
    available = source.get_antpairs()
    result = {}
    for pair in sorted(reference.get_antpairs()):
        matches = groups.get_reds_in_bl_set(pair, bl_set=available, include_conj=True,
                                            match_conj_to_set=False, include_conj_only_if_missing=True)
        if len(matches) > 1:
            raise ValueError("ambiguous redundant source baseline")
        result[baseline_key(pair)] = {"reference_pair": [int(value) for value in pair],
            "source_pair": [int(value) for value in next(iter(matches))] if matches else None,
            "exclusion": None if matches else "source_baseline_absent"}
    return result


def choose_source_files(inventory, target_lsts, buffer_rad):
    """Select the same bracketing cubic knots in deterministic circular order."""
    source, target, order = periodic_source_order([entry["lst_rad"] for entry in inventory], target_lsts)
    before = np.flatnonzero(source <= target.min() - buffer_rad)
    after = np.flatnonzero(source >= target.max() + buffer_rad)
    if not len(before) or not len(after):
        raise ValueError("source inventory does not bracket the reference chunk")
    selected = order[before[-1]:after[0] + 1]
    if len(selected) < 4:
        raise ValueError("insufficient source knots for cubic interpolation")
    return [inventory[index]["path"] for index in selected]


def construct_chunk(reference_file, source_files, mapping, output):
    """Assign visibilities by physical baseline, time and polarization identities."""
    from pyuvdata import UVData

    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    reference = UVData.from_file(reference_file)
    pairs = reference.get_antpairs()
    if any(baseline_key(pair) not in mapping for pair in pairs):
        raise ValueError("reference baseline absent from deterministic mapping")
    requested = sorted({tuple(mapping[baseline_key(pair)]["source_pair"]) for pair in pairs
                        if mapping[baseline_key(pair)]["source_pair"] is not None})
    if not requested:
        raise ValueError("reference chunk has no supported source baselines")
    source = UVData.from_file(list(source_files), bls=requested, axis="blt")
    if not np.array_equal(source.freq_array, reference.freq_array) or source.vis_units != reference.vis_units:
        raise ValueError("source/reference frequency or visibility unit mismatch")
    actual = source.get_antpairs()
    if any(pair not in actual for pair in requested):
        raise ValueError("source reader did not preserve requested baseline orientation")
    source_pol = {int(value): i for i, value in enumerate(source.polarization_array)}
    if any(int(value) not in source_pol for value in reference.polarization_array):
        raise ValueError("source polarization is unavailable")
    pol_indices = [source_pol[int(value)] for value in reference.polarization_array]
    values = np.full(reference.data_array.shape, np.nan + 1j * np.nan, complex)
    supported = np.zeros(values.shape, bool)
    for pair in pairs:
        mapped = mapping[baseline_key(pair)]["source_pair"]
        if mapped is None:
            continue
        rows = np.flatnonzero((reference.ant_1_array == pair[0]) & (reference.ant_2_array == pair[1]))
        rows = rows[np.argsort(reference.time_array[rows], kind="stable")]
        source_rows = np.flatnonzero((source.ant_1_array == mapped[0]) & (source.ant_2_array == mapped[1]))
        knots, targets, order = periodic_source_order(source.lst_array[source_rows], reference.lst_array[rows])
        source_rows = source_rows[order]
        data = source.data_array[source_rows][:, :, pol_indices]
        counts = source.nsample_array[source_rows][:, :, pol_indices]
        valid = (~source.flag_array[source_rows][:, :, pol_indices] & np.isfinite(counts) & (counts > 0))
        values[rows], supported[rows] = interpolate_supported(knots, data, valid, targets)
    provenance = {"reference": file_identity(reference_file), "sources": [file_identity(p) for p in source_files],
                  "baseline_mapping": {baseline_key(pair): mapping[baseline_key(pair)] for pair in sorted(pairs)},
                  "supported_cells": int(supported.sum()), "total_cells": int(supported.size)}
    result = replace_reference_visibilities(reference, values, supported, source.vis_units, source.history,
                                            "Source file identities are stored in the product JSON sidecar.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_uvh5(output, clobber=False, fix_autos=True)
    provenance["output"] = file_identity(output)
    write_json_exclusive(output.with_suffix(".json"), provenance)
    return provenance
