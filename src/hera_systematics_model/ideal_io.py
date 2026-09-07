"""HERA I/O adapter for deterministic ideal visibility chunks."""

from pathlib import Path

import numpy as np

from .configuration import file_identity
from .ideal import interpolate_supported, periodic_source_order, replace_reference_visibilities
from .input_verification import input_identity
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
        mapped = tuple(next(iter(matches))) if matches else None
        stored = mapped if mapped in available else mapped[::-1] if mapped is not None else None
        if stored is not None and stored not in available:
            raise ValueError("mapped source baseline is unavailable in either orientation")
        result[baseline_key(pair)] = {"reference_pair": [int(value) for value in pair],
            "source_pair": list(map(int, mapped)) if mapped else None,
            "stored_pair": list(map(int, stored)) if stored else None,
            "conjugate": mapped != stored,
            "exclusion": None if matches else "source_baseline_absent"}
    return result


def source_polarizations(source_pols, reference_pols, conjugate=False):
    """Reversal conjugates visibility and exchanges the two cross-polarizations."""
    swap = {-7: -8, -8: -7, -3: -4, -4: -3}
    available = {int(value): index for index, value in enumerate(source_pols)}
    requested = [swap.get(int(value), int(value)) if conjugate else int(value) for value in reference_pols]
    if any(value not in available for value in requested):
        raise ValueError("source polarization is unavailable")
    return [available[value] for value in requested]


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


def construct_chunk(reference_file, source_files, mapping, output, baselines=None, input_set=None):
    """Assign visibilities by physical baseline, time and polarization identities."""
    from pyuvdata import UVData

    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    reference_identity = input_identity(reference_file, input_set)
    source_files = list(source_files)
    source_identities = [input_identity(path, input_set) for path in source_files]
    if baselines is not None:
        baselines = sorted(tuple(pair) for pair in baselines)
        if (not baselines or len(set(baselines)) != len(baselines)
                or any(len(pair) != 2 or any(type(a) is not int or a < 0 for a in pair) for pair in baselines)):
            raise ValueError("unique physical antenna pairs are required")
    reference = UVData.from_file(reference_file, bls=baselines)
    pairs = reference.get_antpairs()
    if baselines is not None and sorted(pairs) != baselines:
        raise ValueError("reference reader changed the selected baseline identities")
    if any(baseline_key(pair) not in mapping for pair in pairs):
        raise ValueError("reference baseline absent from deterministic mapping")
    requested = sorted({tuple(mapping[baseline_key(pair)]["stored_pair"]) for pair in pairs
                        if mapping[baseline_key(pair)]["source_pair"] is not None})
    if not requested:
        raise ValueError("reference chunk has no supported source baselines")
    source = UVData.from_file(list(source_files), bls=requested, axis="blt")
    if not np.array_equal(source.freq_array, reference.freq_array) or source.vis_units != reference.vis_units:
        raise ValueError("source/reference frequency or visibility unit mismatch")
    actual = source.get_antpairs()
    if any(pair not in actual for pair in requested):
        raise ValueError("source reader did not preserve requested baseline orientation")
    values = np.full(reference.data_array.shape, np.nan + 1j * np.nan, complex)
    supported = np.zeros(values.shape, bool)
    for pair in pairs:
        entry = mapping[baseline_key(pair)]
        mapped = entry["stored_pair"]
        if mapped is None:
            continue
        pol_indices = source_polarizations(source.polarization_array, reference.polarization_array,
                                          entry["conjugate"])
        rows = np.flatnonzero((reference.ant_1_array == pair[0]) & (reference.ant_2_array == pair[1]))
        rows = rows[np.argsort(reference.time_array[rows], kind="stable")]
        source_rows = np.flatnonzero((source.ant_1_array == mapped[0]) & (source.ant_2_array == mapped[1]))
        knots, targets, order = periodic_source_order(source.lst_array[source_rows], reference.lst_array[rows])
        source_rows = source_rows[order]
        data = source.data_array[source_rows][:, :, pol_indices]
        if entry["conjugate"]:
            data = data.conj()
        # Model visibilities do not represent counted observations. Their stored
        # sample counts may be zero even when finite, unflagged sky values exist.
        valid = ~source.flag_array[source_rows][:, :, pol_indices] & np.isfinite(data)
        values[rows], supported[rows] = interpolate_supported(knots, data, valid, targets)
    for before in [reference_identity, *source_identities]:
        if input_identity(before["path"], input_set) != before:
            raise ValueError("ideal input changed during construction")
    provenance = {"reference": reference_identity, "sources": source_identities,
                  "baseline_mapping": {baseline_key(pair): mapping[baseline_key(pair)] for pair in sorted(pairs)},
                  "supported_cells": int(supported.sum()), "total_cells": int(supported.size),
                  "source_support_policy": "finite_unflagged_interpolation_knots", "source_counts_used": False,
                  "reference_baseline_selection": [list(pair) for pair in baselines] if baselines is not None else None,
                  "batch_input_verification": str(input_set.report_path) if input_set is not None else None}
    result = replace_reference_visibilities(reference, values, supported, source.vis_units, source.history,
                                            "Source file identities are stored in the product JSON sidecar.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_uvh5(output, clobber=False, fix_autos=True)
    provenance["output"] = file_identity(output)
    write_json_exclusive(output.with_suffix(".json"), provenance)
    return provenance
