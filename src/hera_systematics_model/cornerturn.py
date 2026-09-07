"""Deterministic streaming from time chunks into single-baseline UVH5 files."""

from pathlib import Path

import numpy as np

from .configuration import file_identity
from .production import write_json_exclusive
from .row_buffer import RowBuffer
from .visibility_inventory import inventory_visibilities


def _write_rows(writer, indices, arrays):
    import h5py

    if np.any(writer["written"][indices]):
        raise ValueError("duplicate cornerturn output write")
    writer["metadata"].write_uvh5_part(writer["output"], data_array=arrays["visdata"],
        flag_array=arrays["flags"], nsample_array=arrays["nsamples"], blt_inds=indices,
        check_header=writer["first"])
    with h5py.File(writer["output"], "r") as handle:
        for key, expected in arrays.items():
            if not np.array_equal(handle["Data"][key][indices], expected, equal_nan=True):
                raise ValueError("cornerturn changed numerical samples")
    writer["first"] = False
    writer["written"][indices] = True


def cornerturn_baselines(inputs, baselines, output_dir, uvw_policy="preserve", write_buffer_rows=32):
    """Write each physical row once and verify data, flags and counts after writing.

    Metadata for the full time span stays in memory. Only one visibility chunk
    is loaded at a time, with bounded pending rows per output baseline.
    Output ownership is exclusive to this invocation.
    """
    import h5py
    from pyuvdata import UVData
    from .visibility_compatibility import add_legacy_orientation, legacy_orientation

    if type(write_buffer_rows) is not int or write_buffer_rows < 1:
        raise ValueError("a positive integer write-buffer row limit is required")
    if uvw_policy not in ("preserve", "recalculate_unprojected"):
        raise ValueError("unsupported cornerturn UVW policy")
    baselines = [tuple(pair) for pair in baselines]
    if (not baselines or len(set(baselines)) != len(baselines)
            or any(len(pair) != 2 or any(type(ant) is not int or ant < 0 for ant in pair) for pair in baselines)):
        raise ValueError("unique physical antenna pairs are required")
    baselines.sort()
    inventory = inventory_visibilities(inputs)
    entries = inventory["files"]
    for key in ("freq_array", "polarization_array", "antenna_numbers", "antenna_positions"):
        if len({entry["coordinates"][key] for entry in entries}) != 1:
            raise ValueError("cornerturn input coordinates differ")
    if len({entry["vis_units"] for entry in entries}) != 1:
        raise ValueError("cornerturn input units differ")
    available = {tuple(pair) for members in inventory["baseline_inventories"].values() for pair in members}
    if not set(baselines).issubset(available):
        raise ValueError("requested baseline is absent from input chunks")
    files = [entry["path"] for entry in entries]
    identities = [file_identity(path) for path in files]
    orientations = []
    for path in files:
        with h5py.File(path, "r") as handle:
            orientations.append(legacy_orientation(handle["Header"]))
    if len(set(orientations)) != 1:
        raise ValueError("cornerturn input feed orientations differ")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False, parents=True)
    manifest = output_dir / "cornerturn-inputs.json"
    write_json_exclusive(manifest, {"schema_version": 1, "inputs": identities, "baseline_pairs": baselines,
                                    "uvw_policy": uvw_policy, "write_buffer_rows": write_buffer_rows})
    manifest_identity = file_identity(manifest)
    parts = [UVData.from_file(path, read_data=False) for path in files]
    full = parts[0]
    if len(parts) > 1:
        # Appending baseline-major chunks does not preserve a rectangular row
        # ordering. Validate after restoring the global physical-time order.
        full.fast_concat(parts[1:], axis="blt", inplace=True, run_check=False)
        full.reorder_blts(order="time")
    del parts
    writers = {}
    for pair in baselines:
        # Selection can leave strided views owning a copy of every input row.
        # Compact those arrays before retaining one object per baseline.
        metadata = full.select(bls=[pair], inplace=False).copy()
        metadata.reorder_blts(order="time")
        if set(metadata.get_antpairs()) != {pair}:
            raise ValueError("cornerturn metadata changed baseline orientation")
        keys = list(zip(metadata.time_array.tolist(), metadata.ant_1_array.tolist(), metadata.ant_2_array.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate physical input row")
        original_uvws = metadata.uvw_array.copy()
        geometry = {"policy": "preserve", "physical_geometry_check": False}
        if uvw_policy == "recalculate_unprojected":
            from .visibility_geometry import recalculate_unprojected_uvws

            geometry = recalculate_unprojected_uvws(metadata)
            metadata.history += "\nUnprojected UVW metadata recalculated from antenna positions and physical row identities."
        output = output_dir / f"zen.LST.baseline.{pair[0]}_{pair[1]}.sum.uvh5"
        metadata.history += "\nVisibility rows regrouped by physical baseline and time without averaging."
        metadata.initialize_uvh5_file(output, clobber=False, data_write_dtype="c16")
        with h5py.File(output, "r+") as handle:
            feed_metadata = add_legacy_orientation(handle["Header"])
            if feed_metadata["x_orientation"] != orientations[0]:
                raise ValueError("cornerturn changed physical feed orientation")
        writers[pair] = {"metadata": metadata, "output": output, "positions": {key: i for i, key in enumerate(keys)},
                         "written": np.zeros(len(keys), bool), "seen": np.zeros(len(keys), bool), "valid_cells": 0, "first": True,
                         "original_uvws": original_uvws, "geometry": geometry, "feed_metadata": feed_metadata}
        writer = writers[pair]
        writer["buffer"] = RowBuffer(write_buffer_rows,
            lambda indices, arrays, writer=writer: _write_rows(writer, indices, arrays))
    del full
    for entry in entries:
        available_here = {tuple(pair) for pair in inventory["baseline_inventories"][entry["baseline_inventory"]]}
        selected = sorted(set(baselines) & available_here)
        if not selected:
            continue
        with h5py.File(entry["path"], "r") as handle:
            header = {name: handle["Header"][name][()] for name in ("time_array", "lst_array", "integration_time",
                      "ant_1_array", "ant_2_array", "uvw_array", "freq_array", "polarization_array")}
            selections = {pair: np.flatnonzero((header["ant_1_array"] == pair[0])
                                               & (header["ant_2_array"] == pair[1])) for pair in selected}
            requested = np.sort(np.concatenate(list(selections.values())))
            payload = {name: handle["Data"][name][requested] for name in ("visdata", "flags", "nsamples")}
        local_rows = np.full(len(header["time_array"]), -1, int)
        local_rows[requested] = np.arange(len(requested))
        for pair in selected:
            rows = selections[pair]
            rows = rows[np.argsort(header["time_array"][rows], kind="stable")]
            writer = writers[pair]
            indices = np.array([writer["positions"][(float(header["time_array"][row]), *pair)] for row in rows])
            if not len(rows) or np.any(writer["seen"][indices]):
                raise ValueError("missing or duplicate cornerturn row")
            metadata = writer["metadata"]
            for name in ("time_array", "lst_array", "integration_time", "uvw_array"):
                expected = writer["original_uvws"] if name == "uvw_array" else getattr(metadata, name)
                if not np.array_equal(header[name][rows], expected[indices]):
                    raise ValueError("cornerturn row metadata mismatch")
            for name in ("freq_array", "polarization_array"):
                if not np.array_equal(header[name].ravel(), getattr(metadata, name).ravel()):
                    raise ValueError("cornerturn spectral metadata mismatch")
            arrays = {name: values[local_rows[rows]] for name, values in payload.items()}
            writer["buffer"].append(indices, arrays)
            writer["seen"][indices] = True
            writer["valid_cells"] += int((np.isfinite(arrays["visdata"]) & ~arrays["flags"]
                                          & np.isfinite(arrays["nsamples"]) & (arrays["nsamples"] > 0)).sum())
        del payload
    products = []
    for pair, writer in writers.items():
        writer["buffer"].flush()
        if not writer["written"].all():
            raise ValueError("cornerturn left unwritten physical rows")
        metadata = UVData.from_file(writer["output"], read_data=False)
        for name in ("time_array", "lst_array", "integration_time", "ant_1_array", "ant_2_array",
                     "uvw_array", "freq_array", "polarization_array"):
            if not np.array_equal(getattr(metadata, name), getattr(writer["metadata"], name)):
                raise ValueError("cornerturn output metadata changed")
        with h5py.File(writer["output"], "r") as handle:
            if legacy_orientation(handle["Header"]) != orientations[0]:
                raise ValueError("cornerturn output feed orientation changed")
        result = {"schema_version": 1, "output": file_identity(writer["output"]), "inputs": manifest_identity,
                  "baseline_pair": list(pair), "rows": len(writer["written"]), "valid_cells": writer["valid_cells"],
                  "all_rows_written": True, "numerical_samples_preserved": True, "geometry": writer["geometry"],
                  "feed_metadata": writer["feed_metadata"],
                  "write_buffer": {"row_limit": write_buffer_rows,
                                   "maximum_buffered_rows": writer["buffer"].maximum_buffered_rows,
                                   "write_calls": writer["buffer"].write_calls}}
        write_json_exclusive(writer["output"].with_suffix(".json"), result)
        products.append(result)
    for before in identities:
        if file_identity(before["path"]) != before:
            raise ValueError("cornerturn source changed during execution")
    return {"passed": True, "products": products, "inputs": manifest_identity}
