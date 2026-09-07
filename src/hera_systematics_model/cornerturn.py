"""Deterministic streaming from time chunks into single-baseline UVH5 files."""

from pathlib import Path

import numpy as np

from .configuration import file_identity
from .production import write_json_exclusive
from .visibility_inventory import inventory_visibilities


def cornerturn_baselines(inputs, baselines, output_dir):
    """Write each physical row once and verify data, flags and counts after writing.

    Metadata for the full time span stays in memory. Only one visibility chunk
    is loaded at a time. Output ownership is exclusive to this invocation.
    """
    import h5py
    from pyuvdata import UVData

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
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False, parents=True)
    manifest = output_dir / "cornerturn-inputs.json"
    write_json_exclusive(manifest, {"schema_version": 1, "inputs": identities, "baseline_pairs": baselines})
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
        metadata = full.select(bls=[pair], inplace=False)
        metadata.reorder_blts(order="time")
        if set(metadata.get_antpairs()) != {pair}:
            raise ValueError("cornerturn metadata changed baseline orientation")
        keys = list(zip(metadata.time_array.tolist(), metadata.ant_1_array.tolist(), metadata.ant_2_array.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate physical input row")
        output = output_dir / f"zen.LST.baseline.{pair[0]}_{pair[1]}.sum.uvh5"
        metadata.history += "\nVisibility rows regrouped by physical baseline and time without averaging."
        metadata.initialize_uvh5_file(output, clobber=False, data_write_dtype="c16")
        writers[pair] = {"metadata": metadata, "output": output, "positions": {key: i for i, key in enumerate(keys)},
                         "written": np.zeros(len(keys), bool), "valid_cells": 0, "first": True}
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
            if not len(rows) or np.any(writer["written"][indices]):
                raise ValueError("missing or duplicate cornerturn row")
            metadata = writer["metadata"]
            for name in ("time_array", "lst_array", "integration_time", "uvw_array"):
                if not np.array_equal(header[name][rows], getattr(metadata, name)[indices]):
                    raise ValueError("cornerturn row metadata mismatch")
            for name in ("freq_array", "polarization_array"):
                if not np.array_equal(header[name].ravel(), getattr(metadata, name).ravel()):
                    raise ValueError("cornerturn spectral metadata mismatch")
            arrays = {name: values[local_rows[rows]] for name, values in payload.items()}
            metadata.write_uvh5_part(writer["output"], data_array=arrays["visdata"],
                flag_array=arrays["flags"], nsample_array=arrays["nsamples"], blt_inds=indices,
                check_header=writer["first"])
            with h5py.File(writer["output"], "r") as handle:
                for key, expected in arrays.items():
                    if not np.array_equal(handle["Data"][key][indices], expected, equal_nan=True):
                        raise ValueError("cornerturn changed numerical samples")
            writer["first"] = False
            writer["written"][indices] = True
            writer["valid_cells"] += int((np.isfinite(arrays["visdata"]) & ~arrays["flags"]
                                          & np.isfinite(arrays["nsamples"]) & (arrays["nsamples"] > 0)).sum())
        del payload
    products = []
    for pair, writer in writers.items():
        if not writer["written"].all():
            raise ValueError("cornerturn left unwritten physical rows")
        metadata = UVData.from_file(writer["output"], read_data=False)
        for name in ("time_array", "lst_array", "integration_time", "ant_1_array", "ant_2_array",
                     "uvw_array", "freq_array", "polarization_array"):
            if not np.array_equal(getattr(metadata, name), getattr(writer["metadata"], name)):
                raise ValueError("cornerturn output metadata changed")
        result = {"schema_version": 1, "output": file_identity(writer["output"]), "inputs": manifest_identity,
                  "baseline_pair": list(pair), "rows": len(writer["written"]), "valid_cells": writer["valid_cells"],
                  "all_rows_written": True, "numerical_samples_preserved": True}
        write_json_exclusive(writer["output"].with_suffix(".json"), result)
        products.append(result)
    for before in identities:
        if file_identity(before["path"]) != before:
            raise ValueError("cornerturn source changed during execution")
    return {"passed": True, "products": products, "inputs": manifest_identity}
