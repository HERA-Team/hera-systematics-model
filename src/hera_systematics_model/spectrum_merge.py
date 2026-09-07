"""Deterministic, bounded-memory physical copies of single-baseline spectra."""

from pathlib import Path

import numpy as np

from .artifacts import read_artifact, write_artifact
from .configuration import file_identity
from .input_verification import VerifiedInputs
from .parity import exact_equal
from .spectrum_layout import inspect_spectrum, require_compatible, require_equal


def verify_merge_receipt(path):
    """Verify the numerical sidecar and its bound physical spectrum and input report."""
    import h5py
    import json

    arrays, metadata = read_artifact(path, "spectral-merge")
    if metadata.get("passed") is not True or file_identity(metadata["output"]["path"]) != metadata["output"]:
        raise ValueError("merged spectrum has no matching verified merge receipt")
    report_identity = metadata["input_verification"]
    if file_identity(report_identity["path"]) != report_identity:
        raise ValueError("merge input verification report changed")
    report = json.loads(Path(report_identity["path"]).read_text())
    expected = sorted([{key: item[key] for key in ("path", "bytes", "sha256")} for item in metadata["inputs"]],
                      key=lambda item: item["path"])
    if report.get("passed") is not True or report.get("inputs") != expected:
        raise ValueError("merge input verification is incomplete")
    with h5py.File(metadata["output"]["path"], "r") as handle:
        layout = inspect_spectrum(handle[metadata["group"]])
        if any(layout["counts"][key] != value for key, value in metadata["counts"].items()):
            raise ValueError("merge receipt dimensions disagree with the spectrum")
    return arrays, metadata


def stream_slices(shape, axis, offset, block_bytes):
    size = int(np.prod([length for i, length in enumerate(shape) if i != axis]))
    step = max(1, block_bytes // max(1, size))
    for start in range(0, shape[axis], step):
        source = [slice(None)] * len(shape)
        source[axis] = slice(start, min(shape[axis], start + step))
        target = list(source)
        target[axis] = slice(start + offset, min(shape[axis], start + step) + offset)
        yield tuple(source), tuple(target)


def require_attributes(left, right, label):
    if set(left) != set(right):
        raise ValueError(f"spectral metadata attribute schema differs: {label}")
    for name in left:
        require_equal(left[name], right[name], f"{label}/{name}")


def inspect_inputs(paths, group_name, expected_pairs, input_set):
    import h5py

    entries, pairs = [], set()
    input_set.identity(paths[0])
    with h5py.File(paths[0], "r") as reference:
        first = reference[group_name]
        for path in paths:
            identity = input_set.identity(path)
            with h5py.File(path, "r") as handle:
                group = handle[group_name]
                layout = inspect_spectrum(group)
                require_compatible(first, group)
                require_attributes(reference.attrs, handle.attrs, "root")
                require_attributes(first.parent.attrs, group.parent.attrs, "group")
                for name in group:
                    require_attributes(first[name].attrs, group[name].attrs, name)
                if "header" not in handle or len(handle["header"]):
                    raise ValueError("an attribute-only spectral header is required")
                for name in set(reference["header"].attrs) | set(handle["header"].attrs):
                    if name.endswith(".version"):
                        if name not in reference["header"].attrs or name not in handle["header"].attrs:
                            raise ValueError("spectral software provenance is absent")
                        require_equal(reference["header"].attrs[name], handle["header"].attrs[name], name)
                if layout["counts"]["Nblpairs"] != 1 or layout["counts"]["Nbls"] != 1:
                    raise ValueError("single-baseline merge inputs are required")
                pair = int(layout["rows"][0])
                if pair in pairs:
                    raise ValueError("duplicate physical baseline across merge inputs")
                pairs.add(pair)
                entries.append({"path": path, "identity": identity, "pair": pair,
                                "layout": layout, "header": dict(handle["header"].attrs),
                                "history": group.attrs["history"]})
    if pairs != set(expected_pairs) or len(expected_pairs) != len(pairs):
        raise ValueError("measured baseline inventory differs from expected merge inputs")
    return sorted(entries, key=lambda item: item["pair"])


def merged_counts(entries):
    samples = [row for entry in entries for row in entry["layout"]["physical_samples"]]
    if len(set(samples)) != len(samples):
        raise ValueError("duplicate physical samples across merge inputs")
    pairs = {(row[1], row[2]) for row in samples}
    return {"Nbls": len(entries), "Nblpairs": len(entries), "Nbltpairs": len(samples),
            "Ntpairs": len(pairs), "Ntimes": len({time for pair in pairs for time in pair})}


def create_destination(output, reference, group_name, entries, counts):
    """Allocate ordinary datasets; all source-specific attributes stay in a sidecar."""
    for name, value in reference.attrs.items():
        output.attrs[name] = value
    header = output.create_group("header")
    common = set.intersection(*(set(entry["header"]) for entry in entries))
    for name in sorted(common):
        value = entries[0]["header"][name]
        if all(exact_equal(value, entry["header"][name]) for entry in entries):
            header.attrs[name] = value
    source = reference[group_name]
    destination = output.create_group(group_name)
    for name, value in source.parent.attrs.items():
        destination.parent.attrs[name] = value
    for name, value in source.attrs.items():
        destination.attrs[name] = value
    for name, value in counts.items():
        destination.attrs[name] = value
    history = str(source.attrs["history"])
    destination.attrs["history"] = history + "\nPhysical baseline merge; complete source histories are retained in the merge sidecar."
    for name, (kind, axis) in entries[0]["layout"]["axes"].items():
        dataset = source[name]
        shape = list(dataset.shape)
        if kind == "constant":
            source.copy(dataset, destination, name=name)
            continue
        shape[axis] = counts["Nbls"] if kind == "baseline" else counts["Nbltpairs"]
        copied = destination.create_dataset(name, shape=tuple(shape), dtype=dataset.dtype)
        for key, value in dataset.attrs.items():
            copied.attrs[key] = value
    return destination


def transfer_entries(entries, output, group_name, block_bytes, input_set, verify=False):
    import h5py

    row_offset = 0
    destination = output[group_name]
    for baseline_offset, entry in enumerate(entries):
        input_set.identity(entry["path"])
        with h5py.File(entry["path"], "r") as handle:
            source = handle[group_name]
            for name, (kind, axis) in entry["layout"]["axes"].items():
                if kind == "constant":
                    if verify:
                        require_equal(source[name][()], destination[name][()], name)
                    continue
                offset = baseline_offset if kind == "baseline" else row_offset
                for left, right in stream_slices(source[name].shape, axis, offset,
                                                 block_bytes // source[name].dtype.itemsize):
                    values = source[name][left]
                    if verify:
                        require_equal(values, destination[name][right], name)
                    else:
                        destination[name][right] = values
        row_offset += entry["layout"]["counts"]["Nbltpairs"]


def merge_spectra(paths, output_path, expected_pairs, group_name="stokespol/interleave_averaged",
                  block_bytes=16 * 1024 * 1024):
    """Verify exact input coverage, copy, reopen, and compare every written value.

    Baselines are sorted by physical identity; each source's row order remains
    intact. The declared inputs are hashed before use and after verification.
    A failed partial product is retained and cannot be overwritten on retry.
    """
    import h5py

    paths = [Path(path).resolve(strict=True) for path in paths]
    output_path = Path(output_path).resolve()
    sidecar = output_path.with_suffix(".merge.npz")
    input_report = output_path.with_suffix(".inputs.json")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("unique, explicit spectral input paths are required")
    if not expected_pairs or any(type(value) is not int or value <= 0 for value in expected_pairs):
        raise ValueError("explicit expected baseline-pair inventory required")
    if type(block_bytes) is not int or block_bytes < 1:
        raise ValueError("positive copy block size required")
    if any(path.exists() for path in (output_path, sidecar, sidecar.with_suffix(".json"), input_report)):
        raise FileExistsError(output_path)
    arrays, attributes = {}, []
    with VerifiedInputs(input_report) as inputs:
        entries = inspect_inputs(paths, group_name, expected_pairs, inputs)
        counts = merged_counts(entries)
        with h5py.File(entries[0]["path"], "r") as source, h5py.File(output_path, "x") as output:
            create_destination(output, source, group_name, entries, counts)
            transfer_entries(entries, output, group_name, block_bytes, inputs)
        with h5py.File(output_path, "r") as output:
            result = inspect_spectrum(output[group_name])
            transfer_entries(entries, output, group_name, block_bytes, inputs, verify=True)
            if result["counts"] != {**entries[0]["layout"]["counts"], **counts}:
                raise ValueError("merged output dimensions differ from the input inventory")
        for index, entry in enumerate(entries):
            values = {"history": entry["history"], **{f"header/{key}": value for key, value in entry["header"].items()}}
            for number, (name, value) in enumerate(sorted(values.items())):
                key = f"input_{index:05d}_attribute_{number:03d}"
                arrays[key] = np.asarray(value)
                attributes.append({"input_index": index, "attribute": name, "array": key})
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("source attributes require unsupported object serialization")
    metadata = {"passed": True, "group": group_name, "counts": counts, "output": file_identity(output_path),
        "inputs": [{**entry["identity"], "baseline_pair_code": entry["pair"],
                    "row_count": entry["layout"]["counts"]["Nbltpairs"]} for entry in entries],
        "source_attributes": attributes, "input_verification": file_identity(input_report),
        "copy_block_bytes": block_bytes, "payload_verification": "exact reopened source-to-output comparison"}
    write_artifact(sidecar, "spectral-merge", arrays, metadata)
    return metadata
