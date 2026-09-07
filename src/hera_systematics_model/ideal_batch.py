"""Deterministic ideal chunk batches with retained verification products."""

from pathlib import Path
import warnings

from .configuration import digest_json, file_identity
from .ideal_io import construct_chunk
from .ideal_verification import verify_ideal_chunk
from .input_verification import VerifiedInputs
from .production import write_json_exclusive
from .visibility_inventory import visibility_header


def prepare_chunks(chunk_inventory, reference_inventory, baselines=None):
    """Resolve explicit selections from measured reference baseline inventories."""
    if chunk_inventory.get("schema_version") != 1 or reference_inventory.get("schema_version") != 1:
        raise ValueError("unsupported visibility inventory schema")
    entries = {entry["path"]: entry for entry in reference_inventory["files"]}
    if len(entries) != len(reference_inventory["files"]):
        raise ValueError("duplicate reference metadata entries")
    selected = None if baselines is None else {tuple(pair) for pair in baselines}
    if selected is not None and (not selected or len(selected) != len(baselines)
            or any(len(pair) != 2 or any(type(a) is not int or a < 0 for a in pair) for pair in selected)):
        raise ValueError("unique physical baseline selections are required")
    result, references, outputs, covered = [], set(), set(), set()
    for chunk in chunk_inventory["chunks"]:
        reference = str(Path(chunk["reference"]).resolve())
        if reference not in entries or reference in references:
            raise ValueError("absent or duplicate reference chunk metadata")
        references.add(reference)
        entry = entries[reference]
        if entry["baseline_inventory"] != chunk["baseline_inventory"]:
            raise ValueError("chunk baseline inventory differs from reference metadata")
        available = {tuple(pair) for pair in reference_inventory["baseline_inventories"][entry["baseline_inventory"]]}
        pairs = available if selected is None else available & selected
        if not pairs:
            continue
        covered |= pairs
        output = Path(chunk["output"])
        if (output.name != str(output) or output.suffix != ".uvh5" or str(output) in outputs
                or not chunk["sources"] or len(set(chunk["sources"])) != len(chunk["sources"])):
            raise ValueError("invalid or duplicate ideal output/source paths")
        outputs.add(str(output))
        result.append({**chunk, "reference": reference, "selected_baselines": sorted(pairs),
                       "reference_metadata": entry, "selection_applied": selected is not None})
    if not result or selected is not None and covered != selected:
        raise ValueError("selected baselines lack reference chunk coverage")
    return sorted(result, key=lambda item: (item["reference_metadata"]["times_jd"][0], item["reference"]))


def construct_batch(chunks, mapping, output_dir, progress=None):
    """Construct all declared chunks, retaining each check and any failure."""
    if not chunks or len({chunk["output"] for chunk in chunks}) != len(chunks):
        raise ValueError("a nonempty batch with unique outputs is required")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(output_dir / "chunk-tasks.json", {"schema_version": 1, "chunks": chunks})
    results, notice_counts = [], {}
    with VerifiedInputs(output_dir / "input-verification.json", progress=progress) as inputs:
        for index, chunk in enumerate(chunks):
            inputs.identity(chunk["reference"])
            measured = visibility_header(chunk["reference"])
            measured["baseline_inventory"] = digest_json(measured.pop("baseline_pairs"))
            if measured != chunk["reference_metadata"]:
                raise ValueError("reference chunk metadata changed since inventory")
            output = output_dir / chunk["output"]
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                construct_chunk(chunk["reference"], chunk["sources"], mapping, output,
                    baselines=chunk["selected_baselines"] if chunk["selection_applied"] else None, input_set=inputs)
                result = verify_ideal_chunk(chunk["reference"], output, input_set=inputs)
            for notice in notices:
                key = (notice.category.__name__, str(notice.message))
                notice_counts[key] = notice_counts.get(key, 0) + 1
            verification = output.with_suffix(".verification.json")
            write_json_exclusive(verification, result)
            results.append({"output": result["product"], "verification": file_identity(verification),
                            "totals": result["totals"], "baseline_cells": result["baseline_cells"]})
            if progress is not None:
                progress("chunk", index + 1, results[-1])
    summary = {"schema_version": 1, "passed": True, "chunks": results,
               "input_verification": file_identity(output_dir / "input-verification.json"),
               "warnings": [{"category": category, "message": message, "count": count}
                            for (category, message), count in sorted(notice_counts.items())]}
    write_json_exclusive(output_dir / "verification.json", summary)
    return summary
