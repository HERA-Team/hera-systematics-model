"""Explicit file-list interfaces for visibility inventory and ideal products."""

import json
from pathlib import Path

from .production import write_json_exclusive


def read_paths(path):
    value = json.loads(Path(path).read_text())
    if (not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("a nonempty JSON list of unique file paths is required")
    return value


def run_inventory(args):
    from .visibility_inventory import inventory_visibilities

    result = inventory_visibilities(read_paths(args.files))
    write_json_exclusive(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "files": len(result["files"]),
                      "logical_bytes": result["logical_bytes"], "metadata_only": True}))
    return 0


def run_mapping(args):
    from pyuvdata import UVData
    from types import SimpleNamespace

    from .configuration import file_identity
    from .ideal_io import redundant_baseline_map
    from .visibility_inventory import inventory_visibilities

    files = read_paths(args.references)
    inventory = inventory_visibilities(files)
    entries = inventory["files"]
    for key in ("antenna_numbers", "antenna_positions"):
        if len({entry["coordinates"][key] for entry in entries}) != 1:
            raise ValueError("reference telescope geometry differs between chunks")
    pairs = sorted({tuple(pair) for pairs in inventory["baseline_inventories"].values() for pair in pairs})
    reference = UVData.from_file(files[0], read_data=False)
    source = UVData.from_file(args.source, read_data=False)
    reference = SimpleNamespace(telescope=reference.telescope, get_antpairs=lambda: pairs)
    mapping = redundant_baseline_map(reference, source)
    result = {"schema_version": 1, "baselines": mapping, "reference_metadata": inventory,
              "source": file_identity(args.source)}
    write_json_exclusive(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "baselines": len(mapping),
                      "supported": sum(entry["source_pair"] is not None for entry in mapping.values())}))
    return 0


def run_chunk(args):
    from .ideal_io import construct_chunk

    mapping = json.loads(Path(args.mapping).read_text())
    if mapping.get("schema_version") != 1 or not isinstance(mapping.get("baselines"), dict):
        raise ValueError("unsupported baseline mapping schema")
    result = construct_chunk(args.reference, read_paths(args.sources), mapping["baselines"], args.output)
    print(json.dumps({"output": result["output"], "supported_cells": result["supported_cells"],
                      "total_cells": result["total_cells"]}))
    return 0


def run_verification(args):
    from .ideal_verification import verify_ideal_chunk

    result = verify_ideal_chunk(args.reference, args.product)
    write_json_exclusive(args.output, result)
    print(json.dumps({"passed": result["passed"], "totals": result["totals"]}))
    return 0


def add_commands(commands):
    inventory = commands.add_parser("inventory", help="Inventory physical UVH5 metadata from an explicit JSON file list")
    inventory.add_argument("--files", required=True)
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(function=run_inventory)
    ideal = commands.add_parser("ideal", help="Construct and verify source-supported ideal visibilities")
    operations = ideal.add_subparsers(dest="operation", required=True)
    mapping = operations.add_parser("map", help="Build one deterministic redundant-baseline mapping")
    for name in ("references", "source", "output"):
        mapping.add_argument("--" + name, required=True)
    mapping.set_defaults(function=run_mapping)
    chunk = operations.add_parser("chunk", help="Construct a visibility chunk with explicit source files")
    for name in ("reference", "sources", "mapping", "output"):
        chunk.add_argument("--" + name, required=True)
    chunk.set_defaults(function=run_chunk)
    verify = operations.add_parser("verify", help="Check ideal coordinates, validity, counts and file identities")
    for name in ("reference", "product", "output"):
        verify.add_argument("--" + name, required=True)
    verify.set_defaults(function=run_verification)
