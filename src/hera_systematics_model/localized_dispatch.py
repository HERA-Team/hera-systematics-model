"""Collect complete, accepted physical-slice inventories before dispatch."""

import json
from pathlib import Path

from .configuration import AnalysisConfig, digest_json, file_identity
from .scheduler import verify_task_acceptance


def collect_localized_tasks(parents, expected_spws=tuple(range(14))):
    """Validate all inventories before returning any executable slice records.

    This reads and verifies products but never submits work. Run on a compute
    node when the accepted inventories reference large product manifests.
    The per-slice worker must verify its parent acceptance again at execution.
    """
    expected = list(expected_spws)
    if (not expected or any(type(spw) is not int or spw < 0 for spw in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError("unique nonnegative spectral windows required")
    if (len(parents) != len(expected)
            or sorted(item["spw"] for item in parents) != sorted(expected)):
        raise ValueError("incomplete or duplicate spectral window inventory")
    tasks, acceptances = [], []
    for parent in sorted(parents, key=lambda item: item["spw"]):
        run, name, spw = Path(parent["run"]), parent["task"], parent["spw"]
        if not run.is_absolute() or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("absolute run and single task name required")
        acceptance = verify_task_acceptance(run, name)
        source = run / "products" / name
        inventory_path = source / "inventory.json"
        inventory_identity = file_identity(inventory_path)
        inventory = json.loads(inventory_path.read_text())
        verification = json.loads((source / "verification.json").read_text())
        if (inventory.get("schema_version") != 1 or inventory.get("spw") != spw
                or verification.get("passed") is not True or verification.get("spw") != spw):
            raise ValueError("unverified or incompatible localized inventory")
        counts = {axis: verification[key] for axis, key in
                  (("group", "groups"), ("delay", "delays"))}
        if any(type(count) is not int or count < 1 for count in counts.values()):
            raise ValueError("positive physical slice dimensions required")
        required = {(axis, index, method) for axis, count in counts.items()
                    for index in range(count) for method in ("complete", "masked")}
        entries = inventory["tasks"]
        if any(type(entry["index"]) is not int for entry in entries):
            raise ValueError("integer physical slice index required")
        found = [(entry["axis"], entry["index"], entry["method"]) for entry in entries]
        if (len(found) != len(required) or set(found) != required
                or verification["tasks"] != len(required)):
            raise ValueError("physical slices or missing-data methods are absent or repeated")
        for entry in sorted(entries, key=lambda item: (item["axis"], item["index"], item["method"])):
            axis, index, method = entry["axis"], entry["index"], entry["method"]
            slice_name = f"spw-{spw:02d}-{axis}-{index:04d}-{method}"
            config_path = source / "configurations" / (slice_name + ".json")
            config = AnalysisConfig.load(config_path)
            config_identity = file_identity(config_path)
            reference = AnalysisConfig(representations=[inventory["selected_representation"]],
                                       methods=[method], include_kernel=False, **{axis: index})
            shape = [1, counts["delay"]] if axis == "group" else [counts["group"], 1]
            if (entry["name"] != slice_name or entry["identity"]["spw"] != spw
                    or entry["feature_shape"] != shape
                    or entry["configuration_file"] != config_identity
                    or config.as_dict() != entry["configuration"]
                    or config.as_dict() != reference.as_dict()
                    or digest_json(config.as_dict()) != entry["configuration_digest"]
                    or entry["feature_partition_axis"] != ("delay" if axis == "group" else "group")
                    or entry["feature_guard_bins"] != (2 if axis == "group" else 1)):
                raise ValueError("localized slice configuration or identity mismatch")
            tasks.append({"spw": spw, "run": str(run), "inventory_task": name,
                          "slice": slice_name, "configuration": config_identity,
                          "configuration_digest": entry["configuration_digest"],
                          "inventory": inventory_identity})
        if file_identity(inventory_path) != inventory_identity:
            raise ValueError("localized inventory changed during collection")
        acceptances.append({"spw": spw, "run": str(run), "task": name,
                            "acceptance": acceptance})
    return {"schema_version": 1, "spectral_windows": sorted(expected),
            "task_count": len(tasks), "tasks": tasks, "acceptances": acceptances}
