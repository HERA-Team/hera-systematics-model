"""Empirical retained-storage estimates for physical-slice task families."""

import re

from .configuration import digest_json


def project_localized_storage(tasks, measurements, retained_bytes, reserved_bytes=0):
    """Project unfinished slices from accepted, same-family smoke measurements.

    Callers must verify inventory and measurement provenance before calling.
    Each task supplies slice, spw, axis, method, features and configuration_digest.
    Measurements supply slice, configuration_digest and positive retained_bytes,
    including the complete run footprint. Measured slices are already included
    in retained_bytes and are excluded from additional storage. Family estimates
    use the maximum measured footprint, with a single 20 percent contingency
    applied to all additional storage. This is an estimate, not an upper bound.
    """
    if any(type(value) is not int or value < 0
           for value in (retained_bytes, reserved_bytes)):
        raise ValueError("nonnegative integer storage quantities required")
    inventory, families = {}, {}
    for task in tasks:
        name = task["slice"]
        spw, axis, method = task["spw"], task["axis"], task["method"]
        if (type(spw) is not int or not 0 <= spw < 14
                or axis not in ("group", "delay")
                or method not in ("complete", "masked")
                or not isinstance(name, str) or not name
                or type(task["features"]) is not int or task["features"] < 1
                or not isinstance(task["configuration_digest"], str)
                or not re.match(r"[0-9a-f]{64}\Z", task["configuration_digest"])):
            raise ValueError("invalid physical-slice task identity")
        if name in inventory:
            raise ValueError("duplicate physical slice")
        inventory[name] = task
        families.setdefault((spw, axis, method), []).append(task)
    if not inventory:
        raise ValueError("nonempty physical-slice inventory required")
    measured = {}
    for item in measurements:
        name = item["slice"]
        if name not in inventory or name in measured:
            raise ValueError("unknown or duplicate measured slice")
        if item["configuration_digest"] != inventory[name]["configuration_digest"]:
            raise ValueError("measured configuration mismatch")
        size = item["retained_bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("positive measured run footprint required")
        measured[name] = size
    if sum(measured.values()) > retained_bytes:
        raise ValueError("measured footprints exceed total retained storage")
    rows, additional = [], reserved_bytes
    for family, members in sorted(families.items()):
        dimensions = {task["features"] for task in members}
        observed = [measured[task["slice"]] for task in members
                    if task["slice"] in measured]
        if len(dimensions) != 1 or not observed:
            raise ValueError("uniform dimensions and measured coverage required per family")
        remaining = sum(task["slice"] not in measured for task in members)
        estimate = remaining * max(observed)
        additional += estimate
        rows.append({"spw": family[0], "axis": family[1], "method": family[2],
                     "features": next(iter(dimensions)), "tasks": len(members),
                     "measured_tasks": len(observed), "remaining_tasks": remaining,
                     "maximum_measured_run_bytes": max(observed),
                     "projected_additional_bytes": estimate})
    peak = retained_bytes + (additional * 6 + 4) // 5
    return {"schema_version": 1, "families": rows,
            "retained_bytes": retained_bytes, "reserved_bytes": reserved_bytes,
            "projected_additional_bytes": additional,
            "contingency_fraction": 0.2, "projected_peak_bytes": peak,
            "cap_bytes": 1500000000000, "within_cap": peak <= 1500000000000,
            "estimator": "maximum_measured_run_footprint_per_family"}


def remaining_slice_budgets(tasks, assessment):
    """Bind unfinished slice reservations to a consistent saved assessment.

    The caller must verify the assessment's run and products before use, then
    check fresh aggregate storage and scheduler resources before submission.
    This function neither submits work nor reuses completed job identifiers.
    """
    measured = assessment["measurements"]
    expected = project_localized_storage(
        tasks, measured, assessment["retained_bytes"], assessment["reserved_bytes"])
    if any(assessment.get(key) != value for key, value in expected.items()):
        raise ValueError("assessment does not reproduce from its measured inputs")
    if not expected["within_cap"]:
        raise ValueError("assessment exceeds retained-storage cap")
    completed = {item["slice"] for item in measured}
    estimates = {(row["spw"], row["axis"], row["method"]):
                 row["maximum_measured_run_bytes"] for row in expected["families"]}
    remaining = []
    for task in sorted(tasks, key=lambda item: item["slice"]):
        if task["slice"] in completed:
            continue
        record = {key: task[key] for key in
                  ("slice", "spw", "axis", "method", "features", "configuration_digest")}
        record["projected_bytes"] = estimates[(task["spw"], task["axis"], task["method"])]
        remaining.append(record)
    additional = sum(row["projected_bytes"] for row in remaining)
    if additional + expected["reserved_bytes"] != expected["projected_additional_bytes"]:
        raise ValueError("remaining reservations do not reconcile with assessment")
    return {"schema_version": 1, "assessment_digest": digest_json(assessment),
            "completed_slices": sorted(completed), "tasks": remaining,
            "task_count": len(remaining), "projected_additional_bytes": additional,
            "external_reserved_bytes": expected["reserved_bytes"],
            "requires_fresh_storage_and_resource_admission": True}
