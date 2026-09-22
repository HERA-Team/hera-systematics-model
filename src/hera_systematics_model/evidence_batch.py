"""Export matched baseline evidence from verified retained evaluation pairs."""

import argparse
import json
import os
from pathlib import Path

from .configuration import capture_runtime, file_identity
from .evaluation import Evaluation
from .evidence import evaluation_evidence
from .production import write_json_exclusive
from .task_acceptance import validate_catalog, verify_entry


def export_catalog(catalog, output):
    catalog, output = Path(catalog), Path(output)
    if output.exists():
        raise FileExistsError(output)
    catalog_identity = file_identity(catalog)
    entries = validate_catalog(json.loads(catalog.read_text()))
    results = []
    for entry in entries:
        accepted = verify_entry(entry)
        directory = Path(entry["run"]) / "products" / entry["task"]
        paths = [directory / name for name in (
            "evaluation.npz", "evaluation.json", "descriptive.npz", "descriptive.json")]
        identities = [file_identity(path) for path in paths]
        evaluation = Evaluation.load(paths[0])
        descriptive = Evaluation.load(paths[2])
        evidence = evaluation_evidence(evaluation, descriptive)
        if (not evidence["evaluation_complete"]
                or not evidence["descriptive_fit"]["complete"]):
            raise ValueError("complete retained evaluations required")
        if ([file_identity(path) for path in paths] != identities
                or file_identity(directory / "success.json") != accepted["success"]
                or file_identity(directory / "verification.json") != accepted["verification"]):
            raise ValueError("retained evidence changed during export")
        results.append({**entry, "acceptance": accepted, "artifacts": identities,
                        "evidence": evidence})
    if file_identity(catalog) != catalog_identity:
        raise ValueError("evidence catalog changed during export")
    result = {"schema_version": 1, "passed": True, "catalog": catalog_identity,
              "task_count": len(results), "tasks": results, "runtime": capture_runtime(),
              "basis_refitted": False, "selection_repeated": False}
    write_json_exclusive(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("output")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("evidence export requires a scheduler allocation")
    result = export_catalog(args.catalog, args.output)
    print(json.dumps({"passed": result["passed"], "task_count": result["task_count"]}))


if __name__ == "__main__":
    main()
