"""Independently verify an explicit batch of completed immutable tasks."""

import argparse
import json
import os
from pathlib import Path

from .configuration import capture_runtime, file_identity
from .production import write_json_exclusive
from .scheduler import verify_task_acceptance


def validate_catalog(value):
    if (not isinstance(value, dict) or set(value) != {"schema_version", "tasks"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or not isinstance(value["tasks"], list) or not value["tasks"]):
        raise ValueError("invalid task acceptance catalog")
    names, jobs = set(), set()
    for item in value["tasks"]:
        if not isinstance(item, dict) or set(item) != {"run", "task", "job_id"}:
            raise ValueError("invalid task acceptance entry")
        run, name, job = (item[key] for key in ("run", "task", "job_id"))
        if (not all(isinstance(v, str) and v for v in (run, name, job))
                or not Path(run).is_absolute() or Path(name).name != name
                or name in (".", "..") or not job.isdigit()):
            raise ValueError("invalid task acceptance identity")
        key = (str(Path(run).resolve()), name)
        if key in names or job in jobs:
            raise ValueError("duplicate task acceptance identity")
        names.add(key)
        jobs.add(job)
    return value["tasks"]


def verify_entry(item):
    run, name = Path(item["run"]), item["task"]
    submission = json.loads((run / "submissions" / f"{name}.json").read_text())
    if submission.get("job_id") != item["job_id"] or submission.get("returncode") != 0:
        raise ValueError("submitted job differs from acceptance catalog")
    accepted = verify_task_acceptance(run, name)
    receipt = accepted["receipt"]
    if receipt.get("job_id") != item["job_id"]:
        raise ValueError("producing job differs from acceptance catalog")
    directory = run / "products" / name
    verification_path = directory / "verification.json"
    verification_identity = file_identity(verification_path)
    if not any(all(product.get(key) == verification_identity[key]
                   for key in ("path", "bytes", "sha256"))
               for product in receipt["products"]):
        raise ValueError("verification is not bound by the success receipt")
    if json.loads(verification_path.read_text()).get("passed") is not True:
        raise ValueError("task verification did not pass")
    return {
        **item, "passed": True, "scheduler_rows": accepted["scheduler_rows"],
        "run_digest": receipt["run_digest"], "task_digest": receipt["task_digest"],
        "success": file_identity(directory / "success.json"),
        "verification": verification_identity,
        "verified_outputs": len(receipt["products"]),
        "fresh_input_and_output_hashes": True,
        "manifest_members_and_numerical_structures_verified": True,
    }


def verify_catalog(catalog, output):
    catalog, output = Path(catalog), Path(output)
    if output.exists():
        raise FileExistsError(output)
    identity = file_identity(catalog)
    entries = validate_catalog(json.loads(catalog.read_text()))
    rows = [verify_entry(item) for item in entries]
    if file_identity(catalog) != identity:
        raise ValueError("task acceptance catalog changed during verification")
    result = {"schema_version": 1, "passed": True, "catalog": identity,
              "task_count": len(rows), "tasks": rows, "runtime": capture_runtime()}
    write_json_exclusive(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("output")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("batch verification requires a scheduler allocation")
    result = verify_catalog(args.catalog, args.output)
    print(json.dumps({"passed": result["passed"], "task_count": result["task_count"]}))


if __name__ == "__main__":
    main()
