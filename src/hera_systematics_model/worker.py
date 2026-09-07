"""Execute one immutable task and verify its products before recording success."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .artifacts import sha256_file
from .configuration import capture_runtime, digest_json, file_identity
from .execution import execute_checked, storage_monitor
from .production import read_run, require_storage, validate_task, write_json_exclusive


def verify_input(item):
    if file_identity(item["path"]) != item:
        raise ValueError("consumed input changed")


def verify_product(directory, specification):
    path = (Path(directory) / specification["path"]).resolve()
    if not path.is_relative_to(Path(directory).resolve()) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing, empty or escaped output product")
    structure = {}
    if specification["kind"] == "manifest":
        from .product_manifest import verify_product_manifest

        structure = verify_product_manifest(directory, path)
    elif specification["kind"] == "npz":
        with np.load(path, allow_pickle=False) as product:
            for name in product.files:
                array = product[name]
                if array.dtype.hasobject:
                    raise ValueError("object array in output")
                structure[name] = {"shape": list(array.shape), "dtype": array.dtype.str}
        if not structure:
            raise ValueError("empty numerical output")
    elif specification["kind"] == "hdf5":
        import h5py
        with h5py.File(path, "r") as product:
            for required in specification.get("required_paths", []):
                if required not in product:
                    raise ValueError("required HDF5 object absent")
                obj = product[required]
                if isinstance(obj, h5py.Dataset) and (obj.shape is None or obj.size == 0):
                    raise ValueError("required HDF5 dataset is empty")

            def check(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                structure[name] = {"shape": None if obj.shape is None else list(obj.shape), "dtype": str(obj.dtype)}
                if obj.shape is None:
                    structure[name]["null_dataspace"] = True
                    obj[()]
                    return
                if obj.shape == ():
                    obj[()]
                elif obj.chunks and all(obj.shape):
                    for chunk in obj.iter_chunks():
                        obj[chunk]
                elif all(obj.shape):
                    for row in range(obj.shape[0]):
                        obj[row]
            product.visititems(check)
        if not structure:
            raise ValueError("HDF5 output has no datasets")
    return {**file_identity(path), "structure": structure}


def load_task(run, name):
    if Path(name).name != name:
        raise ValueError("invalid task name")
    task = json.loads((Path(run) / "tasks" / f"{name}.json").read_text())
    validate_task(task)
    if task["name"] != name:
        raise ValueError("task name mismatch")
    return task


def verified_receipt(run, name):
    run, task = Path(run), load_task(run, name)
    state = read_run(run)
    directory = run / "products" / name
    receipt = json.loads((directory / "success.json").read_text())
    if (receipt.get("exit_code") != 0 or receipt.get("run_digest") != state["identity_digest"]
            or receipt.get("task_digest") != digest_json(task)):
        raise ValueError("stale task success record")
    for item in task["inputs"]:
        verify_input(item)
    actual = [verify_product(directory, spec) for spec in task["outputs"]]
    if actual != receipt.get("products"):
        raise ValueError("verified product changed")
    return receipt


def run_task(run, name, require_slurm=True):
    run, task = Path(run).resolve(), load_task(run, name)
    state = read_run(run)
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("production tasks require a scheduler allocation")
    directory = run / "products" / name
    if directory.exists():
        return verified_receipt(run, name)
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir(exist_ok=False)
    record = {"run_digest": state["identity_digest"], "task_digest": digest_json(task),
        "job_id": os.environ.get("SLURM_JOB_ID"), "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_unix": time.time(), "exit_code": None, "runtime": capture_runtime()}
    write_json_exclusive(directory / "started.json", record)
    try:
        require_storage(run.parent, task["projected_bytes"])
        for item in task["inputs"]:
            verify_input(item)
        for index, item in enumerate(state["identity"]["snapshots"]):
            copy = run / "snapshots" / f"{index:03d}-{Path(item['path']).name}"
            if sha256_file(copy) != item["sha256"]:
                raise ValueError("consumed snapshot changed")
        command = [arg.replace("{output_dir}", str(directory)) for arg in task["command"]]
        environment = os.environ.copy()
        environment.update({k: v.replace("{output_dir}", str(directory)) for k, v in task["environment"].items()})
        record["storage_monitor"] = {"interval_seconds": 10., "measurement": "sampled logical file sizes"}
        check_limits = storage_monitor(run.parent, directory, task["projected_bytes"], record["storage_monitor"])
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            code = execute_checked(command, directory, environment, stdout, stderr,
                task["resources"]["hours"] * 3600, check_limits)
        record.update(exit_code=code, command=command, finished_unix=time.time())
        if code != 0:
            raise ValueError(f"underlying command exited {code}")
        for item in task["inputs"]:
            verify_input(item)
        record["products"] = [verify_product(directory, spec) for spec in task["outputs"]]
        record["storage"] = require_storage(run.parent, 0)
        write_json_exclusive(directory / "success.json", record)
        return record
    except Exception as error:
        record.update(finished_unix=time.time(), failure_type=type(error).__name__, failure=str(error))
        write_json_exclusive(directory / "failure.json", record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute and verify a single compute task")
    parser.add_argument("run")
    parser.add_argument("task")
    args = parser.parse_args(argv)
    run_task(args.run, args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
