"""Bounded subprocess concurrency inside one scheduler allocation."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import re
import subprocess
import time

from .production import write_json_exclusive


def validate_batch(tasks, workers, allocated_cpus, allocated_memory_mib,
                   cpus_per_command=2, memory_per_command_mib=16384):
    """Require deterministic unique task identities and conservative reservations."""
    resources = (workers, allocated_cpus, allocated_memory_mib, cpus_per_command, memory_per_command_mib)
    if (any(type(value) is not int or value < 1 for value in resources) or workers > 4
            or allocated_cpus > 16 or allocated_memory_mib > 131072
            or workers * cpus_per_command > allocated_cpus
            or workers * memory_per_command_mib > allocated_memory_mib):
        raise ValueError("batch concurrency exceeds its allocated CPU or memory reservation")
    if not tasks or any(not isinstance(task, dict) or set(task) != {"name", "command", "environment"} for task in tasks):
        raise ValueError("explicit batch task names, commands and environments required")
    names = []
    for task in tasks:
        if (not isinstance(task["name"], str) or not re.match(r"[a-z0-9][a-z0-9_-]*\Z", task["name"])
                or not isinstance(task["command"], list) or not task["command"]
                or any(not isinstance(arg, str) or not arg for arg in task["command"])
                or not isinstance(task["environment"], dict)
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in task["environment"].items())):
            raise ValueError("invalid batch command specification")
        names.append(task["name"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate batch task identity")
    return {"workers": workers, "allocated_cpus": allocated_cpus, "allocated_memory_mib": allocated_memory_mib,
            "cpus_per_command": cpus_per_command, "memory_per_command_mib": memory_per_command_mib}


def run_bounded_commands(tasks, directory, workers, allocated_cpus, allocated_memory_mib,
                         cpus_per_command=2, memory_per_command_mib=16384, on_completion=None):
    """Drain active commands after failure, retaining outputs and launch order.

    Threads only supervise subprocesses; interpreter and notebook state stay
    isolated. Children inherit the enclosing worker process group so its
    wall-time or storage termination reaches every active descendant.
    """
    reservation = validate_batch(tasks, workers, allocated_cpus, allocated_memory_mib,
                                 cpus_per_command, memory_per_command_mib)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(directory / "batch-start.json", {"tasks": tasks, "resources": reservation,
                                                          "started_unix": time.time()})
    records = [{"name": task["name"], "status": "not_started", "exit_code": None} for task in tasks]

    def execute(index):
        task = tasks[index]
        folder = directory / task["name"]
        folder.mkdir(exist_ok=False)
        command = [arg.replace("{output_dir}", str(folder)) for arg in task["command"]]
        environment = {**os.environ, **task["environment"]}
        record = {"name": task["name"], "command": command, "started_unix": time.time()}
        try:
            with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
                code = subprocess.run(command, cwd=folder, env=environment, stdout=stdout, stderr=stderr).returncode
            record.update(exit_code=code, status="exited_zero" if code == 0 else "failed")
        except OSError as error:
            record.update(exit_code=None, status="failed", error=str(error))
        record["finished_unix"] = time.time()
        write_json_exclusive(folder / "command-result.json", record)
        return record

    launched = 0
    failed = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        while active or (not failed and launched < len(tasks)):
            while not failed and launched < len(tasks) and len(active) < workers:
                active[executor.submit(execute, launched)] = launched
                launched += 1
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                index = active.pop(future)
                records[index] = future.result()
                if on_completion is not None:
                    records[index] = on_completion(records[index], directory / tasks[index]["name"])
                failed |= records[index]["status"] != "exited_zero"
    report = {"resources": reservation, "records": records, "launched_commands": launched,
              "all_commands_exited_zero": launched == len(tasks) and all(r["exit_code"] == 0 for r in records),
              "completion_checks_passed": None if on_completion is None else not failed and launched == len(tasks),
              "products_verified": False, "finished_unix": time.time()}
    write_json_exclusive(directory / "batch-result.json", report)
    return report
