"""Serialized Slurm submissions with aggregate resource and storage bounds."""

import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess

from .configuration import digest_json
from .production import Resources, read_run, require_storage, write_json_exclusive
from .worker import load_task, verified_receipt


def memory_mib(value, cpus):
    match = re.match(r"([0-9.]+)([KMGT]?)([cn]?)\Z", value.strip())
    if not match:
        raise ValueError("unknown scheduler memory format")
    amount, unit, scope = match.groups()
    scale = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "": 1}[unit]
    return int(float(amount) * scale * (cpus if scope == "c" else 1) + .5)


def queued_resources(output):
    jobs = []
    for line in output.splitlines():
        job, name, cpu, memory = line.split("|")
        if not name.startswith("hsm-"):
            continue
        if not job.isdigit():
            raise ValueError("unaccounted array task in active resource inventory")
        cpus = int(cpu)
        jobs.append({"job_id": job, "cpus": cpus, "memory_mib": memory_mib(memory, cpus)})
    return jobs


def require_resources(active, requested):
    requested = Resources(**requested)
    if (len(active) + 1 > 2 or sum(j["cpus"] for j in active) + requested.cpus > 16
            or sum(j["memory_mib"] for j in active) + requested.memory_mib > 131072):
        raise ValueError("aggregate active task resources exceed limits")


def require_dependency_resources(active, requested, dependencies, predecessors):
    """Bound every possible overlap using recorded successful-completion edges."""
    ancestors = {}

    def collect(job, visiting=frozenset()):
        if job in visiting:
            raise ValueError("cyclic scheduler dependencies")
        if job not in predecessors:
            raise ValueError("dependencies must identify recorded workflow jobs")
        if job not in ancestors:
            parents = set(predecessors[job])
            ancestors[job] = parents | set().union(*(collect(parent, visiting | {job}) for parent in parents))
        return ancestors[job]

    before = set(dependencies)
    for job in [*before, *(item["job_id"] for item in active)]:
        collect(job)
    for job in dependencies:
        before |= ancestors[job]
    possible = [job for job in active if job["job_id"] not in before]
    require_resources([], requested)
    for index, job in enumerate(possible):
        require_resources([job], requested)
        for other in possible[index + 1:]:
            a, b = job["job_id"], other["job_id"]
            if a not in ancestors[b] and b not in ancestors[a]:
                raise ValueError("aggregate active task resources exceed limits")
    return possible


def submit_task(run, name, python, package_source, partition="hera", afterok=()):
    """Reserve one task under a workflow-wide lock; never submit it twice."""
    run = Path(run).resolve()
    task = load_task(run, name)
    dependencies = sorted(set(map(str, afterok)))
    if len(dependencies) != len(afterok) or any(not value.isdigit() for value in dependencies):
        raise ValueError("unique scheduler dependency identifiers are required")
    state = read_run(run)
    root = run.parent
    submissions = run / "submissions"
    submissions.mkdir(exist_ok=True)
    marker = submissions / f"{name}.json"
    with (root / ".scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            existing = json.loads(marker.read_text())
            if existing.get("task_digest") != digest_json(task) or existing.get("run_digest") != state["identity_digest"]:
                raise ValueError("submission identity changed")
            if existing.get("afterok", []) != dependencies:
                raise ValueError("submission dependencies changed")
            if "job_id" in existing:
                return existing
            raise ValueError("submission outcome requires reconciliation")
        queue = subprocess.check_output(["squeue", "--me", "--noheader", "--format=%i|%j|%C|%m"], text=True)
        active = queued_resources(queue)
        ids = {job["job_id"] for job in active}
        reservation = 0
        found = set()
        known_jobs = set()
        predecessors = {}
        for path in root.glob("*/submissions/*.json"):
            if path.name.endswith(".dispatch.json"):
                continue
            record = json.loads(path.read_text())
            if record.get("job_id"):
                known_jobs.add(record["job_id"])
                predecessors[record["job_id"]] = record.get("afterok", [])
            if record.get("job_id") in ids:
                reservation += record["projected_bytes"]
                found.add(record["job_id"])
        if found != ids:
            raise ValueError("active workflow jobs lack storage reservations")
        if not set(dependencies).issubset(known_jobs):
            raise ValueError("dependencies must identify recorded workflow jobs")
        simultaneous = require_dependency_resources(active, task["resources"], dependencies, predecessors)
        storage = require_storage(root, reservation + task["projected_bytes"])
        worker = ["env", f"PYTHONPATH={Path(package_source).resolve()}", "PYTHONDONTWRITEBYTECODE=1",
                  "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                  str(Path(python).resolve()), "-m", "hera_systematics_model.worker", str(run), name]
        resources = task["resources"]
        command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", f"--partition={partition}",
            f"--job-name=hsm-{run.name[:8]}-{name}", f"--cpus-per-task={resources['cpus']}",
            f"--mem={resources['memory_mib']}M", f"--time={resources['hours']}:00:00",
            f"--output={submissions / (name + '.slurm.log')}", f"--chdir={run}", "--wrap", shlex.join(worker)]
        if dependencies:
            command.insert(1, "--dependency=afterok:" + ":".join(dependencies))
        record = {"command": command, "projected_bytes": task["projected_bytes"],
                  "task_digest": digest_json(task), "run_digest": state["identity_digest"],
                  "resources": resources, "active_at_submission": active, "storage": storage,
                  "afterok": dependencies, "potentially_simultaneous": simultaneous}
        # A reservation written before dispatch makes an uncertain reply non-retryable.
        write_json_exclusive(marker, record)
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        outcome = {**record, "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        job_id = process.stdout.strip().split(";")[0]
        if process.returncode == 0 and job_id.isdigit():
            outcome["job_id"] = job_id
        write_json_exclusive(submissions / f"{name}.dispatch.json", outcome)
        if "job_id" not in outcome:
            raise ValueError("scheduler submission failed or returned an uncertain identifier")
        # Dispatch records are immutable; the reservation is enriched once under the lock.
        marker.write_text(json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n")
        return outcome


def verify_task_acceptance(run, name):
    submission = json.loads((Path(run) / "submissions" / f"{name}.json").read_text())
    job = submission["job_id"]
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", job,
        "--format=JobIDRaw,State,ExitCode,AllocCPUS,ReqMem,Elapsed,MaxRSS"], text=True)
    rows = [line.split("|") for line in output.splitlines() if line]
    base = [row for row in rows if row[0] == job]
    if len(base) != 1 or any(row[1] != "COMPLETED" or row[2] != "0:0" for row in rows):
        raise ValueError("scheduler job or step did not complete successfully")
    return {"scheduler_rows": rows, "receipt": verified_receipt(run, name)}
