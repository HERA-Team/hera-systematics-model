"""Immutable run definitions, retained-storage checks and input snapshots."""

from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil

from .artifacts import canonical_json, sha256_file
from .configuration import digest_json, file_identity

_INDEX = ".retained-index.json"
_LOCK = ".retained-index.lock"
_BOOKKEEPING = {_INDEX, _LOCK, _INDEX + ".tmp"}


@dataclass(frozen=True)
class Resources:
    cpus: int = 1
    memory_mib: int = 16384
    hours: int = 24

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.cpus, self.memory_mib, self.hours))
                or not 1 <= self.cpus <= 16 or not 1 <= self.memory_mib <= 131072
                or not 1 <= self.hours <= 24):
            raise ValueError("task exceeds CPU, memory or wall-time bounds")


def _bookkeeping(path):
    return Path(path).name in _BOOKKEEPING


def _directory_bytes(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (base / name).is_symlink()]
        for name in filenames:
            file_path = base / name
            if _bookkeeping(file_path) or file_path.is_symlink() or not file_path.is_file():
                continue
            total += file_path.stat().st_size
    return total


def _frozen_signature(path):
    """Identify run directories whose products cannot grow until a receipt changes."""
    submissions = Path(path) / "submissions"
    if not submissions.is_dir():
        return None
    products = Path(path) / "products"
    tasks = []
    found = False
    try:
        markers = sorted(submissions.iterdir())
    except FileNotFoundError:
        return None
    for marker in markers:
        name = marker.name
        if not name.endswith(".json") or name.endswith(".dispatch.json"):
            continue
        found = True
        task = name[:-5]
        success = products / task / "success.json"
        failure = products / task / "failure.json"
        started = products / task / "started.json"
        if started.is_file() and not success.is_file() and not failure.is_file():
            return None
        receipt = success if success.is_file() else failure if failure.is_file() else marker
        state = "success" if success.is_file() else "failure" if failure.is_file() else "queued"
        st = receipt.stat()
        tasks.append([task, state, st.st_mtime_ns, st.st_size])
    run_json = Path(path) / "run.json"
    if not found or not run_json.is_file():
        return None
    st = run_json.stat()
    return [st.st_mtime_ns, st.st_size, tasks]


def _read_index(root):
    path = Path(root) / _INDEX
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"schema_version": 1, "entries": {}}
    entries = payload.get("entries")
    if payload.get("schema_version") != 1 or not isinstance(entries, dict):
        return {"schema_version": 1, "entries": {}}
    return payload


def _write_index(root, payload):
    root = Path(root)
    path = root / _INDEX
    tmp = root / (_INDEX + ".tmp")
    with (root / _LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tmp.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        os.replace(tmp, path)


def retained_bytes(root):
    """Count new logical file bytes without following links to shared inputs."""
    root = Path(root)
    index = _read_index(root)
    entries = dict(index.get("entries", {}))
    total = 0
    seen = set()
    dirty = False
    try:
        children = list(root.iterdir())
    except FileNotFoundError:
        return 0
    for child in children:
        if child.is_symlink() or _bookkeeping(child):
            continue
        if child.is_file():
            total += child.stat().st_size
            continue
        if not child.is_dir():
            continue
        seen.add(child.name)
        signature = _frozen_signature(child)
        cached = entries.get(child.name)
        if signature is not None and cached and cached.get("signature") == signature:
            total += cached["bytes"]
            continue
        nbytes = _directory_bytes(child)
        total += nbytes
        if signature is not None:
            entries[child.name] = {"signature": signature, "bytes": nbytes}
            dirty = True
        elif child.name in entries:
            del entries[child.name]
            dirty = True
    stale = [name for name in entries if name not in seen]
    if stale:
        for name in stale:
            del entries[name]
        dirty = True
    if dirty:
        _write_index(root, {"schema_version": 1, "entries": entries})
    return total


def require_storage(root, projected_additional_bytes, cap_bytes=1500000000000):
    if type(projected_additional_bytes) is not int or projected_additional_bytes < 0:
        raise ValueError("nonnegative measured storage projection required")
    retained = retained_bytes(root)
    projected = retained + int(projected_additional_bytes * 1.2 + .5)
    if projected > cap_bytes:
        raise ValueError("retained storage plus projection and contingency exceeds cap")
    return {"retained_bytes": retained, "projected_additional_bytes": projected_additional_bytes,
            "contingency_fraction": .2, "projected_peak_bytes": projected, "cap_bytes": cap_bytes}


def write_json_exclusive(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical_json(value) + "\n")


def create_run(root, code_commit, configuration, inputs, snapshots=(), projected_additional_bytes=0):
    """Capture actual consumed files; changed content produces a distinct run."""
    if not re.match(r"[0-9a-f]{40}\Z", code_commit):
        raise ValueError("full code commit required")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = [file_identity(path) for path in sorted(map(str, inputs))]
    copied = [file_identity(path) for path in sorted(map(str, snapshots))]
    identity = {"code_commit": code_commit, "configuration": configuration,
                "inputs": files, "snapshots": copied}
    digest = digest_json(identity)
    destination = root / f"{code_commit[:12]}-{digest[:16]}"
    storage = require_storage(root, projected_additional_bytes)
    destination.mkdir(exist_ok=False)
    snapshot_dir = destination / "snapshots"
    snapshot_dir.mkdir()
    for index, item in enumerate(copied):
        output = snapshot_dir / f"{index:03d}-{Path(item['path']).name}"
        shutil.copyfile(item["path"], output)
        if sha256_file(output) != item["sha256"]:
            raise ValueError("snapshot changed during copy")
    write_json_exclusive(destination / "run.json", {"schema_version": 1, "identity": identity,
        "identity_digest": digest, "storage": storage})
    return destination


def read_run(path):
    result = json.loads((Path(path) / "run.json").read_text())
    canonical_json(result)
    if result.get("schema_version") != 1 or digest_json(result["identity"]) != result["identity_digest"]:
        raise ValueError("run identity mismatch")
    return result


def validate_task(task):
    required = {"name", "command", "environment", "outputs", "inputs", "resources", "projected_bytes"}
    if not isinstance(task, dict) or set(task) != required or not re.match(r"[a-z0-9][a-z0-9_-]*\Z", task["name"]):
        raise ValueError("invalid task definition")
    if not task["command"] or any(not isinstance(x, str) or not x for x in task["command"]):
        raise ValueError("task needs an explicit command argument vector")
    command = task["command"]
    if len(command) >= 3 and command[1] == "-c" and re.match(r"python(?:[0-9]+(?:\.[0-9]+)?)?\Z", Path(command[0]).name):
        try:
            compile(command[2], "compute-task", "exec")
        except SyntaxError as error:
            raise ValueError(f"invalid Python task script at line {error.lineno}: {error.msg}") from error
    if not isinstance(task["environment"], dict) or any(not isinstance(v, str) for v in task["environment"].values()):
        raise ValueError("task environment must contain strings")
    Resources(**task["resources"])
    if type(task["projected_bytes"]) is not int or task["projected_bytes"] < 0:
        raise ValueError("task requires a retained-size projection")
    paths = []
    for output in task["outputs"]:
        path = Path(output["path"])
        if path.is_absolute() or ".." in path.parts or output["kind"] not in ("file", "npz", "hdf5", "manifest"):
            raise ValueError("invalid task output declaration")
        paths.append(str(path))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("unique expected task outputs required")
    for item in task["inputs"]:
        if set(item) != {"path", "bytes", "sha256"} or not Path(item["path"]).is_absolute():
            raise ValueError("invalid input identity")
    canonical_json(task)
    return task


def define_task(run, task):
    """Reserve a deterministic task name; task definitions cannot be replaced."""
    read_run(run)
    validate_task(task)
    directory = Path(run) / "tasks"
    directory.mkdir(exist_ok=True)
    destination = directory / f"{task['name']}.json"
    write_json_exclusive(destination, task)
    return destination
