from pathlib import Path

import pytest

from hera_systematics_model.production import Resources, create_run, define_task, read_run, require_storage, retained_bytes, validate_task


def task():
    return {"name": "baseline-1", "command": ["/bin/true"], "environment": {}, "inputs": [],
            "outputs": [{"path": "spectrum.npz", "kind": "npz"}], "projected_bytes": 1000000,
            "resources": {"cpus": 1, "memory_mib": 1024, "hours": 1}}


def test_run_snapshots_dirty_consumed_content_and_never_replaces_it(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    root = tmp_path / "runs"
    run = create_run(root, "a" * 40, {"spws": list(range(14))}, [source], [source])
    with pytest.raises(FileExistsError):
        create_run(root, "a" * 40, {"spws": list(range(14))}, [source], [source])
    source.write_text("value = 2\n")
    changed = create_run(root, "a" * 40, {"spws": list(range(14))}, [source], [source])
    assert changed != run
    assert next((run / "snapshots").iterdir()).read_text() == "value = 1\n"
    assert read_run(run)["identity_digest"] != read_run(changed)["identity_digest"]
    define_task(run, task())
    with pytest.raises(FileExistsError):
        define_task(run, task())


def test_storage_includes_contingency_without_counting_shared_targets(tmp_path):
    external = tmp_path / "shared"
    external.write_bytes(b"1234567890")
    root = tmp_path / "runs"
    root.mkdir()
    (root / "reference").symlink_to(external)
    (root / "retained").write_bytes(b"12345")
    assert retained_bytes(root) == 5
    assert require_storage(root, 10, cap_bytes=17)["projected_peak_bytes"] == 17
    with pytest.raises(ValueError):
        require_storage(root, 10, cap_bytes=16)


@pytest.mark.parametrize("kwargs", [{"cpus": 17}, {"memory_mib": 131073}, {"hours": 25}])
def test_task_resources_cannot_exceed_aggregate_bounds(kwargs):
    with pytest.raises(ValueError):
        Resources(**kwargs)


def test_python_commands_are_compiled_before_task_definition(tmp_path):
    run = create_run(tmp_path / "runs", "a" * 40, {}, [])
    definition = {**task(), "command": ["/usr/bin/python3.12", "-c", "value = 'first\nsecond'"]}
    with pytest.raises(ValueError, match="invalid Python task script"):
        define_task(run, definition)
    assert not (run / "tasks/baseline-1.json").exists()
    definition["command"][-1] = "value = 'first' + chr(10) + 'second'"
    assert validate_task(definition) == definition
