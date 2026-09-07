import pytest
from types import SimpleNamespace

from hera_systematics_model import scheduler
from hera_systematics_model.scheduler import memory_mib, queued_resources, require_resources, submit_task
from hera_systematics_model.production import create_run, define_task
from test_production import task


def test_memory_accounts_for_node_and_cpu_scope():
    assert memory_mib("8Gc", 4) == 32768
    assert memory_mib("64Gn", 8) == 65536
    assert memory_mib("1024M", 8) == 1024
    assert memory_mib("1048576K", 1) == 1024


def test_pending_and_running_reservations_share_the_same_limits():
    active = queued_resources("12|hsm-a-one|8|64Gn\n13|other|80|512G")
    require_resources(active, {"cpus": 8, "memory_mib": 65536, "hours": 24})
    for resources in [{"cpus": 9, "memory_mib": 1}, {"cpus": 1, "memory_mib": 65537}]:
        with pytest.raises(ValueError):
            require_resources(active, resources)
    with pytest.raises(ValueError):
        require_resources(active * 2, {"cpus": 1, "memory_mib": 1})


def test_array_resources_cannot_be_silently_undercounted():
    with pytest.raises(ValueError, match="array"):
        queued_resources("12_[1-4]|hsm-a-one|8|64Gn")


def test_submission_is_single_and_uncertain_dispatch_cannot_retry(tmp_path, monkeypatch):
    run = create_run(tmp_path / "runs", "a" * 40, {}, [])
    define_task(run, task())
    calls = []
    monkeypatch.setattr(scheduler.subprocess, "check_output", lambda *a, **k: "")
    def dispatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
    monkeypatch.setattr(scheduler.subprocess, "run", dispatch)
    first = submit_task(run, "baseline-1", "/usr/bin/python3", tmp_path)
    assert submit_task(run, "baseline-1", "/usr/bin/python3", tmp_path) == first
    assert len(calls) == 1
    assert "--nodes=1" in calls[0] and "--cpus-per-task=1" in calls[0]
    other = task()
    other["name"] = "baseline-2"
    define_task(run, other)
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="lost reply"))
    with pytest.raises(ValueError, match="uncertain"):
        submit_task(run, "baseline-2", "/usr/bin/python3", tmp_path)
    with pytest.raises(ValueError, match="reconciliation"):
        submit_task(run, "baseline-2", "/usr/bin/python3", tmp_path)


def test_dependencies_allow_only_nonoverlapping_predecessors_and_keep_storage_reservations(tmp_path, monkeypatch):
    run = create_run(tmp_path / "runs", "a" * 40, {}, [])
    active = []
    calls = []
    monkeypatch.setattr(scheduler.subprocess, "check_output", lambda *a, **k: "\n".join(active))

    def dispatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", dispatch)
    for index in (1, 2, 3, 4):
        definition = {**task(), "name": f"baseline-{index}"}
        define_task(run, definition)
    submit_task(run, "baseline-1", "/usr/bin/python3", tmp_path)
    active.append("101|hsm-a-one|1|1G")
    submit_task(run, "baseline-2", "/usr/bin/python3", tmp_path)
    active.append("102|hsm-a-two|1|1G")
    with pytest.raises(ValueError, match="aggregate"):
        submit_task(run, "baseline-3", "/usr/bin/python3", tmp_path)
    with pytest.raises(ValueError, match="recorded"):
        submit_task(run, "baseline-3", "/usr/bin/python3", tmp_path, afterok=["999"])
    record = submit_task(run, "baseline-3", "/usr/bin/python3", tmp_path, afterok=["101"])
    assert "--dependency=afterok:101" in calls[-1]
    assert record["afterok"] == ["101"]
    assert [j["job_id"] for j in record["potentially_simultaneous"]] == ["102"]
    assert record["storage"]["projected_additional_bytes"] == 3 * task()["projected_bytes"]
    with pytest.raises(ValueError, match="dependencies changed"):
        submit_task(run, "baseline-3", "/usr/bin/python3", tmp_path)
    active.append("103|hsm-a-three|1|1G")
    with pytest.raises(ValueError, match="aggregate"):
        submit_task(run, "baseline-4", "/usr/bin/python3", tmp_path, afterok=["102"])
