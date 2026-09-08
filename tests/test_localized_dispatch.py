import json

import pytest

from hera_systematics_model.configuration import AnalysisConfig, digest_json, file_identity
from hera_systematics_model.localized_dispatch import collect_localized_tasks


@pytest.fixture
def inventories(tmp_path, monkeypatch):
    parents = []
    for spw in (0, 1):
        run, name = tmp_path / str(spw), f"inventory-{spw}"
        source = run / "products" / name
        (source / "configurations").mkdir(parents=True)
        entries = []
        for axis, count in (("group", 2), ("delay", 3)):
            for index in range(count):
                for method in ("complete", "masked"):
                    slice_name = f"spw-{spw:02d}-{axis}-{index:04d}-{method}"
                    config = AnalysisConfig(representations=["linear"], methods=[method],
                                            include_kernel=False, **{axis: index}).as_dict()
                    path = source / "configurations" / (slice_name + ".json")
                    path.write_text(json.dumps(config))
                    entries.append({"name": slice_name, "axis": axis, "index": index,
                        "method": method, "identity": {"spw": spw},
                        "feature_shape": [1, 3] if axis == "group" else [2, 1],
                        "configuration_file": file_identity(path), "configuration": config,
                        "configuration_digest": digest_json(config),
                        "feature_partition_axis": "delay" if axis == "group" else "group",
                        "feature_guard_bins": 2 if axis == "group" else 1})
        (source / "inventory.json").write_text(json.dumps({"schema_version": 1, "spw": spw,
            "selected_representation": "linear", "tasks": entries}))
        (source / "verification.json").write_text(json.dumps({"passed": True, "spw": spw,
            "groups": 2, "delays": 3, "tasks": 10}))
        parents.append({"spw": spw, "run": str(run), "task": name})
    monkeypatch.setattr("hera_systematics_model.localized_dispatch.verify_task_acceptance",
                        lambda run, name: {"verified": str(run / name)})
    return parents


def test_collect_is_deterministic_and_requires_all_windows(inventories):
    result = collect_localized_tasks(inventories, (0, 1))
    assert result == collect_localized_tasks(inventories[::-1], (1, 0))
    assert result["task_count"] == 20
    assert len({item["slice"] for item in result["tasks"]}) == 20
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        collect_localized_tasks(inventories)
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        collect_localized_tasks([inventories[0]] * 2, (0, 1))


@pytest.mark.parametrize("change", ["missing", "duplicate", "guard", "schema", "spw", "shape"])
def test_rejects_incomplete_or_altered_inventory(inventories, change):
    parent = inventories[0]
    from pathlib import Path
    path = Path(parent["run"]) / "products" / parent["task"] / "inventory.json"
    data = json.loads(path.read_text())
    if change == "missing":
        data["tasks"].pop()
    elif change == "duplicate":
        data["tasks"][-1] = data["tasks"][0]
    elif change == "guard":
        data["tasks"][0]["feature_guard_bins"] = 0
    elif change == "shape":
        data["tasks"][0]["feature_shape"] = [3, 1]
    else:
        data["schema_version" if change == "schema" else "spw"] = 9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        collect_localized_tasks(inventories, (0, 1))


def test_failed_acceptance_prevents_collection(inventories, monkeypatch):
    def fail(run, name):
        raise ValueError("scheduler job or step did not complete successfully")
    monkeypatch.setattr("hera_systematics_model.localized_dispatch.verify_task_acceptance", fail)
    with pytest.raises(ValueError, match="did not complete"):
        collect_localized_tasks(inventories, (0, 1))
