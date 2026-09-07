from dataclasses import replace
import json

import numpy as np
import pytest

from hera_systematics_model.cli import main
from hera_systematics_model.configuration import AnalysisConfig, digest_json, file_identity
from hera_systematics_model.evaluation import Evaluation
from hera_systematics_model.localized import localized_inventory
from test_samples import paired


@pytest.fixture
def full_fit(tmp_path, paired):
    ids = np.arange(120)
    expanded = {name: np.repeat(getattr(paired, name)[:1], len(ids), axis=0)
                for name in ("corrupted", "ideal", "pn", "valid", "weights")}
    samples = replace(paired, **expanded, window_ids=ids,
        time_jd=2450000. + (ids + .5) * 270. / 86400., lst_rad=(6.1 + ids * .02) % (2 * np.pi))
    source, fit, config = (tmp_path / name for name in ("samples.npz", "fit.npz", "config.json"))
    samples.save(source)
    config.write_text(json.dumps({"max_rank": 0, "include_kernel": False}))
    assert main(["fit", "--samples", str(source), "--config", str(config), "--output", str(fit)]) == 0
    return source, fit, samples


def test_inventory_covers_every_physical_slice_and_each_method(full_fit, tmp_path):
    source, fit, samples = full_fit
    result = localized_inventory(source, fit)
    assert result == localized_inventory(source, fit)
    assert result["inputs"] == [file_identity(p) for name in (source, fit)
                                for p in (name, name.with_suffix(".json"))]
    tasks = result["tasks"]
    assert len(tasks) == 2 * sum(samples.corrupted.shape[1:])
    assert len({task["name"] for task in tasks}) == len(tasks)
    assert result["physical_window_ids"] == list(range(120))
    for axis, count in (("group", len(samples.group_ids)), ("delay", len(samples.delay_s))):
        selected = [task for task in tasks if task["axis"] == axis]
        assert [(task["index"], task["method"]) for task in selected] == [
            (index, method) for index in range(count) for method in ("complete", "masked")]
        for task in selected:
            config = AnalysisConfig(**task["configuration"])
            assert config.max_rank == 20 and config.guard == 12 and not config.include_kernel
            assert config.methods == [task["method"]]
            assert config.representations == [result["selected_representation"]]
            assert task["configuration_digest"] == digest_json(config.as_dict())
            assert task["feature_partition_axis"] == ("delay" if axis == "group" else "group")
            assert task["feature_guard_bins"] == (2 if axis == "group" else 1)
            coordinate = "group_ids" if axis == "group" else "delay_s"
            assert task["identity"][coordinate] == [getattr(samples, coordinate)[task["index"]]]
    output = tmp_path / "inventory.json"
    assert main(["inventory-localized", "--samples", str(source), "--fit", str(fit), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == result
    with pytest.raises(SystemExit):
        main(["inventory-localized", "--samples", str(source), "--fit", str(fit), "--output", str(output)])


def test_inventory_rejects_changed_sample_identity(full_fit, tmp_path):
    _, fit, samples = full_fit
    changed = tmp_path / "changed.npz"
    samples.corrupted[:] += 1
    samples.save(changed)
    with pytest.raises(ValueError, match="matched complete full-plane"):
        localized_inventory(changed, fit)


@pytest.mark.parametrize("configuration", [{"group": 0}, {"delay": 0}, {"guard": 8}, {"region": "horizon"}])
def test_inventory_rejects_nonprimary_configuration(full_fit, tmp_path, configuration):
    source, fit, _ = full_fit
    product = Evaluation.load(fit)
    product.metadata["configuration"].update(configuration)
    other = tmp_path / "other.npz"
    product.save(other)
    with pytest.raises(ValueError, match="matched complete full-plane"):
        localized_inventory(source, other)
