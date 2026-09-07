import json
from pathlib import Path
import shutil

import pytest

import hera_systematics_model.spectral_batch as batch
from hera_systematics_model.configuration import capture_runtime, file_identity
from hera_systematics_model.window_membership import WindowMemberships
from test_spectral_receipts import products


@pytest.fixture(autouse=True)
def synthetic_import_context(monkeypatch):
    monkeypatch.setattr(batch, "capture_imports", lambda names: {"runtime": capture_runtime(), "imports": {}})


def setup_inventory(tmp_path):
    original, identity, native = products(tmp_path)
    auto = tmp_path / "zen.LST.baseline.0_0.sum.uvh5"
    auto.write_text("synthetic auto")
    grid = tmp_path / "native_grid"
    grid.write_text(json.dumps({"schema_version": 1, "policy": "shared", "native_time_jd": native.tolist(),
        "native_samples_per_window": 2, "anchor_jd": 10., "window_seconds": 86400., "sources": ["synthetic"]}))
    config = {"EFIELD_HEALPIX_BEAM_FILE": str(tmp_path / "beam"), "FR_SPECTRA_FILE": str(tmp_path / "fringe_rate_cache")}
    value = {"schema_version": 1, "code_commit": "a" * 40, "notebook": str(tmp_path / "notebook"),
        "native_grid": str(grid), "configuration": config, "baselines": [{"baseline_pair_code": 106134106134,
            "file": str(tmp_path / "single_baseline"), "auto": str(auto)}]}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(value))
    return path, value, original


def synthetic_execution(original, value, missing=False):
    def execute(tasks, directory, *args, on_completion):
        directory.mkdir()
        records = []
        for task in tasks:
            folder = directory / task["name"]
            folder.mkdir()
            if not missing:
                for name in ("spectrum.pspec.h5", "spectrum.tavg.pspec.h5", "instrumented.ipynb", "spectrum.ipynb"):
                    shutil.copyfile(original / name, folder / name)
                membership = WindowMemberships.load(original / "window-memberships.npz")
                membership.metadata["spectrum_source"] = file_identity(folder / "spectrum.pspec.h5")
                membership.save(folder / "window-memberships.npz")
            execution = {"parameters": {**value["configuration"], "SINGLE_BL_FILE": value["baselines"][0]["file"],
                "OUT_PSPEC_FILE": str(folder / "spectrum.pspec.h5"), "OUT_TAVG_PSPEC_FILE": str(folder / "spectrum.tavg.pspec.h5")},
                "notebook": file_identity(value["notebook"]), "single_baseline": file_identity(value["baselines"][0]["file"])}
            (folder / "execution.json").write_text(json.dumps(execution))
            (folder / "import-runtime.json").write_text(json.dumps({"runtime": capture_runtime()}))
            records.append(on_completion({"name": task["name"], "status": "exited_zero", "exit_code": 0}, folder))
        return {"records": records, "all_commands_exited_zero": True}
    return execute


def test_batch_accepts_products_then_reuses_without_launch_or_overwrite(tmp_path, monkeypatch):
    path, value, original = setup_inventory(tmp_path)
    monkeypatch.setattr(batch, "run_bounded_commands", synthetic_execution(original, value))
    first = batch.run_spectral_batch(path, tmp_path / "first", 1, 2, 16384)
    assert first["passed"]
    entry = first["baselines"][0]
    old = Path(entry["directory"])
    stamps = {p.name: p.stat().st_mtime_ns for p in old.iterdir()}
    value["baselines"][0]["reuse"] = str(old)
    reuse_inventory = tmp_path / "reuse.json"
    reuse_inventory.write_text(json.dumps(value))

    def no_commands(*args):
        raise AssertionError("verified reuse must not execute notebooks")

    monkeypatch.setattr(batch, "run_bounded_commands", no_commands)
    second = batch.run_spectral_batch(reuse_inventory, tmp_path / "second", 1, 2, 16384)
    assert second["passed"] and second["baselines"][0]["reused"]
    assert stamps == {p.name: p.stat().st_mtime_ns for p in old.iterdir()}
    value["configuration"]["taper"] = "changed"
    reuse_inventory.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different code, inputs or configuration"):
        batch.run_spectral_batch(reuse_inventory, tmp_path / "changed", 1, 2, 16384)


def test_zero_exit_without_spectral_payload_is_a_failed_batch(tmp_path, monkeypatch):
    path, value, original = setup_inventory(tmp_path)
    monkeypatch.setattr(batch, "run_bounded_commands", synthetic_execution(original, value, missing=True))
    result = batch.run_spectral_batch(path, tmp_path / "batch", 1, 2, 16384)
    assert not result["passed"]
    assert "output product" in result["baselines"][0]["verification_error"]
    assert (tmp_path / "batch/input-verification.json").exists()


def test_expected_runtime_is_captured_after_scientific_imports(tmp_path, monkeypatch):
    path, value, original = setup_inventory(tmp_path)
    primed = []
    actual = capture_runtime

    def runtime():
        result = actual()
        if primed:
            result["distributions"] = [*result["distributions"], ["bundled-dependency", "1.0"]]
        return result

    def imports(names):
        assert names == batch.SPECTRAL_MODULES
        primed.append(True)
        return {"runtime": runtime(), "imports": {}}

    monkeypatch.setattr(batch, "capture_imports", imports)
    monkeypatch.setattr(batch, "capture_runtime", runtime)
    monkeypatch.setitem(globals(), "capture_runtime", runtime)
    monkeypatch.setattr(batch, "run_bounded_commands", synthetic_execution(original, value))
    result = batch.run_spectral_batch(path, tmp_path / "batch", 1, 2, 16384)
    assert result["passed"] and primed == [True]
    saved = json.loads((tmp_path / "batch/import-runtime.json").read_text())
    assert ["bundled-dependency", "1.0"] in saved["runtime"]["distributions"]
