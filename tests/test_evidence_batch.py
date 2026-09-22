import json

import pytest

from hera_systematics_model import evidence_batch
from hera_systematics_model.configuration import file_identity
from test_evidence import products


def fixture(tmp_path, monkeypatch, n=123, zero=False):
    directory = tmp_path / "products" / "sensitivity"
    directory.mkdir(parents=True)
    evaluation, descriptive = products(n=n, zero=zero)
    evaluation.save(directory / "evaluation.npz")
    descriptive.save(directory / "descriptive.npz")
    for name in ("success.json", "verification.json"):
        (directory / name).write_text('{}\n')
    entry = {"run": str(tmp_path), "task": "sensitivity", "job_id": "42"}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "tasks": [entry]}))
    acceptance = {**entry, "passed": True,
                  "success": file_identity(directory / "success.json"),
                  "verification": file_identity(directory / "verification.json")}
    monkeypatch.setattr(evidence_batch, "verify_entry", lambda item: acceptance)
    return catalog, tmp_path / "evidence.json", directory


def test_export_retains_baselines_and_zero_difference(tmp_path, monkeypatch):
    catalog, output, _ = fixture(tmp_path, monkeypatch, zero=True)
    report = evidence_batch.export_catalog(catalog, output)
    row = report["tasks"][0]
    assert report["task_count"] == 1 and not report["selection_repeated"]
    assert row["acceptance"]["job_id"] == "42" and len(row["artifacts"]) == 4
    assert set(row["evidence"]["predictive_loss"]) == {"selected", "zero", "mean"}
    assert row["evidence"]["baseline_comparisons"]["zero"]["baseline_minus_selected"]["mean"] == 0
    assert json.loads(output.read_text()) == report


def test_failed_independent_verification_prevents_export(tmp_path, monkeypatch):
    catalog, output, _ = fixture(tmp_path, monkeypatch)
    def fail(item):
        raise ValueError("producer verification failed")
    monkeypatch.setattr(evidence_batch, "verify_entry", fail)
    with pytest.raises(ValueError, match="producer verification failed"):
        evidence_batch.export_catalog(catalog, output)
    assert not output.exists()


def test_incomplete_evaluation_never_exports_success(tmp_path, monkeypatch):
    catalog, output, _ = fixture(tmp_path, monkeypatch, n=20)
    with pytest.raises(ValueError, match="complete retained"):
        evidence_batch.export_catalog(catalog, output)
    assert not output.exists()


def test_changed_artifact_during_export_is_rejected(tmp_path, monkeypatch):
    catalog, output, directory = fixture(tmp_path, monkeypatch)
    original = evidence_batch.evaluation_evidence
    def changed(*args):
        result = original(*args)
        with (directory / "evaluation.json").open("a") as stream:
            stream.write("\n")
        return result
    monkeypatch.setattr(evidence_batch, "evaluation_evidence", changed)
    with pytest.raises(ValueError, match="changed during export"):
        evidence_batch.export_catalog(catalog, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    catalog, output, _ = fixture(tmp_path, monkeypatch)
    output.write_text("retained")
    with pytest.raises(FileExistsError):
        evidence_batch.export_catalog(catalog, output)
    assert output.read_text() == "retained"


def test_cli_requires_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["evidence_batch", "catalog", "output"])
    with pytest.raises(RuntimeError, match="scheduler allocation"):
        evidence_batch.main()
