import json
import os

import pytest

from hera_systematics_model.input_verification import VerifiedInputs, input_identity


def test_batch_hashes_each_input_twice_and_never_reuses_context(tmp_path, monkeypatch):
    import hera_systematics_model.input_verification as module

    source, report = tmp_path / "source", tmp_path / "verification.json"
    source.write_text("visibility")
    calls = []
    actual = module.file_identity

    def measured(path):
        calls.append(str(path))
        return actual(path)

    monkeypatch.setattr(module, "file_identity", measured)
    with VerifiedInputs(report) as inputs:
        first = input_identity(source, inputs)
        for _ in range(5):
            assert inputs.identity(source) == first
        first["sha256"] = "changed copy"
        assert inputs.identity(source)["sha256"] != first["sha256"]
        assert len(calls) == 1
    assert len(calls) == 2
    assert json.loads(report.read_text())["passed"]
    with pytest.raises(ValueError, match="active"):
        inputs.identity(source)
    with pytest.raises(ValueError, match="reused"):
        with inputs:
            pass


def test_changed_input_fails_even_when_mtime_and_size_are_restored(tmp_path):
    source, report = tmp_path / "source", tmp_path / "verification.json"
    source.write_text("before")
    with pytest.raises(ValueError, match="changed"):
        with VerifiedInputs(report) as inputs:
            inputs.identity(source)
            before = source.stat()
            source.write_text("after!")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            inputs.identity(source)
    assert not json.loads(report.read_text())["passed"]


def test_final_hash_and_child_failures_cannot_produce_acceptance(tmp_path, monkeypatch):
    import hera_systematics_model.input_verification as module

    source, report = tmp_path / "source", tmp_path / "verification.json"
    source.write_text("input")
    with pytest.raises(ValueError, match="final batch"):
        with VerifiedInputs(report) as inputs:
            first = inputs.identity(source)
            monkeypatch.setattr(module, "file_identity", lambda path: {**first, "sha256": "different"})
    assert not json.loads(report.read_text())["passed"]
    report = tmp_path / "child-failure.json"
    with pytest.raises(RuntimeError, match="child failed"):
        with VerifiedInputs(report) as inputs:
            inputs.identity(source)
            raise RuntimeError("child failed")
    assert json.loads(report.read_text())["failure"] == "child failed"
