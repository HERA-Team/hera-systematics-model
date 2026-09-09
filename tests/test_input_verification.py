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
    failure = json.loads(report.read_text())
    assert not failure["passed"]
    assert failure["final_mismatch"] == {"path": str(source.resolve()),
        "stamp_fields": [], "identity_fields": ["sha256"],
        "expected": first, "measured": {**first, "sha256": "different"}}
    report = tmp_path / "child-failure.json"
    with pytest.raises(RuntimeError, match="child failed"):
        with VerifiedInputs(report) as inputs:
            inputs.identity(source)
            raise RuntimeError("child failed")
    assert json.loads(report.read_text())["failure"] == "child failed"


def test_final_metadata_mismatch_identifies_input_and_fields(tmp_path, monkeypatch):
    import hera_systematics_model.input_verification as module

    source, report = tmp_path / "source", tmp_path / "verification.json"
    source.write_text("input")
    actual = module.file_stamp
    calls = 0

    def changed_stamp(path):
        nonlocal calls
        calls += 1
        stamp = actual(path)
        return stamp if calls <= 2 else (*stamp[:-1], stamp[-1] + 1)

    monkeypatch.setattr(module, "file_stamp", changed_stamp)
    with pytest.raises(ValueError, match=r"source \(ctime_ns\)"):
        with VerifiedInputs(report) as inputs:
            inputs.identity(source)
    failure = json.loads(report.read_text())
    assert failure["final_mismatch"]["path"] == str(source.resolve())
    assert failure["final_mismatch"]["stamp_fields"] == ["ctime_ns"]
    assert failure["final_mismatch"]["identity_fields"] == []


def test_accepted_inventory_is_checked_before_first_consumption(tmp_path):
    from hera_systematics_model.configuration import file_identity, digest_json

    source, other = tmp_path / "source", tmp_path / "other"
    source.write_text("original")
    other.write_text("another")
    expected = [file_identity(source)]
    report = tmp_path / "passed.json"
    with VerifiedInputs(report, expected_identities=expected) as inputs:
        assert inputs.identity(source) == expected[0]
    assert json.loads(report.read_text())["expected_inventory_digest"] == digest_json(expected)
    source.write_text("modified")
    report = tmp_path / "changed.json"
    with pytest.raises(ValueError, match="differs from accepted"):
        with VerifiedInputs(report, expected_identities=expected) as inputs:
            inputs.identity(source)
    assert not json.loads(report.read_text())["passed"]
    report = tmp_path / "unlisted.json"
    with pytest.raises(ValueError, match="absent from accepted"):
        with VerifiedInputs(report, expected_identities=expected) as inputs:
            inputs.identity(other)
    assert not json.loads(report.read_text())["passed"]


def test_expected_inventory_rejects_duplicates_and_invalid_hashes(tmp_path):
    from hera_systematics_model.configuration import file_identity

    source = tmp_path / "source"
    source.write_text("input")
    identity = file_identity(source)
    for entries in ([], [identity, identity], [{**identity, "sha256": "invalid"}],
                    [{**identity, "bytes": -1}], [{**identity, "path": "relative"}]):
        with pytest.raises(ValueError):
            VerifiedInputs(tmp_path / "report.json", expected_identities=entries)
    assert not (tmp_path / "report.json").exists()
