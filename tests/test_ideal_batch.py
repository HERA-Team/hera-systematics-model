import json

import numpy as np
import pytest

from hera_systematics_model.ideal_batch import prepare_chunks, construct_batch
from hera_systematics_model.visibility_inventory import inventory_visibilities


def test_chunk_selection_is_physical_deterministic_and_has_unique_writers(tmp_path):
    from test_visibility_inventory import create

    references = [tmp_path / "second.uvh5", tmp_path / "first.uvh5"]
    for path in references:
        create(path)
    inventory = inventory_visibilities(references)
    entries = inventory["files"]
    chunks = {"schema_version": 1, "chunks": [{"reference": entry["path"], "output": f"chunk-{i}.uvh5",
        "sources": ["source.uvh5"], "baseline_inventory": entry["baseline_inventory"]} for i, entry in enumerate(entries)]}
    result = prepare_chunks(chunks, inventory, [[0, 1]])
    assert [c["reference"] for c in result] == sorted(map(str, references))
    assert all(c["selected_baselines"] == [(0, 1)] for c in result)
    with pytest.raises(ValueError, match="lack"):
        prepare_chunks(chunks, inventory, [[0, 99]])
    chunks["chunks"][1]["output"] = chunks["chunks"][0]["output"]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_chunks(chunks, inventory)


def test_batch_acceptance_requires_reverified_inputs_and_retains_chunk_checks(tmp_path):
    pytest.importorskip("pyuvdata")
    from test_ideal_uvh5 import visibility

    source = visibility(2459000. + np.arange(4) * 10 / 86400)
    source.data_array[:] = 1 + 2j
    source_path = tmp_path / "source.uvh5"
    source.write_uvh5(source_path)
    reference = visibility(2459000. + np.array([15., 25.]) / 86400)
    reference.flag_array[:] = True
    reference_path = tmp_path / "reference.uvh5"
    reference.write_uvh5(reference_path)
    inventory = inventory_visibilities([reference_path])
    chunks = {"schema_version": 1, "chunks": [{"reference": str(reference_path), "output": "ideal.uvh5",
        "sources": [str(source_path)], "baseline_inventory": inventory["files"][0]["baseline_inventory"]}]}
    mapping = {"0_1": {"reference_pair": [0, 1], "source_pair": [0, 1], "stored_pair": [0, 1],
                       "conjugate": False, "exclusion": None}}
    output = tmp_path / "products"
    from hera_systematics_model.configuration import file_identity

    expected = [file_identity(source_path), file_identity(reference_path)]
    result = construct_batch(prepare_chunks(chunks, inventory), mapping, output, expected_identities=expected)
    assert result["passed"] and result["chunks"][0]["totals"]["valid_cells"] == 16
    assert json.loads((output / "input-verification.json").read_text())["hash_checks_per_input"] == 2
    assert (output / "ideal.verification.json").exists()
    with pytest.raises(FileExistsError):
        construct_batch(prepare_chunks(chunks, inventory), mapping, output)
    source_path.write_bytes(source_path.read_bytes() + b"changed source payload")
    changed_output = tmp_path / "changed-input"
    with pytest.raises(ValueError, match="differs from accepted"):
        construct_batch(prepare_chunks(chunks, inventory), mapping, changed_output, expected_identities=expected)
    assert not (changed_output / "ideal.uvh5").exists()
    assert not json.loads((changed_output / "input-verification.json").read_text())["passed"]
