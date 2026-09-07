import json

import numpy as np
import pytest

nbformat = pytest.importorskip("nbformat")

from hera_systematics_model.notebook_averaging import instrument_averaging, load_native_grid, shared_time_slice


def grid_file(tmp_path, policy="shared"):
    path = tmp_path / "grid.json"
    times = 2459000. + np.arange(56) * 10 / 86400
    path.write_text(json.dumps({"schema_version": 1, "policy": policy,
        "native_time_jd": times.tolist(), "anchor_jd": times[0] - 5 / 86400,
        "window_seconds": 280., "native_samples_per_window": 28, "sources": [{"sha256": "abc"}]}))
    return path


def test_shared_grid_changes_only_explicit_boundary_and_exports_measured_state(tmp_path):
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "ORed_flags = flags.any(axis=0)\ntslice = slice(1, 30)\nkeep = 5\n")])
    changed, evidence = instrument_averaging(original, grid_file(tmp_path))
    assert original.cells[0].source.endswith("keep = 5\n")
    assert "slice(1, 30)" in original.cells[0].source
    assert "ORed_flags = flags.any(axis=0)" in changed.cells[1].source
    assert "shared_time_slice(single_bl_times" in changed.cells[1].source
    assert "[d.times for d in deint_filt_data]" in changed.cells[-1].source
    assert "interleaved_uvp.time_avg_array" in changed.cells[-1].source
    assert evidence["boundary_assignment_replaced"]
    for cell in changed.cells:
        compile(cell.source, "instrumented", "exec")


def test_retained_boundaries_remain_unmodified_and_time_mismatch_fails(tmp_path):
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("tslice = slice(1, 30)")])
    changed, evidence = instrument_averaging(original, grid_file(tmp_path, "retained"))
    assert changed.cells[1].source == original.cells[0].source
    assert not evidence["boundary_assignment_replaced"]
    assert shared_time_slice([1, 2, 3], [1, 2, 3]) == slice(0, 3)
    with pytest.raises(ValueError, match="reference arc"):
        shared_time_slice([1, 3], [1, 2, 3])
    path = grid_file(tmp_path)
    value = json.loads(path.read_text())
    value["native_time_jd"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete shared"):
        load_native_grid(path)


def test_unrelated_ipython_magic_is_preserved(tmp_path):
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("%matplotlib inline"),
        nbformat.v4.new_code_cell("tslice = slice(1, 30)")])
    changed, _ = instrument_averaging(original, grid_file(tmp_path))
    assert changed.cells[0].source == "%matplotlib inline"
