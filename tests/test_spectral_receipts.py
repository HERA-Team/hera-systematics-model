import json

import h5py
import numpy as np
import pytest

from hera_systematics_model.configuration import digest_json, file_identity
from hera_systematics_model.spectral_receipts import accept_spectral_entry, reuse_spectral_entry
from hera_systematics_model.window_membership import WindowMemberships
from test_spectrum_layout import make_spectrum


def products(tmp_path):
    directory = tmp_path / "products"
    directory.mkdir()
    inputs = {}
    for role in ("notebook", "single_baseline", "auto", "native_grid", "beam", "fringe_rate_cache"):
        path = tmp_path / role
        path.write_text(role)
        inputs[role] = file_identity(path)
    raw = make_spectrum(directory / "spectrum.pspec.h5")
    averaged = make_spectrum(directory / "spectrum.tavg.pspec.h5", rows=1)
    with h5py.File(averaged, "r+") as handle:
        handle.move("stokespol/interleave_averaged", "stokespol/time_and_interleave_averaged")
    with h5py.File(raw) as handle:
        times = handle["stokespol/interleave_averaged/time_avg_array"][()]
    native = np.array([[10., 10.1], [11., 11.1], [12., 12.1]]).ravel()
    WindowMemberships(np.repeat("6_34:6_34", 3), times, np.arange(3), np.arange(6).reshape(3, 2), native,
        {"spectrum_source": file_identity(raw), "native_time_source": inputs["native_grid"],
         "n_interleaves": 2, "averaging_configuration": {"width": 2}}).save(directory / "window-memberships.npz")
    identity = {"code_commit": "a" * 40, "source_digest": "b" * 64, "configuration": {},
        "runtime_digest": digest_json({"source_digest": "b" * 64}),
        "inputs": inputs, "baseline_pair_code": 106134106134, "native_grid_digest": digest_json(native.tolist())}
    execution = {"notebook": inputs["notebook"], "single_baseline": inputs["single_baseline"],
        "parameters": {"SINGLE_BL_FILE": inputs["single_baseline"]["path"],
            "OUT_PSPEC_FILE": str(raw), "OUT_TAVG_PSPEC_FILE": str(averaged)}}
    for name, contents in (("execution.json", execution),
        ("import-runtime.json", {"runtime": {"source_digest": "b" * 64}}),
        ("instrumented.ipynb", {"cells": []}), ("spectrum.ipynb", {"cells": []})):
        (directory / name).write_text(json.dumps(contents))
    return directory, identity, native


def test_reuse_checks_exact_inputs_and_payloads_without_overwriting(tmp_path):
    directory, identity, native = products(tmp_path)
    receipt = accept_spectral_entry(directory, identity, native, 2)
    stamps = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
    assert reuse_spectral_entry(directory, identity, native, 2) == receipt
    assert stamps == {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
    with pytest.raises(ValueError, match="different code"):
        reuse_spectral_entry(directory, {**identity, "configuration": {"taper": "changed"}}, native, 2)
    with h5py.File(directory / "spectrum.tavg.pspec.h5", "r+") as handle:
        handle["stokespol/time_and_interleave_averaged/data_spw0"][0, 1, 0] += 1
    with pytest.raises(ValueError, match="products changed"):
        reuse_spectral_entry(directory, identity, native, 2)


def test_changed_input_invalidates_previously_accepted_products(tmp_path):
    directory, identity, native = products(tmp_path)
    accept_spectral_entry(directory, identity, native, 2)
    (tmp_path / "beam").write_text("modified beam")
    with pytest.raises(ValueError, match="consumed input changed"):
        reuse_spectral_entry(directory, identity, native, 2)


def test_missing_window_and_notebook_error_cannot_be_accepted(tmp_path):
    directory, identity, native = products(tmp_path)
    (directory / "spectrum.ipynb").write_text(json.dumps({"cells": [{"outputs": [{"output_type": "error"}]}]}))
    with pytest.raises(ValueError, match="notebook contains an error"):
        accept_spectral_entry(directory, identity, native, 2)
    assert not (directory / "spectrum-verification.json").exists()
    with h5py.File(directory / "spectrum.pspec.h5", "r+") as handle:
        del handle["stokespol/interleave_averaged/data_spw13"]
    with pytest.raises(ValueError, match="datasets are absent"):
        accept_spectral_entry(directory, identity, native, 2)
