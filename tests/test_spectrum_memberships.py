import json

import h5py
import numpy as np
import pytest

from hera_systematics_model.cli import main
from hera_systematics_model.configuration import file_identity
from hera_systematics_model.spectrum_merge import merge_spectra
from hera_systematics_model.spectrum_memberships import merge_membership_files
from hera_systematics_model.spectrum_selection import baseline_pair_identity
from hera_systematics_model.window_membership import WindowMemberships
from test_spectrum_layout import make_spectrum


def make_export(spectrum, path):
    with h5py.File(spectrum) as handle:
        group = handle["stokespol/interleave_averaged"]
        ids = np.array([baseline_pair_identity(code) for code in group["blpair_array"][()]])
        times = group["time_avg_array"][()]
    native = np.array([10., 10.1, 11., 11.1, 12., 12.1])
    result = WindowMemberships(ids, times, np.arange(3), np.arange(6).reshape(3, 2), native,
        {"spectrum_source": file_identity(spectrum), "native_time_source": {"sha256": "native"},
         "n_interleaves": 2, "averaging_configuration": {"samples_per_stream": 1}})
    result.save(path)
    return path


def test_cli_merges_reordered_spectra_and_exact_memberships(tmp_path, capsys):
    paths = [make_spectrum(tmp_path / "a.h5"), make_spectrum(tmp_path / "b.h5", baseline=100191)]
    exports = [make_export(path, path.with_suffix(".windows.npz")) for path in paths]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema_version": 1, "inputs": [
        {"spectrum": str(path), "memberships": str(export), "baseline_pair_code": code}
        for path, export, code in zip(paths, exports, [106134106134, 100191100191])]}))
    output, memberships = tmp_path / "merged.h5", tmp_path / "windows.npz"
    assert main(["spectra", "merge", "--inventory", str(inventory), "--output", str(output),
                 "--memberships-output", str(memberships)]) == 0
    result = WindowMemberships.load(memberships)
    np.testing.assert_equal(result.baseline_ids, ["0_91:0_91"] * 3 + ["6_34:6_34"] * 3)
    np.testing.assert_equal(result.native_ids, np.tile(np.arange(6).reshape(3, 2), (2, 1)))
    assert result.metadata["spectrum_source"] == file_identity(output)
    assert main(["verify", str(output.with_suffix(".merge.npz"))]) == 0
    assert '"passed": true' in capsys.readouterr().out


def test_native_exports_must_be_bound_to_each_exact_spectral_source(tmp_path):
    paths = [make_spectrum(tmp_path / "a.h5"), make_spectrum(tmp_path / "b.h5", baseline=100191)]
    exports = [make_export(path, path.with_suffix(".windows.npz")) for path in paths]
    merged = tmp_path / "merged.h5"
    merge_spectra(paths, merged, [106134106134, 100191100191])
    with pytest.raises(ValueError, match="do not cover"):
        merge_membership_files(merged, exports[:1], tmp_path / "missing.npz")
    unrelated = make_spectrum(tmp_path / "different.h5")
    export = make_export(unrelated, tmp_path / "different.npz")
    with pytest.raises(ValueError, match="different source"):
        merge_membership_files(merged, [export, exports[1]], tmp_path / "wrong.npz")
    with h5py.File(merged, "r+") as handle:
        handle["stokespol/interleave_averaged/data_spw0"][1, 1, 1] += 1
    with pytest.raises(ValueError, match="verified merge receipt"):
        merge_membership_files(merged, exports, tmp_path / "corrupt.npz")
    with pytest.raises(SystemExit) as error:
        main(["verify", str(merged.with_suffix(".merge.npz"))])
    assert error.value.code == 2
