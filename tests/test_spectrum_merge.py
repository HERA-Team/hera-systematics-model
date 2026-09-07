import h5py
import numpy as np
import pytest

from hera_systematics_model.artifacts import read_artifact
from hera_systematics_model.spectrum_merge import merge_spectra
from test_spectrum_layout import make_spectrum


def test_merge_orders_physical_baselines_and_preserves_all_rows_and_metadata(tmp_path):
    paths = [make_spectrum(tmp_path / "a.h5"), make_spectrum(tmp_path / "b.h5", baseline=100191, start=11., rows=4)]
    result = tmp_path / "merged.h5"
    metadata = merge_spectra(paths, result, [106134106134, 100191100191], block_bytes=100)
    assert metadata["counts"] == {"Nbls": 2, "Nblpairs": 2, "Nbltpairs": 7, "Ntpairs": 5, "Ntimes": 10}
    arrays, saved = read_artifact(result.with_suffix(".merge.npz"), "spectral-merge")
    assert saved == metadata and saved["passed"]
    with h5py.File(result) as output:
        group = output["stokespol/interleave_averaged"]
        np.testing.assert_equal(group["bl_array"][()], [100191, 106134])
        assert "frf_losses" not in output["header"].attrs
        assert output["header"].attrs["hera_pspec.version"] == "0.4.3"
        offset = 0
        for index, path in enumerate(paths[::-1]):
            with h5py.File(path) as source:
                original = source["stokespol/interleave_averaged"]
                rows = original.attrs["Nbltpairs"]
                for spw in range(14):
                    np.testing.assert_equal(group[f"data_spw{spw}"][offset:offset + rows], original[f"data_spw{spw}"][()])
                np.testing.assert_equal(group["label_1_array"][:, offset:offset + rows], original["label_1_array"][()])
                attrs = [item for item in saved["source_attributes"] if item["input_index"] == index]
                frf = next(item for item in attrs if item["attribute"] == "header/frf_losses")
                np.testing.assert_equal(arrays[frf["array"]], source["header"].attrs["frf_losses"])
                offset += rows
    with pytest.raises(FileExistsError):
        merge_spectra(paths, result, [106134106134, 100191100191])


@pytest.mark.parametrize("expected", [[106134106134], [106134106134, 100191100191, 100110100110]])
def test_merge_requires_the_complete_declared_inventory(tmp_path, expected):
    paths = [make_spectrum(tmp_path / "a.h5"), make_spectrum(tmp_path / "b.h5", baseline=100191)]
    result = tmp_path / "merged.h5"
    with pytest.raises(ValueError, match="inventory differs"):
        merge_spectra(paths, result, expected)
    assert not result.exists()
    assert not result.with_suffix(".merge.json").exists()


def test_merge_rejects_duplicate_baselines_and_changed_conventions(tmp_path):
    paths = [make_spectrum(tmp_path / "a.h5"), make_spectrum(tmp_path / "b.h5")]
    with pytest.raises(ValueError, match="duplicate physical baseline"):
        merge_spectra(paths, tmp_path / "duplicate.h5", [106134106134])
    with h5py.File(paths[1], "r+") as handle:
        handle["stokespol/interleave_averaged"].attrs["norm_units"] = "different"
    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_spectra(paths, tmp_path / "units.h5", [106134106134])


def test_reopened_verification_rejects_corrupted_copy(tmp_path, monkeypatch):
    import hera_systematics_model.spectrum_merge as module

    path = make_spectrum(tmp_path / "source.h5")
    original_transfer = module.transfer_entries

    def corrupt(entries, output, group, block_bytes, inputs, verify=False):
        original_transfer(entries, output, group, block_bytes, inputs, verify)
        if not verify:
            output[group]["data_spw6"][1, 0, 0] += 1

    monkeypatch.setattr(module, "transfer_entries", corrupt)
    destination = tmp_path / "merged.h5"
    with pytest.raises(ValueError, match="metadata mismatch: data_spw6"):
        merge_spectra([path], destination, [106134106134])
    assert destination.exists()
    assert not destination.with_suffix(".merge.json").exists()
