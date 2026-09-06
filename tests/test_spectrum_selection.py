import h5py
import numpy as np
import pytest

from hera_systematics_model.spectrum_selection import SpectralSelection


def test_merged_selection_preserves_physical_rows_polarization_and_geometry(tmp_path):
    with h5py.File(tmp_path / "merged.h5", "w") as f:
        g = f.create_group("spectrum")
        g["blpair_array"] = [106134106134, 100110100110, 106134106134]
        g["bl_array"] = [100110, 106134]
        g["bl_vecs"] = np.arange(6).reshape(2, 3)
        g["time_avg_array"] = [10., 11., 12.]
        g.attrs["polpair_array"] = [2121, 2222]
        g.attrs["scalar_array"] = np.arange(28).reshape(14, 2)
        g["data_spw0"] = np.arange(24).reshape(3, 4, 2)
        g["wgt_spw0"] = np.arange(60).reshape(3, 5, 2, 2)
        selected = SpectralSelection(g, 106134106134, 2222)
        np.testing.assert_equal(selected.dataset("time_avg_array"), [10., 12.])
        np.testing.assert_equal(selected.dataset("bl_vecs"), [[3, 4, 5]])
        np.testing.assert_equal(selected.dataset("data_spw0"), g["data_spw0"][:][[0, 2], :, 1:2])
        assert selected.dataset("wgt_spw0").shape == (2, 5, 2, 1)
        np.testing.assert_equal(selected.attribute("scalar_array"), np.arange(28).reshape(14, 2)[:, 1:2])
        assert selected.record()["row_indices"] == [0, 2]
        for baseline, pol in [(100999100999, 2121), (106134106134, 2424)]:
            with pytest.raises(ValueError, match="absent"):
                SpectralSelection(g, baseline, pol)
        g["bl_array"][1] = 100111
        with pytest.raises(ValueError, match="geometry disagree"):
            SpectralSelection(g, 106134106134)
