import h5py
import numpy as np
import pytest

from hera_systematics_model.visibility_inventory import inventory_visibilities, visibility_header


def create(path, time=1., order=(0, 1, 2, 3)):
    with h5py.File(path, "w") as f:
        h, d = f.create_group("Header"), f.create_group("Data")
        for key, value in {"ant_1_array": [0, 0, 0, 0], "ant_2_array": [1, 2, 1, 2],
                           "time_array": [time, time, time + .01, time + .01],
                           "lst_array": [6.28, 6.28, .001, .001], "integration_time": [8., 8., 8., 8.]}.items():
            h[key] = np.asarray(value)[list(order)]
        h["freq_array"] = [100., 101.]
        h["polarization_array"] = [-5, -6]
        h["antenna_numbers"] = [0, 1, 2]
        h["antenna_positions"] = np.eye(3)
        h["vis_units"], h["history"] = "Jy", "source"
        d["visdata"] = np.zeros((4, 2, 2), complex)
        d["flags"] = np.ones((4, 2, 2), bool)
        d["nsamples"] = np.zeros((4, 2, 2))


def test_physical_inventory_preserves_wrap_and_does_not_infer_validity(tmp_path):
    a, b = tmp_path / "later.uvh5", tmp_path / "earlier.uvh5"
    create(a, 2., (3, 1, 2, 0))
    create(b, 1.)
    result = inventory_visibilities([a, b])
    assert result["metadata_only"]
    assert len(result["baseline_inventories"]) == 1
    assert [entry["path"] for entry in result["files"]] == [str(b), str(a)]
    assert result["files"][1]["lsts_rad"] == [6.28, .001]
    assert result["files"][0]["cross_baselines"] == 2
    with pytest.raises(ValueError, match="unique"):
        inventory_visibilities([a, a])


def test_inventory_rejects_duplicate_rows_and_inconsistent_lsts(tmp_path):
    a, b = tmp_path / "duplicate.uvh5", tmp_path / "lst.uvh5"
    create(a, order=(0, 0, 2, 3))
    create(b)
    with h5py.File(b, "r+") as f:
        f["Header/lst_array"][1] = 1.
    with pytest.raises(ValueError, match="duplicate"):
        visibility_header(a)
    with pytest.raises(ValueError, match="LST"):
        visibility_header(b)
