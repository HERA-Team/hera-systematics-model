"""Synthetic streaming I/O checks with the optional visibility stack."""

import numpy as np
import pytest

pytest.importorskip("pyuvdata")
from pyuvdata import UVData

from hera_systematics_model.cornerturn import cornerturn_baselines
from test_ideal_uvh5 import visibility


def test_cornerturn_preserves_reordered_baselines_and_real_time_gaps(tmp_path):
    times = 2459000. + np.array([0., 10., 40., 50.]) / 86400
    inputs = []
    expected = {}
    for block in range(2):
        data = visibility(times[block * 2:block * 2 + 2], pairs=((0, 2), (0, 1)))
        data.data_array[:] = (data.ant_2_array[:, None, None] + np.arange(4)[None, :, None]) * (1 + 2j)
        data.flag_array[0, 1, 0] = True
        data.nsample_array[0, 1, 0] = 0
        data.data_array[:, 0, 0] = 0.
        data.reorder_blts(order="baseline")
        path = tmp_path / f"chunk-{block}.uvh5"
        data.write_uvh5(path)
        inputs.append(path)
        for row in range(data.Nblts):
            expected[(float(data.time_array[row]), int(data.ant_2_array[row]))] = tuple(
                getattr(data, name)[row].copy() for name in ("data_array", "flag_array", "nsample_array"))
    output = tmp_path / "baselines"
    result = cornerturn_baselines(inputs[::-1], [(0, 2), (0, 1)], output)
    assert result["passed"] and len(result["products"]) == 2
    for product in result["products"]:
        import h5py
        assert product["feed_metadata"]["x_orientation"] == "east"
        with h5py.File(product["output"]["path"]) as handle:
            assert handle["Header/x_orientation"][()] == b"east"
            assert "feed_angle" in handle["Header"]
        data = UVData.from_file(product["output"]["path"])
        np.testing.assert_equal(data.time_array, times)
        assert product["all_rows_written"] and product["numerical_samples_preserved"]
        for row in range(data.Nblts):
            truth = expected[(float(data.time_array[row]), int(data.ant_2_array[row]))]
            for name, value in zip(("data_array", "flag_array", "nsample_array"), truth):
                np.testing.assert_equal(getattr(data, name)[row], value)
    with pytest.raises(FileExistsError):
        cornerturn_baselines(inputs, [(0, 1)], output)


def test_cornerturn_rejects_duplicate_or_absent_baselines(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        cornerturn_baselines([], [(0, 1), (0, 1)], tmp_path / "out")
    data = visibility(2459000. + np.array([0., 10.]) / 86400)
    path = tmp_path / "chunk.uvh5"
    data.write_uvh5(path)
    with pytest.raises(ValueError, match="absent"):
        cornerturn_baselines([path], [(0, 2)], tmp_path / "out")
    duplicate = tmp_path / "duplicate.uvh5"
    duplicate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        cornerturn_baselines([path, duplicate], [(0, 1)], tmp_path / "duplicate-out")


def test_explicit_unprojected_uvw_correction_preserves_numerical_samples(tmp_path):
    import h5py

    data = visibility(2459000. + np.array([0., 10.]) / 86400)
    data.data_array[:] = 3 + 4j
    path = tmp_path / "chunk.uvh5"
    data.write_uvh5(path)
    with h5py.File(path, "r+") as handle:
        handle["Header/uvw_array"][0] += [80., 0., 0.]
    result = cornerturn_baselines([path], [(0, 1)], tmp_path / "corrected", uvw_policy="recalculate_unprojected")
    product = result["products"][0]
    assert product["geometry"]["changed_rows"] == 1
    np.testing.assert_allclose(product["geometry"]["maximum_uvw_change_m"], 80.)
    output = UVData.from_file(product["output"]["path"])
    output.check(strict_uvw_antpos_check=True)
    np.testing.assert_allclose(output.uvw_array, data.uvw_array, atol=1e-12)
    for name in ("data_array", "flag_array", "nsample_array", "time_array", "lst_array"):
        np.testing.assert_equal(getattr(output, name), getattr(data, name))


def test_uvw_correction_refuses_phased_data_and_loaded_visibility_payloads():
    from hera_systematics_model.visibility_geometry import recalculate_unprojected_uvws

    data = visibility(2459000. + np.array([0., 10.]) / 86400)
    with pytest.raises(ValueError, match="metadata-only"):
        recalculate_unprojected_uvws(data)
    metadata = data.copy(metadata_only=True)
    for entry in metadata.phase_center_catalog.values():
        entry["cat_type"] = "sidereal"
    with pytest.raises(ValueError, match="unprojected"):
        recalculate_unprojected_uvws(metadata)


def test_cornerturn_rejects_different_physical_feed_orientations(tmp_path):
    inputs = []
    for block, orientation in enumerate(["east", "north"]):
        data = visibility(2459000. + np.array([block * 20., block * 20. + 10.]) / 86400)
        data.telescope.set_feeds_from_x_orientation(orientation)
        path = tmp_path / f"chunk-{block}.uvh5"
        data.write_uvh5(path)
        inputs.append(path)
    output = tmp_path / "baselines"
    with pytest.raises(ValueError, match="feed orientations differ"):
        cornerturn_baselines(inputs, [(0, 1)], output)
    assert not output.exists()


def test_buffered_cornerturn_matches_single_row_writes(tmp_path):
    import h5py

    times = 2459000. + np.arange(10) * 10 / 86400
    inputs = []
    for block in range(5):
        data = visibility(times[block * 2:block * 2 + 2], pairs=((0, 1), (0, 2)))
        data.data_array[:] = (block + 1) * (1 + 2j)
        data.data_array[:, 0, 0] = 0
        data.flag_array[0, 1, 0] = True
        data.nsample_array[0, 1, 0] = 0
        path = tmp_path / f"chunk-{block}.uvh5"
        data.write_uvh5(path)
        inputs.append(path)
    reference = cornerturn_baselines(inputs, [(0, 1), (0, 2)], tmp_path / "single", write_buffer_rows=1)
    buffered = cornerturn_baselines(inputs[::-1], [(0, 2), (0, 1)], tmp_path / "buffered", write_buffer_rows=4)
    for original, changed in zip(reference["products"], buffered["products"]):
        assert original["write_buffer"]["write_calls"] == 10
        assert changed["write_buffer"] == {"row_limit": 4, "maximum_buffered_rows": 4, "write_calls": 3}
        assert original["valid_cells"] == changed["valid_cells"]
        with h5py.File(original["output"]["path"]) as a, h5py.File(changed["output"]["path"]) as b:
            for name in ["Data/visdata", "Data/flags", "Data/nsamples", "Header/time_array", "Header/lst_array",
                         "Header/uvw_array", "Header/ant_1_array", "Header/ant_2_array", "Header/x_orientation"]:
                np.testing.assert_equal(a[name][()], b[name][()])


def test_baseline_metadata_does_not_retain_full_inventory_arrays(tmp_path, monkeypatch):
    data = visibility(2459000. + np.arange(8) * 10. / 86400,
                      pairs=((0, 1), (0, 2)))
    data.reorder_blts(order="time")
    path = tmp_path / "two-baselines.uvh5"
    data.write_uvh5(path)
    initialize = UVData.initialize_uvh5_file
    checked = []

    def bounded_metadata(self, *args, **kwargs):
        assert self.Nbls == 1
        for name in self:
            parameter = getattr(self, name)
            value = parameter.value
            if isinstance(value, np.ndarray) and "Nblts" in parameter.form:
                owner = value
                while isinstance(owner.base, np.ndarray):
                    owner = owner.base
                assert owner.nbytes <= value.nbytes, parameter.name
        checked.append(self.Nblts)
        return initialize(self, *args, **kwargs)

    monkeypatch.setattr(UVData, "initialize_uvh5_file", bounded_metadata)
    result = cornerturn_baselines([path], [(0, 1), (0, 2)], tmp_path / "out")
    assert result["passed"] and checked == [8, 8]
