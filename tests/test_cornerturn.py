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
