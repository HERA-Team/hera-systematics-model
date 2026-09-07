"""Synthetic I/O checks, invoked separately when pyuvdata is installed."""

import numpy as np
import pytest

pyuvdata = pytest.importorskip("pyuvdata")
from astropy.coordinates import EarthLocation
from astropy.utils import iers

from hera_systematics_model.ideal_io import construct_chunk
from hera_systematics_model.ideal_verification import verify_ideal_chunk


def visibility(times, pairs=((0, 1),)):
    iers.conf.auto_download = False
    telescope = pyuvdata.Telescope.new("synthetic", EarthLocation.from_geodetic(21., -30., 1000),
        antenna_positions={0: np.array([0., 0., 0.]), 1: np.array([10., 0., 0.]), 2: np.array([0., 10., 0.])},
        instrument="synthetic", x_orientation="east", feeds=["x", "y"], update_from_known=False, mount_type="fixed")
    return pyuvdata.UVData.new(freq_array=np.array([100e6, 101e6, 102e6, 103e6]),
        polarization_array=[-5, -6], times=np.asarray(times), telescope=telescope, antpairs=list(pairs),
        do_blt_outer=True, integration_time=10., channel_width=1e6, empty=True,
        vis_units="Jy", update_telescope_from_known=False)


def test_model_zero_counts_do_not_erase_finite_source_support(tmp_path):
    times = 2459000. + np.arange(4) * 10 / 86400
    sources = []
    for part in range(2):
        source = visibility(times[part * 2:part * 2 + 2])
        source.data_array[:] = (1 + source.lst_array[:, None, None]) * (1 + 2j)
        source.data_array[:, 0, 0] = 0.
        source.nsample_array[:] = 0.
        if part == 0:
            source.flag_array[0, 1, 0] = True
        path = tmp_path / f"source-{part}.uvh5"
        source.write_uvh5(path)
        sources.append(str(path))
    reference = visibility(2459000. + np.array([15., 25.]) / 86400)
    reference.flag_array[:] = True
    reference.nsample_array[:] = 7.
    path, output = tmp_path / "reference.uvh5", tmp_path / "ideal.uvh5"
    reference.write_uvh5(path)
    mapping = {"0_1": {"reference_pair": [0, 1], "source_pair": [0, 1], "stored_pair": [0, 1],
                       "conjugate": False, "exclusion": None}}
    result = construct_chunk(path, sources, mapping, output)
    verified = verify_ideal_chunk(path, output)
    assert result["source_counts_used"] is False
    assert verified["totals"]["valid_cells"] == 14
    assert verified["totals"]["valid_zero_cells"] == 2
    assert verified["totals"]["invalid_cells"] == 2
    actual = pyuvdata.UVData.from_file(output)
    np.testing.assert_allclose(actual.data_array[:, 2, 0], (1 + reference.lst_array) * (1 + 2j), rtol=1e-12)
    assert actual.flag_array[:, 1, 0].all()


def test_selected_ideal_baseline_is_bound_to_full_reference_identity(tmp_path):
    import json

    times = 2459000. + np.arange(4) * 10 / 86400
    source = visibility(times, pairs=((0, 1), (0, 2)))
    source.data_array[:] = source.ant_2_array[:, None, None] * (1 + 2j)
    source_path = tmp_path / "source.uvh5"
    source.write_uvh5(source_path)
    reference = visibility(2459000. + np.array([15., 25.]) / 86400, pairs=((0, 2), (0, 1)))
    reference.flag_array[:] = True
    reference_path, output = tmp_path / "reference.uvh5", tmp_path / "selected.uvh5"
    reference.write_uvh5(reference_path)
    mapping = {"0_2": {"reference_pair": [0, 2], "source_pair": [0, 2], "stored_pair": [0, 2],
                       "conjugate": False, "exclusion": None}}
    result = construct_chunk(reference_path, [source_path], mapping, output, baselines=[(0, 2)])
    assert result["reference_baseline_selection"] == [[0, 2]]
    verified = verify_ideal_chunk(reference_path, output)
    assert set(verified["baseline_cells"]) == {"0_2"}
    assert verified["totals"]["newly_unflagged_cells"] == 16
    actual = pyuvdata.UVData.from_file(output)
    np.testing.assert_allclose(actual.data_array, 2 + 4j)
    sidecar = json.loads(output.with_suffix(".json").read_text())
    sidecar["reference_baseline_selection"] = [[0, 1]]
    output.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="physical metadata"):
        verify_ideal_chunk(reference_path, output)
