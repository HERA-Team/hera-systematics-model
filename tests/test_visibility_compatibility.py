"""Feed metadata conversion must preserve physical orientation and data."""

import h5py
import numpy as np
import pytest

from hera_systematics_model.visibility_compatibility import copy_for_legacy_reader, legacy_orientation


def make_file(path, angles=((np.pi / 2, 0), (np.pi / 2, 0))):
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h["Nants_telescope"] = 2
        h["feed_array"] = np.array([[b"x", b"y"], [b"x", b"y"]])
        h["feed_angle"] = angles
        h["polarization_array"] = [-5, -7, -8, -6]
        h.create_dataset("optional", dtype="f8")
        f["Data/visdata"] = np.array([0, -3, np.nan, 1 + 2j])
        f["Data/flags"] = [False, False, True, False]
        f["Data/nsamples"] = [1., 1., 0., 1.]


def test_lossless_copy_and_exclusive_output(tmp_path):
    source, output = tmp_path / "source.h5", tmp_path / "output.h5"
    make_file(source)
    result = copy_for_legacy_reader(source, output)
    assert result["passed"] and not result["numerical_arrays_changed"]
    assert result["added_datasets"] == ["Header/x_orientation"]
    assert len(result["verified_unchanged_datasets"]) == 8
    with h5py.File(source) as f:
        assert "x_orientation" not in f["Header"]
    with h5py.File(output) as f:
        assert f["Header/x_orientation"][()] == b"east"
    with pytest.raises(FileExistsError):
        copy_for_legacy_reader(source, output)
    next_output = tmp_path / "next.h5"
    assert copy_for_legacy_reader(output, next_output)["added_datasets"] == []


@pytest.mark.parametrize("angles", [((0, 0), (0, 0)), ((np.pi / 2, 0), (0, np.pi / 2)),
                                     ((np.nan, 0), (np.pi / 2, 0)),
                                     ((3 * np.pi / 2, 0), (3 * np.pi / 2, 0))])
def test_ambiguous_feeds_rejected_before_copy(tmp_path, angles):
    source, output = tmp_path / "source.h5", tmp_path / "output.h5"
    make_file(source, angles)
    with pytest.raises(ValueError):
        copy_for_legacy_reader(source, output)
    assert not output.exists()


def test_legacy_conflict_and_missing_metadata(tmp_path):
    source = tmp_path / "source.h5"
    make_file(source)
    with h5py.File(source, "r+") as f:
        h = f["Header"]
        h["x_orientation"] = np.bytes_("north")
        with pytest.raises(ValueError, match="disagree"):
            legacy_orientation(h)
        del h["feed_angle"]
        with pytest.raises(ValueError, match="incomplete"):
            legacy_orientation(h)
        del h["feed_array"]
        assert legacy_orientation(h) == "north"
        del h["x_orientation"]
        with pytest.raises(ValueError, match="unavailable"):
            legacy_orientation(h)


def test_reordered_feeds_keep_north_orientation(tmp_path):
    source = tmp_path / "source.h5"
    make_file(source, ((0, np.pi / 2), (np.pi / 2, 0)))
    with h5py.File(source, "r+") as f:
        f["Header/feed_array"][1] = [b"y", b"x"]
        assert legacy_orientation(f["Header"]) == "north"


def test_orientation_agrees_with_pyuvdata(tmp_path):
    pyuvdata = pytest.importorskip("pyuvdata")
    source = tmp_path / "source.h5"
    make_file(source)
    with h5py.File(source) as f:
        h = f["Header"]
        assert legacy_orientation(h) == pyuvdata.utils.pol.get_x_orientation_from_feeds(
            h["feed_array"][()].astype("U"), h["feed_angle"][()], tols=(0, 1e-6))
