"""Lossless UVH5 feed metadata compatibility for older spectral readers."""

from pathlib import Path
import shutil

import numpy as np

from .configuration import file_identity
from .production import write_json_exclusive


def legacy_orientation(header):
    """Require one cardinal linear-feed orientation shared by every antenna."""
    modern = [name in header for name in ("feed_array", "feed_angle")]
    if any(modern) and not all(modern):
        raise ValueError("incomplete feed orientation metadata")
    orientation = None
    if all(modern):
        feeds = np.asarray(header["feed_array"][()]).astype("U")
        angles = np.asarray(header["feed_angle"][()])
        nants = int(header["Nants_telescope"][()])
        if feeds.shape != (nants, 2) or angles.shape != feeds.shape or not np.isfinite(angles).all():
            raise ValueError("invalid per-antenna linear feed metadata")
        if not np.all(np.sort(feeds, axis=1) == ["x", "y"]):
            raise ValueError("legacy orientation requires x and y feeds")
        for candidate, expected in (("east", {"x": np.pi / 2, "y": 0}),
                                    ("north", {"x": 0, "y": np.pi / 2})):
            target = np.where(feeds == "x", expected["x"], expected["y"])
            if np.allclose(angles, target, rtol=0, atol=1e-6):
                orientation = candidate
        if orientation is None:
            raise ValueError("feed angles have no common cardinal orientation")
    if "x_orientation" in header:
        value = header["x_orientation"][()]
        value = value.decode() if isinstance(value, bytes) else str(value)
        if value not in ("east", "north") or (orientation is not None and value != orientation):
            raise ValueError("legacy and modern feed orientations disagree")
        orientation = value
    if orientation is None:
        raise ValueError("feed orientation is unavailable")
    return orientation


def copy_for_legacy_reader(source, output):
    """Add a derived legacy field to an exclusive copy; verify every old dataset."""
    import h5py

    source, output = Path(source), Path(output)
    sidecar = output.with_suffix(".compatibility.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    before = file_identity(source)
    with h5py.File(source, "r") as handle:
        orientation = legacy_orientation(handle["Header"])
        already_present = "x_orientation" in handle["Header"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, output.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    with h5py.File(output, "r+") as handle:
        if not already_present:
            handle["Header"].create_dataset("x_orientation", data=np.bytes_(orientation))
    checked = []
    with h5py.File(source, "r") as original, h5py.File(output, "r") as copied:
        def compare(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            actual = copied[name]
            if actual.shape != obj.shape or actual.dtype != obj.dtype:
                raise ValueError("compatibility copy changed dataset structure")
            if obj.shape is not None:
                chunks = range(0, obj.shape[0], 16) if obj.shape else [None]
                for start in chunks:
                    key = () if start is None else slice(start, start + 16)
                    a, b = obj[key], actual[key]
                    same = np.array_equal(a, b, equal_nan=True) if obj.dtype.kind in "fc" else np.array_equal(a, b)
                    if not same:
                        raise ValueError("compatibility copy changed dataset values")
            checked.append(name)
        original.visititems(compare)
        if legacy_orientation(copied["Header"]) != orientation:
            raise ValueError("compatibility output orientation differs")
    if file_identity(source) != before:
        raise ValueError("compatibility source changed during copying")
    result = {"schema_version": 1, "passed": True, "source": before, "output": file_identity(output),
              "x_orientation": orientation, "added_datasets": [] if already_present else ["Header/x_orientation"],
              "verified_unchanged_datasets": checked, "numerical_arrays_changed": False}
    write_json_exclusive(sidecar, result)
    return result
