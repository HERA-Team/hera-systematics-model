"""Physical metadata inventories without loading visibility data arrays."""

import hashlib
from pathlib import Path

import numpy as np

from .configuration import digest_json


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def visibility_header(path):
    """Read physical identities and array layouts; data validity is not inferred."""
    import h5py

    path = Path(path).resolve()
    before = path.stat()
    with h5py.File(path, "r") as handle:
        header, data = handle["Header"], handle["Data"]
        arrays = {key: header[key][()] for key in ("ant_1_array", "ant_2_array", "time_array",
                  "lst_array", "freq_array", "polarization_array", "integration_time",
                  "antenna_numbers", "antenna_positions")}
        a, b, times, lsts = (arrays[key] for key in ("ant_1_array", "ant_2_array", "time_array", "lst_array"))
        if (times.ndim != 1 or not len(times) or any(value.shape != times.shape for value in (a, b, lsts))
                or not np.isfinite(times).all() or not np.isfinite(lsts).all()):
            raise ValueError("invalid physical visibility row coordinates")
        rows = np.rec.fromarrays([times, a, b], names="time,ant1,ant2")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("duplicate physical visibility row")
        unique_times, first = np.unique(times, return_index=True)
        for time, index in zip(unique_times, first):
            if not np.all(lsts[times == time] == lsts[index]):
                raise ValueError("inconsistent LST at a physical time")
        integration = arrays["integration_time"]
        if integration.shape != times.shape or not np.isfinite(integration).all() or np.any(integration <= 0):
            raise ValueError("invalid native integration durations")
        frequencies, polarizations = arrays["freq_array"].ravel(), arrays["polarization_array"]
        if (not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0)
                or len(np.unique(polarizations)) != len(polarizations)):
            raise ValueError("invalid spectral visibility coordinates")
        shape = (len(times), len(frequencies), len(polarizations))
        structures = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                      for key in ("visdata", "flags", "nsamples")}
        if any(tuple(value["shape"]) != shape for value in structures.values()):
            raise ValueError("visibility payload layout disagrees with coordinates")
        units = header["vis_units"][()]
        units = units.decode() if isinstance(units, bytes) else str(units)
        history = header["history"][()]
        history = history if isinstance(history, bytes) else str(history).encode()
        pairs = np.unique(np.column_stack([a, b]), axis=0)
        result = {"path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                  "times_jd": unique_times.tolist(), "lsts_rad": lsts[first].tolist(),
                  "baseline_pairs": pairs.tolist(), "cross_baselines": int((pairs[:, 0] != pairs[:, 1]).sum()),
                  "auto_baselines": int((pairs[:, 0] == pairs[:, 1]).sum()), "vis_units": units,
                  "integration_seconds": [float(integration.min()), float(integration.max())],
                  "coordinates": {key: array_digest(value) for key, value in arrays.items()},
                  "history_sha256": hashlib.sha256(history).hexdigest(), "arrays": structures}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("visibility file changed during metadata inventory")
    return result


def inventory_visibilities(paths):
    """Deduplicate baseline sets and order files by physical time then path."""
    paths = list(map(Path, paths))
    resolved = [str(path.resolve()) for path in paths]
    if not paths or len(resolved) != len(set(resolved)):
        raise ValueError("unique nonempty visibility paths required")
    entries, baselines = [], {}
    for path in paths:
        entry = visibility_header(path)
        pairs = entry.pop("baseline_pairs")
        identity = digest_json(pairs)
        baselines[identity] = pairs
        entry["baseline_inventory"] = identity
        entries.append(entry)
    entries.sort(key=lambda entry: (entry["times_jd"][0], entry["path"]))
    return {"schema_version": 1, "metadata_only": True, "files": entries,
            "baseline_inventories": baselines, "logical_bytes": sum(entry["bytes"] for entry in entries)}
