"""Versioned numerical artifacts with checked metadata and no object arrays."""

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
KINDS = {"paired-samples", "fitted-model", "evaluation", "spectrum-records", "diagnostics", "window-memberships",
         "spectral-merge"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    """Reject nonfinite JSON numbers; unavailable values need explicit reasons."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_artifact(path, kind, arrays, metadata):
    """Create an immutable NPZ and a hash-linked JSON sidecar.

    A partial write has no valid sidecar and cannot pass read verification.
    Existing products are never replaced, including incomplete products.
    """
    path = Path(path)
    if path.suffix != ".npz" or kind not in KINDS:
        raise ValueError("invalid artifact path or kind")
    if not arrays or not isinstance(metadata, dict):
        raise ValueError("arrays and metadata are required")
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if any(not isinstance(key, str) or not key for key in arrays):
        raise ValueError("array names must be nonempty strings")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are not supported")
    canonical_json(metadata)
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    description = {
        "schema_version": SCHEMA_VERSION, "kind": kind, "metadata": metadata,
        "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str}
                   for key, value in arrays.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    description["sha256"] = sha256_file(path)
    with sidecar.open("x") as stream:
        stream.write(canonical_json(description) + "\n")
    return description


def read_artifact(path, expected_kind):
    """Verify version, file hash, array structure and kind before returning data."""
    path = Path(path)
    with path.with_suffix(".json").open() as stream:
        description = json.load(stream)
    canonical_json(description)
    if description.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema")
    if expected_kind not in KINDS or description.get("kind") != expected_kind:
        raise ValueError("artifact kind mismatch")
    if description.get("sha256") != sha256_file(path):
        raise ValueError("artifact hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    actual = {key: {"shape": list(value.shape), "dtype": value.dtype.str}
              for key, value in arrays.items()}
    if actual != description.get("arrays"):
        raise ValueError("artifact array schema mismatch")
    return arrays, description["metadata"]
