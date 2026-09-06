import json

import h5py
import numpy as np
import pytest

from hera_systematics_model.configuration import file_identity
from hera_systematics_model.ideal_verification import verify_ideal_chunk
from test_visibility_inventory import create


def test_verified_ideal_retains_zero_and_rejects_count_or_metadata_changes(tmp_path):
    reference, ideal, source = [tmp_path / name for name in ("reference.uvh5", "ideal.uvh5", "source.txt")]
    create(reference)
    create(ideal)
    source.write_text("source")
    with h5py.File(ideal, "r+") as f:
        f["Data/flags"][:] = False
        f["Data/nsamples"][:] = 1.
    sidecar = {"reference": file_identity(reference), "output": file_identity(ideal),
               "sources": [file_identity(source)], "supported_cells": 16, "total_cells": 16}

    def record():
        sidecar["output"] = file_identity(ideal)
        ideal.with_suffix(".json").write_text(json.dumps(sidecar))

    record()
    result = verify_ideal_chunk(reference, ideal)
    assert result["totals"]["valid_zero_cells"] == 16
    assert result["totals"]["newly_unflagged_cells"] == 16
    with h5py.File(ideal, "r+") as f:
        f["Data/nsamples"][0, 0, 0] = 2.
    record()
    with pytest.raises(ValueError, match="unit-count"):
        verify_ideal_chunk(reference, ideal)
    with h5py.File(ideal, "r+") as f:
        f["Data/nsamples"][0, 0, 0] = 1.
        f["Header/integration_time"][0] = 16.
    record()
    with pytest.raises(ValueError, match="physical metadata"):
        verify_ideal_chunk(reference, ideal)
