"""Independent product checks for ideal visibility flags and metadata."""

import json
from pathlib import Path

import numpy as np

from .configuration import file_identity
from .visibility_inventory import visibility_header


def verify_ideal_chunk(reference_file, product_file, require_supported=True):
    """Check every cell and source identity against the construction sidecar."""
    import h5py

    product_file = Path(product_file)
    provenance = json.loads(product_file.with_suffix(".json").read_text())
    if provenance["reference"] != file_identity(reference_file) or provenance["output"] != file_identity(product_file):
        raise ValueError("ideal product or reference identity mismatch")
    for item in provenance["sources"]:
        if item != file_identity(item["path"]):
            raise ValueError("ideal source identity mismatch")
    reference, product = visibility_header(reference_file), visibility_header(product_file)
    if (reference["coordinates"] != product["coordinates"] or reference["vis_units"] != product["vis_units"]
            or reference["baseline_pairs"] != product["baseline_pairs"]):
        raise ValueError("ideal physical metadata differ from reference")
    totals = {"valid_cells": 0, "invalid_cells": 0, "newly_unflagged_cells": 0,
              "valid_zero_cells": 0, "nonzero_invalid_cells": 0}
    baselines = {f"{a}_{b}": {"valid_cells": 0, "invalid_cells": 0} for a, b in product["baseline_pairs"]}
    with h5py.File(reference_file) as ref, h5py.File(product_file) as out:
        for key in ("uvw_array", "phase_center_id_array"):
            if key in ref["Header"]:
                if key not in out["Header"] or not np.array_equal(ref["Header"][key][()], out["Header"][key][()]):
                    raise ValueError("ideal phase coordinates differ from reference")
        a, b = out["Header/ant_1_array"][:], out["Header/ant_2_array"][:]
        for row, (ant1, ant2) in enumerate(zip(a, b)):
            values = out["Data/visdata"][row]
            flags = out["Data/flags"][row]
            counts = out["Data/nsamples"][row]
            if flags.dtype.kind != "b" or not np.isfinite(values).all() or not np.isfinite(counts).all():
                raise ValueError("invalid ideal payload arrays")
            if not np.array_equal(counts, (~flags).astype(counts.dtype)) or np.any(values[flags] != 0):
                raise ValueError("ideal unit-count or invalid-cell policy violated")
            valid, invalid = int((~flags).sum()), int(flags.sum())
            totals["valid_cells"] += valid
            totals["invalid_cells"] += invalid
            totals["newly_unflagged_cells"] += int((ref["Data/flags"][row] & ~flags).sum())
            totals["valid_zero_cells"] += int(((values == 0) & ~flags).sum())
            baselines[f"{ant1}_{ant2}"]["valid_cells"] += valid
            baselines[f"{ant1}_{ant2}"]["invalid_cells"] += invalid
    if (totals["valid_cells"] != provenance["supported_cells"]
            or totals["valid_cells"] + totals["invalid_cells"] != provenance["total_cells"]):
        raise ValueError("ideal support count differs from construction record")
    if require_supported and totals["valid_cells"] == 0:
        raise ValueError("ideal chunk contains no supported samples")
    return {"passed": True, "reference": provenance["reference"], "product": provenance["output"],
            "totals": totals, "baseline_cells": baselines, "times_jd": product["times_jd"],
            "lsts_rad": product["lsts_rad"], "vis_units": product["vis_units"],
            "support_unavailable_reason": None if totals["valid_cells"] else "no finite unflagged source support",
            "integration_seconds": product["integration_seconds"]}
