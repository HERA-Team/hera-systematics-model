"""Canonical label metadata for a fully averaged single-baseline spectrum."""

import numpy as np


def canonicalize_time_average_labels(spectrum):
    """Collapse repeated, identical labels only after all rows are averaged.

    Some spectral averaging versions retain the incoming interleave label
    axis after reducing physical rows to one. Distinct labels cannot be
    assigned to that row by this operation and are rejected. Numerical
    spectra, uncertainties, coordinates and averaging weights are untouched.
    """
    if spectrum.Nbltpairs != 1 or np.asarray(spectrum.blpair_array).shape != (1,):
        raise ValueError("label normalization requires one averaged physical row")
    expected = (spectrum.Nspws, 1, spectrum.Npols)
    updates, records = {}, {}
    for name in ("label_1_array", "label_2_array"):
        labels = np.asarray(getattr(spectrum, name))
        if (labels.ndim != 3 or labels.shape[0] != expected[0]
                or labels.shape[2] != expected[2] or labels.shape[1] < 1
                or labels.dtype.kind not in "iu" or np.any(labels < 0)
                or np.any(labels >= len(spectrum.labels))):
            raise ValueError("invalid averaged spectral label coordinates")
        collapsed = labels[:, :1, :].copy()
        if not np.array_equal(labels, np.broadcast_to(collapsed, labels.shape)):
            raise ValueError("distinct labels remain on the averaged physical row")
        updates[name] = collapsed
        records[name] = {"input_shape": list(labels.shape), "output_shape": list(collapsed.shape),
                         "input_labels": labels.tolist(), "output_labels": collapsed.tolist(),
                         "changed": labels.shape != collapsed.shape}
    changed = any(item["changed"] for item in records.values())
    for name, labels in updates.items():
        setattr(spectrum, name, labels)
    if changed:
        spectrum.history += "\nRepeated identical interleave labels collapsed to the single averaged physical row."
    return {"schema_version": 1, "passed": True, "changed": changed,
            "physical_rows": 1, "label_arrays": records,
            "numerical_payload_modified": False}
