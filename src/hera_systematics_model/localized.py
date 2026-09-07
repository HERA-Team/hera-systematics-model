"""Deterministic physical slice inventories from accepted full-plane fits."""

from pathlib import Path

import numpy as np

from .configuration import AnalysisConfig, digest_json, file_identity
from .evaluation import Evaluation
from .samples import PairedSamples
from .views import analysis_view


def localized_inventory(samples_path, fit_path):
    """Bind both slice directions and missing-data methods to a selected input.

    The representation family comes from the full-plane descriptive selection.
    Each localized run selects its own rank and training-derived transform
    parameters. These are conditional diagnostics of that representation.
    """
    samples = PairedSamples.load(samples_path)
    fit = Evaluation.load(fit_path)
    config = AnalysisConfig(**fit.metadata["configuration"])
    _, _, identity = analysis_view(samples)
    inputs = [file_identity(path) for name in (samples_path, fit_path)
              for path in (Path(name), Path(name).with_suffix(".json"))]
    if (fit.metadata.get("purpose") != "descriptive_fit" or not fit.metadata.get("complete")
            or config.group is not None or config.delay is not None
            or config.group_exclusion is not None or config.region != "full" or config.guard != 12
            or fit.metadata.get("identity") != identity
            or fit.metadata.get("input") != inputs[0]
            or fit.metadata.get("input_metadata") != inputs[1]
            or not np.array_equal(fit.arrays["window_ids"], samples.window_ids)):
        raise ValueError("localized inventory requires a matched complete full-plane guard-12 fit")
    representation = fit.metadata["selected"]["representation"]
    tasks = []
    for axis, count in (("group", len(samples.group_ids)), ("delay", len(samples.delay_s))):
        for index in range(count):
            _, shape, coordinates = analysis_view(samples, **{axis: index})
            for method in ("complete", "masked"):
                local = AnalysisConfig(representations=[representation], methods=[method],
                                       include_kernel=False, **{axis: index})
                tasks.append({"name": f"spw-{identity['spw']:02d}-{axis}-{index:04d}-{method}",
                    "axis": axis, "index": index, "method": method,
                    "identity": coordinates, "feature_shape": list(shape),
                    "configuration": local.as_dict(), "configuration_digest": digest_json(local.as_dict()),
                    "feature_partition_axis": "delay" if axis == "group" else "group",
                    "feature_guard_bins": 2 if axis == "group" else 1})
    return {"schema_version": 1, "inputs": inputs, "spw": identity["spw"],
        "selected_representation": representation, "selection_scope": "full-plane descriptive fit",
        "localized_rank_selection": "nested physical-time validation",
        "outer_blocks": 4, "inner_blocks": 3, "feature_blocks": 5,
        "physical_window_ids": samples.window_ids.tolist(), "tasks": tasks}
