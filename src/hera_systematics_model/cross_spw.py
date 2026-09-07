"""Descriptive cross-spectrum comparisons with explicit physical coordinates."""

import numpy as np

from .baselines import cross_window_similarity, match_groups
from .rebinning import rebinned_mode_similarity
from .reconstruction import physical_mode_energy
from .scoring import CandidateFailure
from .statistics import summary
from .views import analysis_view


def cross_spw_diagnostics(left, right, left_fit, right_fit):
    """Compare saved descriptive models; report unavailable bases explicitly."""
    match_groups(left, right)
    if left.metadata["spw"] == right.metadata["spw"]:
        raise ValueError("two distinct spectral windows are required")
    output, models, descriptions = {}, [], []
    for label, samples, fit in (("left", left, left_fit), ("right", right, right_fit)):
        arrays, shape, identity = analysis_view(samples)
        if (fit.metadata.get("purpose") != "descriptive_fit" or not fit.metadata.get("complete")
                or fit.metadata.get("identity") != identity
                or not np.array_equal(fit.arrays["window_ids"], samples.window_ids)):
            raise ValueError("cross-spectrum samples must match a complete full-plane descriptive fit")
        model = fit.models["descriptive"][0]
        models.append(model)
        description = {"identity": identity, "selected": fit.metadata["selected"],
                       "rank": model.rank, "native_cells": int(np.prod(shape)),
                       "supported_features": int(model.feature_mask.sum())}
        output[label + "_feature_support"] = model.feature_mask.copy()
        for name in ("window_ids", "time_jd", "lst_rad", "delay_s", "kparallel", "kperp", "baseline_length_m"):
            output[label + "_" + name] = getattr(samples, name).copy()
        try:
            energies, energy_metadata = physical_mode_energy(model, fit.arrays["training_scores"],
                arrays[1], arrays[2], arrays[3], np.broadcast_to(samples.kparallel > .3, shape).ravel())
            output.update({label + "_mode_" + key: value for key, value in energies.items()})
            energy_metadata.update(kparallel_threshold=.3, kparallel_units="h Mpc^-1",
                                   energy_units=f"({identity['power_units']})^2")
            energy_metadata["summaries"] = [{key: summary(energies[key][mode]) for key in
                ("full_window_energy", "high_k_window_energy", "high_k_energy_fraction")}
                for mode in range(model.rank)]
            description["physical_mode_energy"] = energy_metadata
        except CandidateFailure as error:
            description["physical_mode_energy"] = {"available": False, "reason": str(error)}
        descriptions.append(description)
    metadata = {"purpose": "cross_spw_diagnostics", "complete": True,
        "left": descriptions[0], "right": descriptions[1],
        "spw_0_6_comparison": {left.metadata["spw"], right.metadata["spw"]} == {0, 6},
        "uses_validation_for_selection": False,
        "limitations": ["descriptive fits of one correlated LST arc", "mode signs are arbitrary",
            "different spectral windows probe different cosmological coordinates",
            "component similarity does not identify a physical cause",
            "conditional nonlinear mode contrasts are not a universal physical-power basis"]}
    if any(model.rank == 0 or model.metadata["method"] == "kernel" for model in models):
        metadata["mode_similarity"] = {"available": False,
            "reason": "a selected model has rank zero or a nonlinear kernel decoder without linear components"}
        return output, metadata
    components = [model.components[:model.rank] for model in models]
    for label, values in zip(("left", "right"), components):
        output[label + "_components"] = values.copy()
    parameters = (left, right, *components, *(model.feature_mask for model in models))
    comparison = {"available": True, "space": "saved residual representation coordinates",
        "same_representation": models[0].metadata["representation"] == models[1].metadata["representation"],
        "physical_power_basis": False}
    if not comparison["same_representation"]:
        comparison["representation_limitation"] = "different transforms; similarity is not equality of physical power modes"
    try:
        exact = cross_window_similarity(*parameters)
        output.update({"exact_" + key: value for key, value in exact.items() if isinstance(value, np.ndarray)})
        comparison["exact"] = {"available": exact["matched_cells"] > 0,
            **{key: value for key, value in exact.items() if not isinstance(value, np.ndarray)}}
        if not exact["matched_cells"]:
            comparison["exact"]["reason"] = "no shared trained support on matching native coordinates"
    except ValueError as error:
        if str(error) != "no matched physical group/delay coordinates":
            raise
        comparison["exact"] = {"available": False, "reason": str(error)}
    try:
        rebinned, report = rebinned_mode_similarity(*parameters)
        output.update({"rebinned_" + key: value for key, value in rebinned.items()})
        comparison["rebinned"] = {"available": report["matched_cells"] > 0, **report}
        if not report["matched_cells"]:
            comparison["rebinned"]["reason"] = "no fully supported common bins"
    except ValueError as error:
        if str(error) != "no complete common delay bin":
            raise
        comparison["rebinned"] = {"available": False, "reason": str(error)}
    metadata["mode_similarity"] = comparison
    return output, metadata
