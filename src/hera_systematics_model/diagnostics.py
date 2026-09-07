"""Measured cylindrical maps and fit/evaluation diagnostics on explicit support."""

import warnings

import numpy as np

from .baselines import baseline_inventory, mode_localization
from .scoring import CandidateFailure, score_predictions, training_mean
from .statistics import score_diagnostics, summary, surrogate_gaussianity
from .views import analysis_view, geometry_masks, sample_view
from .configuration import AnalysisConfig


def regional_losses(prediction, truth, pn, target, eligible, masks):
    """Keep every region's coverage denominators alongside its predictive loss."""
    report = {}
    for name, mask in masks.items():
        selected = target & mask.ravel()
        denominator = eligible & mask.ravel()
        if selected.any():
            score = score_predictions(prediction, truth, pn, selected)
            measured = summary(score.per_window)
        else:
            measured = {"n": 0, "mean": None, "reason": "no scored target cells in region"}
        report[name] = {"window_loss": measured, "scored_cells": int(selected.sum()),
                       "eligible_cells": int(denominator.sum()), "geometric_features": int(mask.sum())}
    return report


def residual_diagnostics(samples, descriptive, evaluation, n_surrogates=1000, seed=0):
    """Combine measured maps and descriptive diagnostics without fitting new modes."""
    samples.validate()
    config = AnalysisConfig(**descriptive.metadata.get("configuration", {}))
    compared = AnalysisConfig(**evaluation.metadata.get("configuration", {})).as_dict()
    expected = config.as_dict()
    compared.pop("guard")
    expected.pop("guard")
    if compared != expected:
        raise ValueError("diagnostic fit and evaluation configurations disagree")
    samples = sample_view(samples, config.group, config.delay)
    arrays, shape, identity = analysis_view(samples)
    for artifact in (descriptive, evaluation):
        if artifact.metadata.get("identity") != identity:
            raise ValueError("diagnostic input physical feature identities disagree")
        if not np.array_equal(artifact.arrays["window_ids"], samples.window_ids):
            raise ValueError("diagnostic input physical time identities disagree")
    if descriptive.metadata.get("purpose") != "descriptive_fit" or not descriptive.metadata.get("complete"):
        raise ValueError("verified descriptive fit required for fitted-mode diagnostics")
    power, ideal, pn, valid = arrays
    output = {"window_ids": samples.window_ids.copy(), "time_jd": samples.time_jd.copy(),
              "lst_rad": samples.lst_rad.copy(), "lst_unwrapped_hours": np.unwrap(samples.lst_rad) * 12 / np.pi,
              "delay_s": samples.delay_s.copy(), "kparallel": samples.kparallel.copy(),
              "kperp": samples.kperp.copy(), "baseline_length_m": samples.baseline_length_m.copy(),
              "validity": samples.valid.copy(), "valid_fraction": samples.valid.mean(axis=0),
              "contributor_counts": samples.contributor_counts, "noise": samples.pn.copy(),
              "mean_corrupted": training_mean(power, valid).reshape(shape),
              "mean_ideal": training_mean(ideal, valid).reshape(shape),
              "mean_residual": training_mean(power - ideal, valid).reshape(shape),
              "scores": descriptive.arrays["training_scores"].copy(),
              "prediction": evaluation.arrays["prediction"].copy(),
              "window_loss": evaluation.arrays["window_loss"].copy(),
              "projection_window_loss": evaluation.arrays.get("projection_window_loss", np.full(len(power), np.nan)).copy(),
              "zero_baseline_loss": evaluation.arrays["zero_baseline_loss"].copy(),
              "mean_baseline_loss": evaluation.arrays["mean_baseline_loss"].copy()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output["median_noise"] = np.nanmedian(np.where(samples.valid, samples.pn, np.nan), axis=0)
    model = descriptive.models["descriptive"][0]
    from .reconstruction import physical_mode_energy

    try:
        energy_arrays, energy_metadata = physical_mode_energy(model, output["scores"], ideal, pn, valid,
            np.broadcast_to(samples.kparallel > .3, shape).ravel())
        output.update({"mode_" + key: value for key, value in energy_arrays.items()})
        energy_metadata.update(available=model.rank > 0, kparallel_threshold=.3,
                               kparallel_units="h Mpc^-1", energy_units=f"({identity['power_units']})^2")
        energy_metadata["mode_summaries"] = [{name: summary(energy_arrays[name][mode]) for name in
            ("full_window_energy", "high_k_window_energy", "high_k_energy_fraction")} for mode in range(model.rank)]
    except CandidateFailure as error:
        energy_metadata = {"available": False, "reason": str(error)}
    modes = {"available": False, "reason": "rank-zero model has no components"}
    if model.rank and model.metadata["method"] != "kernel":
        output["components"] = model.components[:model.rank].copy()
        output["singular_values"] = model.singular_values.copy()
        output["explained_variance_ratio"] = model.explained_variance_ratio.copy()
        output["cumulative_variance_ratio"] = np.cumsum(model.explained_variance_ratio)
        modes = {"available": True, "space": model.metadata["representation"],
                 "localization": mode_localization(samples, output["components"], model.metadata["representation"]),
                 "variance_available": bool(len(model.explained_variance_ratio)),
                 "variance_unavailable_reason": None if len(model.explained_variance_ratio) else "observed-entry factorization has no ordinary PCA EVR"}
    elif model.rank:
        output["kernel_eigenvalues"] = model.eigenvalues.copy()
        modes = {"available": False, "reason": "kernel model has a nonlinear decoder and no conventional linear power basis"}
    score_report, score_arrays = score_diagnostics(output["scores"], samples.window_ids,
        output["lst_unwrapped_hours"], valid.mean(axis=1))
    output.update(score_arrays)
    for mode, record in enumerate(score_report):
        record["surrogate"] = surrogate_gaussianity(output["scores"][:, mode], samples.window_ids, n_surrogates, seed + mode)
    masks = geometry_masks(samples, samples.baseline_length_m / 299792458.)
    output.update({f"region_{name}": mask for name, mask in masks.items()})
    output.update({name: evaluation.arrays[name].copy() for name in ("target", "eligible", "modeled", "mean_only", "zero_only", "unavailable", "excluded")})
    regions = regional_losses(output["prediction"], power - ideal, pn, output["target"], output["eligible"], masks)
    for name in (key for key in evaluation.arrays if key.startswith("inner_losses_")):
        output[name] = evaluation.arrays[name].copy()
    metadata = {"identity": identity, "configuration": config.as_dict(), "purpose": "residual_diagnostics", "selected": descriptive.metadata["selected"],
        "evaluation_complete": evaluation.metadata["complete"], "candidates": evaluation.metadata["candidates"],
        "modes": modes, "physical_mode_energy": energy_metadata, "score_diagnostics": score_report,
        "baseline_inventory": baseline_inventory(samples), "regions": regions, "folds": evaluation.metadata["folds"],
        "noise_assumption": samples.metadata["noise_model"], "horizon_definition": "baseline group length divided by speed of light",
        "horizon_buffer_ns": 500., "scores_source": "full-data descriptive fit",
        "loss_source": "outer physical-time and withheld-feature predictions",
        "projection_available": "projection_window_loss" in evaluation.arrays,
        "projection_unavailable_reason": None if "projection_window_loss" in evaluation.arrays else "not recorded in this evaluation artifact",
        "limitations": ["correlated windows from one LST arc", "no signal-preservation injections",
                        "no independent-realization generalization", "no calibrated nuisance priors or inference integration"]}
    return output, metadata
