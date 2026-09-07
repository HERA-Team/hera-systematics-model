"""Reconstruct physical residuals from saved scores without target power inputs."""

import numpy as np

from .residuals import INVERSES
from .scoring import CandidateFailure


def decode_scores(model, scores, ideal, pn):
    """Apply the saved representation inverse and linear-mean fallback.

    Only ideal power and the recorded corrupted noise scale accompany scores.
    No corrupted target value is needed to reconstruct a prediction.
    """
    scores, ideal, pn = (np.asarray(value, dtype=float) for value in (scores, ideal, pn))
    if (ideal.ndim != 2 or pn.shape != ideal.shape or scores.shape != (len(ideal), model.rank)
            or ideal.shape[1] != len(model.linear_mean) or not np.isfinite(scores).all()):
        raise ValueError("score reconstruction dimensions or coefficients are invalid")
    if not model.metadata.get("converged", False):
        raise CandidateFailure("cannot reconstruct a nonconverged model")
    method = model.metadata["method"]
    if method == "zero":
        return np.zeros_like(ideal)
    prediction = np.broadcast_to(model.linear_mean, ideal.shape).copy()
    if model.rank == 0:
        return prediction
    mask = model.feature_mask
    if method == "kernel":
        from .kernel import rbf

        decoded = rbf(scores, model.training_scores, model.metadata["decoder_gamma"]) @ model.dual
        decoded = decoded * model.metadata["target_scale"] + model.mean[mask]
    elif method in ("complete", "masked"):
        decoded = model.mean[mask] + scores @ model.components[:model.rank, mask]
    else:
        raise ValueError("unsupported score decoder")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prediction[:, mask] = INVERSES[model.metadata["representation"]](
            decoded, ideal[:, mask], pn[:, mask], **model.metadata["params"]) - ideal[:, mask]
    return prediction


def individual_mode_contrasts(model, scores, ideal, pn):
    """Measure one-score departures from the zero-score reconstruction.

    For nonlinear decoders these are conditional contrasts, not an additive
    decomposition or a universal basis in physical power units.
    """
    scores = np.asarray(scores, dtype=float)
    reference = decode_scores(model, np.zeros_like(scores), ideal, pn)
    for mode in range(model.rank):
        selected = np.zeros_like(scores)
        selected[:, mode] = scores[:, mode]
        yield decode_scores(model, selected, ideal, pn) - reference


def physical_mode_energy(model, scores, ideal, pn, valid, high_k_mask):
    """Report squared residual contrasts with explicit common-cell denominators."""
    valid, high = np.asarray(valid), np.asarray(high_k_mask)
    if (valid.shape != np.shape(ideal) or valid.dtype.kind != "b" or high.dtype.kind != "b"
            or high.shape != (valid.shape[1],)):
        raise ValueError("physical mode diagnostic support disagrees")
    target = valid & model.feature_mask
    selected = target & high
    arrays = {"mean_squared_contrast": np.full((model.rank, valid.shape[1]), np.nan),
              "full_window_energy": np.full((model.rank, len(valid)), np.nan),
              "high_k_window_energy": np.full((model.rank, len(valid)), np.nan),
              "high_k_energy_fraction": np.full((model.rank, len(valid)), np.nan),
              "feature_counts": target.sum(axis=0),
              "full_counts": target.sum(axis=1), "high_k_counts": selected.sum(axis=1)}
    for mode, contrast in enumerate(individual_mode_contrasts(model, scores, ideal, pn)):
        if not np.isfinite(contrast[target]).all():
            raise CandidateFailure("nonfinite physical mode contrast on required support")
        squared = np.where(target, contrast ** 2, 0.)
        if not np.isfinite(squared).all():
            raise CandidateFailure("physical mode energy overflow")
        count = target.sum(axis=0)
        arrays["mean_squared_contrast"][mode] = np.divide(squared.sum(axis=0), count,
            out=np.full(len(count), np.nan), where=count > 0)
        for region, mask in (("full", target), ("high_k", selected)):
            count = mask.sum(axis=1)
            arrays[region + "_window_energy"][mode] = np.divide(np.where(mask, squared, 0.).sum(axis=1), count,
                out=np.full(len(count), np.nan), where=count > 0)
        total_energy = squared.sum(axis=1)
        arrays["high_k_energy_fraction"][mode] = np.divide(np.where(selected, squared, 0.).sum(axis=1), total_energy,
            out=np.full(len(valid), np.nan), where=total_energy > 0)
    metadata = {"definition": "squared one-score reconstruction minus zero-score reconstruction",
        "additive_contrasts": model.metadata["method"] in ("complete", "masked")
                              and model.metadata["representation"] in ("linear", "noise_weighted"),
        "additive_energy": False, "common_valid_cells": int(target.sum()),
        "high_k_valid_cells": int(selected.sum()), "high_k_geometric_features": int(high.sum()),
        "unavailable_reason": "rank-zero model has no mode contrasts" if model.rank == 0 else None,
        "normalization": "unweighted mean per objectively valid modeled cell in each physical window"}
    return arrays, metadata
