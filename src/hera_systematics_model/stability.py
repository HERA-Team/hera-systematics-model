"""Segment-preserving block resampling and signed/subspace comparisons."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .prediction import fit_candidate
from .scoring import CandidateFailure
from .splits import continuous_segments


def block_bootstrap_indices(window_ids, block_length, rng):
    """Sample noncircular blocks separately within each continuous time segment.

    Segment sample sizes remain fixed. The final block in each segment is
    truncated to that size. Edge rows have different inclusion probabilities.
    Segments shorter than the requested block length are unsupported.
    """
    ids = np.asarray(window_ids)
    if type(block_length) is not int or block_length < 1:
        raise ValueError("positive integer block length required")
    rows, starts, lengths = [], [], []
    for segment in continuous_segments(ids, np.arange(len(ids))):
        if len(segment) < block_length:
            raise CandidateFailure("continuous time segment shorter than bootstrap block")
        remaining = len(segment)
        while remaining:
            start = int(rng.integers(0, len(segment) - block_length + 1))
            size = min(remaining, block_length)
            rows.extend(segment[start:start + size])
            starts.append(int(segment[start]))
            lengths.append(size)
            remaining -= size
    return np.asarray(rows, int), np.asarray(starts, int), np.asarray(lengths, int)


def compare_components(reference, trial, reference_mask=None, trial_mask=None):
    """Match signed directions and compare full subspaces on shared features."""
    reference, trial = np.asarray(reference, float), np.asarray(trial, float)
    if reference.ndim != 2 or trial.ndim != 2 or reference.shape != trial.shape:
        raise ValueError("component arrays must have matching rank and feature axes")
    k, nf = reference.shape
    if k == 0:
        return {"signed_cosines": np.empty(0), "principal_angles": np.empty(0),
                "assignment": np.empty(0, int), "signs": np.empty(0), "common_features": nf}
    mask = np.ones(nf, bool)
    for value in (reference_mask, trial_mask):
        if value is not None:
            value = np.asarray(value)
            if value.shape != (nf,) or value.dtype.kind != "b":
                raise ValueError("invalid component feature support")
            mask &= value
    a, b = reference[:, mask], trial[:, mask]
    if not np.isfinite(a).all() or not np.isfinite(b).all() or mask.sum() < k:
        raise CandidateFailure("insufficient shared component support")
    norms = [np.linalg.norm(x, axis=1) for x in (a, b)]
    if any(np.any(n == 0) for n in norms):
        raise CandidateFailure("component vanishes on shared support")
    cosine = (a / norms[0][:, None]) @ (b / norms[1][:, None]).T
    left, right = linear_sum_assignment(-np.abs(cosine))
    matched = np.clip(cosine[left, right], -1, 1)
    bases = []
    for values in (a, b):
        _, singular, vt = np.linalg.svd(values, full_matrices=False)
        if np.count_nonzero(singular > singular[0] * 1e-10) < k:
            raise CandidateFailure("rank-deficient shared subspace")
        bases.append(vt)
    overlap = np.linalg.svd(bases[0] @ bases[1].T, compute_uv=False)
    return {"signed_cosines": matched, "principal_angles": np.arccos(np.clip(overlap, 0, 1)),
            "assignment": right, "signs": np.where(matched < 0, -1., 1.), "common_features": int(mask.sum())}


def bootstrap_stability(arrays, window_ids, candidate, n_replicates=500, block_length=12, seed=0):
    """Refit the fixed selected configuration, including training preprocessing."""
    if n_replicates < 1:
        raise ValueError("at least one bootstrap replicate required")
    arrays = tuple(np.asarray(a) for a in arrays)
    ids = np.asarray(window_ids)
    rng = np.random.default_rng(seed)
    rows = np.arange(len(ids))
    predictors = np.ones(arrays[0].shape[1], bool)
    reference = fit_candidate(arrays, rows, candidate, predictors, ids)
    rank = candidate["rank"]
    result = {"sample_rows": np.full((n_replicates, len(ids)), -1, int),
              "signed_cosines": np.full((n_replicates, rank), np.nan),
              "principal_angles": np.full((n_replicates, rank), np.nan),
              "common_features": np.zeros(n_replicates, int)}
    records = []
    kernel = candidate["method"] == "kernel"
    if kernel:
        _, reference_scores = reference.predict(*arrays, predictors)
        ref_components = (reference_scores - reference_scores.mean(axis=0)).T
    else:
        ref_components = reference.components[:rank] if rank else np.empty((0, len(predictors)))
    for index in range(n_replicates):
        try:
            sampled, starts, lengths = block_bootstrap_indices(ids, block_length, rng)
            result["sample_rows"][index] = sampled
            replicate = fit_candidate(tuple(a[sampled] for a in arrays), rows, candidate, predictors, rows)
            if not replicate.metadata["converged"]:
                raise CandidateFailure("bootstrap factorization did not converge")
            if kernel:
                _, scores = replicate.predict(*arrays, predictors)
                compared = compare_components(ref_components, (scores - scores.mean(axis=0)).T)
            else:
                trial = replicate.components[:rank] if rank else np.empty_like(ref_components)
                compared = compare_components(ref_components, trial, reference.feature_mask, replicate.feature_mask)
            for key in ("signed_cosines", "principal_angles", "common_features"):
                result[key][index] = compared[key]
            records.append({"replicate": index, "status": "evaluated", "block_start_rows": starts.tolist(),
                            "block_lengths": lengths.tolist(), "iterations": replicate.metadata.get("iterations")})
        except (CandidateFailure, np.linalg.LinAlgError) as error:
            records.append({"replicate": index, "status": "unavailable", "reason": str(error)})
    return result, {"candidate": candidate, "replicates": n_replicates, "block_length": block_length,
        "seed": seed, "comparison_space": "scores on original windows" if kernel else "selected representation features",
        "resampling": "noncircular moving blocks within fixed continuous segments", "records": records,
        "complete": all(item["status"] == "evaluated" for item in records)}
