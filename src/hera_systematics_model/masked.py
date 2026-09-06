"""Low-rank least squares on observed entries, with explicit convergence."""

import numpy as np

from .models import LinearModel, prepare_training
from .scoring import CandidateFailure, training_mean


def fit_masked(power, ideal, pn, valid, rank, representation="linear", log_margin=1.,
               training_ids=None, max_iter=500, tolerance=1e-7):
    """Fit a mean and factors without treating absent entries as measurements."""
    x, observed, linear_mean, params = prepare_training(power, ideal, pn, valid, representation, log_margin)
    if not isinstance(rank, int) or rank < 0 or rank > len(x) - 2:
        raise CandidateFailure("rank exceeds training support")
    if max_iter < 1 or not 0 < tolerance < 1:
        raise ValueError("invalid convergence settings")
    feature_mask = observed.sum(axis=0) >= max(1, rank + 2)
    if feature_mask.sum() < rank + 2:
        raise CandidateFailure("insufficient identifiable features")
    values, mask = x[:, feature_mask], observed[:, feature_mask]
    if np.any(mask.sum(axis=1) < rank + 2):
        raise CandidateFailure("insufficient observed cells in a training row")
    mean = training_mean(values, mask)
    # Filling is only an initialization; every subsequent objective uses mask.
    centered = np.where(mask, values - mean, 0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if rank and np.count_nonzero(s > s[0] * 1e-10) < rank:
        raise CandidateFailure("rank exceeds initial numerical support")
    scores, basis = u[:, :rank] * s[:rank], vt[:rank].copy()
    column_patterns, column_groups = np.unique(mask.T, axis=0, return_inverse=True)
    row_patterns, row_groups = np.unique(mask, axis=0, return_inverse=True)

    def objective():
        error = np.where(mask, values - mean - scores @ basis, 0)
        return float(np.sum(error ** 2))

    initial = previous = objective()
    numerical_floor = np.finfo(float).eps * max(float(np.sum(centered ** 2)), 1.)
    converged = rank == 0
    iteration = 0
    for iteration in range(1, max_iter + 1) if rank else ():
        for group, usable in enumerate(row_patterns):
            rows = np.flatnonzero(row_groups == group)
            fit, _, effective, _ = np.linalg.lstsq(
                basis[:, usable].T, (values[np.ix_(rows, usable)] - mean[usable]).T, rcond=1e-10)
            if effective < rank:
                raise CandidateFailure("rank-deficient masked row solve")
            scores[rows] = fit.T
        for group, usable in enumerate(column_patterns):
            columns = np.flatnonzero(column_groups == group)
            design = np.column_stack([np.ones(usable.sum()), scores[usable]])
            fit, _, effective, _ = np.linalg.lstsq(design, values[np.ix_(usable, columns)], rcond=1e-10)
            if effective < rank + 1:
                raise CandidateFailure("rank-deficient masked feature solve")
            mean[columns], basis[:, columns] = fit[0], fit[1:]
        score_mean = scores.mean(axis=0)
        mean += score_mean @ basis
        scores -= score_mean
        q, r = np.linalg.qr(basis.T, mode="reduced")
        basis, scores = q.T, scores @ r.T
        current = objective()
        if not np.isfinite(current) or current > previous + 1e-10 * max(initial, 1.):
            raise CandidateFailure("masked objective is nonfinite or increased")
        if abs(previous - current) <= tolerance * max(previous, numerical_floor):
            converged = True
            previous = current
            break
        previous = current
    if rank:
        _, singular, rotation = np.linalg.svd(scores, full_matrices=False)
        basis = rotation @ basis
        signs = np.sign(basis[np.arange(rank), np.argmax(np.abs(basis), axis=1)])
        basis *= np.where(signs == 0, 1, signs)[:, None]
    else:
        singular = np.empty(0)
    full_mean = np.zeros(x.shape[1])
    full_mean[feature_mask] = mean
    components = np.zeros((rank, x.shape[1]))
    components[:, feature_mask] = basis
    ids = np.arange(len(x)) if training_ids is None else np.asarray(training_ids)
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid training identities")
    return LinearModel(components, full_mean, linear_mean, feature_mask, singular,
        np.empty(0), ids, {"method": "masked", "representation": representation,
         "params": params, "rank": rank, "converged": converged, "iterations": iteration,
         "initial_observed_loss": initial, "observed_loss": previous,
         "objective_numerical_floor": numerical_floor,
         "tolerance": tolerance, "max_iter": max_iter})
