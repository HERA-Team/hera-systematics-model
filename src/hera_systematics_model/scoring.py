"""Predictive scoring on fixed support in linear power units."""

from dataclasses import dataclass

import numpy as np


class CandidateFailure(ValueError):
    """A candidate cannot supply valid predictions on its required support."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


def training_mean(residual, valid):
    residual, valid = np.asarray(residual), np.asarray(valid)
    if residual.ndim != 2 or valid.shape != residual.shape or valid.dtype.kind != "b":
        raise ValueError("training arrays need matching row/feature shapes")
    if not np.isfinite(residual[valid]).all():
        raise ValueError("nonfinite measured training residual")
    counts = valid.sum(axis=0)
    total = np.where(valid, residual, 0).sum(axis=0)
    return np.divide(total, counts, out=np.full(total.shape, np.nan), where=counts > 0)


@dataclass(frozen=True)
class Score:
    per_window: np.ndarray
    counts: np.ndarray
    mean: float


def score_predictions(prediction, truth, pn, target):
    """Score every required cell; predictions cannot alter the target mask."""
    prediction, truth, pn, target = map(np.asarray, (prediction, truth, pn, target))
    if (truth.ndim != 2 or any(a.shape != truth.shape for a in (prediction, pn, target))
            or target.dtype.kind != "b"):
        raise ValueError("score arrays must share row/feature shapes")
    if not np.isfinite(truth[target]).all() or not np.all(np.isfinite(pn[target]) & (pn[target] > 0)):
        raise ValueError("target mask includes invalid measurements or noise")
    if not np.isfinite(prediction[target]).all():
        raise CandidateFailure("nonfinite prediction on required target cells")
    counts = target.sum(axis=1)
    losses = np.full(len(truth), np.nan)
    for row in np.flatnonzero(counts):
        error = prediction[row, target[row]] - truth[row, target[row]]
        noise = pn[row, target[row]]
        weights = (noise.min() / noise) ** 2
        losses[row] = np.sum(weights * error ** 2) / weights.sum()
    if not np.isfinite(losses[counts > 0]).all():
        raise CandidateFailure("nonfinite weighted loss")
    if not (counts > 0).any():
        raise CandidateFailure("no eligible target cells")
    return Score(losses, counts, float(np.mean(losses[counts > 0])))


def choose_simplest(candidates, fold_losses):
    """Apply the one-standard-error rule across physical-time folds."""
    losses = np.asarray(fold_losses, dtype=float)
    if losses.ndim != 2 or len(candidates) != len(losses) or losses.shape[1] < 2:
        raise ValueError("need candidate by time-fold losses")
    usable = np.isfinite(losses).all(axis=1)
    if not usable.any():
        raise CandidateFailure("no candidate completed every required time fold")
    means = np.full(len(candidates), np.inf)
    means[usable] = losses[usable].mean(axis=1)
    best = int(np.argmin(means))
    standard_error = float(losses[best].std(ddof=1) / np.sqrt(losses.shape[1]))
    cutoff = float(means[best] + standard_error)
    methods = {"zero": 0, "mean": 1, "complete": 2, "masked": 3, "kernel": 4}
    representations = {"linear": 0, "noise_weighted": 1, "signed_asinh": 2, "log_ratio": 3}

    def complexity(index):
        candidate = candidates[index]
        return (candidate["rank"], methods[candidate["method"]],
                representations[candidate["representation"]], means[index], index)

    selected = min(np.flatnonzero(usable & (means <= cutoff)), key=complexity)
    return int(selected), {"best_index": best, "cutoff": cutoff, "standard_error": standard_error}
