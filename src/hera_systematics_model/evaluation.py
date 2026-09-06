"""Nested physical-time selection with disjoint target-feature prediction."""

from dataclasses import dataclass

import numpy as np

from .artifacts import write_artifact
from .prediction import candidate_grid, predict_partitioned
from .scoring import CandidateFailure, choose_simplest
from .splits import feature_partitions, time_folds


def select_within(arrays, window_ids, pool, partitions, candidates, guard=12, n_splits=3):
    """Choose a candidate using only rows in the supplied training pool."""
    folds = time_folds(window_ids, n_splits=n_splits, guard=guard, pool=pool)
    losses = np.full((len(candidates), len(folds)), np.nan)
    failures = []
    for fi, fold in enumerate(folds):
        if not fold.supported:
            failures.append({"fold": fi, "candidate": None, "reason": fold.reason})
            continue
        cache = {}
        for ci, candidate in enumerate(candidates):
            try:
                result = predict_partitioned(arrays, window_ids, fold.train, fold.test,
                                             partitions, candidate, cache=cache)
                losses[ci, fi] = result.loss.mean
            except (CandidateFailure, np.linalg.LinAlgError) as error:
                failures.append({"fold": fi, "candidate": ci, "reason": str(error)})
    selected, rule = choose_simplest(candidates, losses)
    return selected, losses, {"rule": rule, "failures": failures,
        "folds": [{"train_window_ids": window_ids[fold.train].tolist(),
                   "test_window_ids": window_ids[fold.test].tolist(),
                   "guard_window_ids": window_ids[fold.guard_rows].tolist(),
                   "interval": list(fold.interval)} for fold in folds]}


@dataclass
class Evaluation:
    arrays: dict
    metadata: dict
    models: dict

    def save(self, path):
        return write_artifact(path, "evaluation", self.arrays, self.metadata)


def evaluate_nested(arrays, window_ids, feature_shape, candidates=None, guard=12,
                    n_outer=4, n_inner=3, axis="delay"):
    """Freeze each outer choice before evaluating its physical-time block."""
    arrays = tuple(np.asarray(a) for a in arrays)
    window_ids = np.asarray(window_ids)
    if (len(arrays) != 4 or arrays[0].ndim != 2 or len(arrays[0]) != len(window_ids)
            or np.prod(feature_shape) != arrays[0].shape[1]):
        raise ValueError("evaluation axes do not match input arrays")
    candidates = candidate_grid() if candidates is None else candidates
    partitions = feature_partitions(feature_shape, axis=axis, guard=2 if axis == "delay" else 1)
    outer = time_folds(window_ids, n_splits=n_outer, guard=guard)
    shape = arrays[0].shape
    output = {"prediction": np.full(shape, np.nan), "target": np.zeros(shape, bool),
              "modeled": np.zeros(shape, bool), "mean_only": np.zeros(shape, bool),
              "zero_only": np.zeros(shape, bool),
              "window_loss": np.full(shape[0], np.nan), "outer_fold": np.full(shape[0], -1),
              "mean_baseline_loss": np.full(shape[0], np.nan),
              "zero_baseline_loss": np.full(shape[0], np.nan), "window_ids": window_ids.copy(),
              "feature_targets": np.stack([p.target for p in partitions]),
              "feature_predictors": np.stack([p.predictor for p in partitions]),
              "feature_guards": np.stack([p.guard for p in partitions])}
    reports, models = [], {}
    for fi, fold in enumerate(outer):
        report = {"outer_fold": fi, "train_window_ids": window_ids[fold.train].tolist(),
                  "test_window_ids": window_ids[fold.test].tolist(),
                  "guard_window_ids": window_ids[fold.guard_rows].tolist(), "interval": list(fold.interval)}
        reports.append(report)
        output["outer_fold"][fold.test] = fi
        if not fold.supported:
            report.update(status="insufficient_support", reason=fold.reason)
            continue
        try:
            selected, losses, inner = select_within(arrays, window_ids, fold.train, partitions,
                                                    candidates, guard=guard, n_splits=n_inner)
            output[f"inner_losses_{fi}"] = losses
            report.update(selected=candidates[selected], selected_index=selected, inner=inner)
            result = predict_partitioned(arrays, window_ids, fold.train, fold.test, partitions,
                                         candidates[selected], keep_models=True)
            for name in ("prediction", "target", "modeled", "mean_only", "zero_only"):
                output[name][fold.test] = getattr(result, name)
            output["window_loss"][fold.test] = result.loss.per_window
            models[fi] = result.models
            for baseline in ("zero", "mean"):
                reference = predict_partitioned(arrays, window_ids, fold.train, fold.test, partitions,
                    {"method": baseline, "rank": 0, "representation": "linear"})
                output[f"{baseline}_baseline_loss"][fold.test] = reference.loss.per_window
            report.update(status="evaluated", loss=result.loss.mean, scored_cells=int(result.target.sum()),
                          modeled_cells=int(result.modeled.sum()), mean_only_cells=int(result.mean_only.sum()),
                          zero_only_cells=int(result.zero_only.sum()))
        except (CandidateFailure, np.linalg.LinAlgError) as error:
            report.update(status="candidate_failure", reason=str(error))
    metadata = {"guard_windows": guard, "outer_folds": n_outer, "inner_folds": n_inner,
                "feature_shape": list(feature_shape), "feature_axis": axis,
                "candidates": candidates, "folds": reports,
                "complete": all(r["status"] == "evaluated" for r in reports)}
    return Evaluation(output, metadata, models)
