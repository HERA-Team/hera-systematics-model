"""Nested physical-time selection with disjoint target-feature prediction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import read_artifact, write_artifact
from .model_io import load_model_collection, save_model_collection
from .prediction import candidate_grid, predict_partitioned
from .scoring import CandidateFailure, choose_simplest
from .splits import feature_partitions, time_folds


def structural_rank_ceiling(valid, folds, partitions, candidate, candidates):
    """Bound rank by physical splits and observed predictor counts, not EVR."""
    if candidate["rank"] == 0:
        return 0
    maximum = max(c["rank"] for c in candidates if c["method"] == candidate["method"]
                  and c["representation"] == candidate["representation"])
    for rank in range(maximum, 0, -1):
        supported = True
        for fold in folds:
            if not fold.supported or len(fold.train) < rank + 2:
                supported = False
                break
            counts = valid[fold.train].sum(axis=0)
            features = counts >= rank + 2 if candidate["method"] == "masked" else counts == len(fold.train)
            for partition in partitions:
                if partition.target.any() and np.any((valid[fold.test] & features & partition.predictor).sum(axis=1) < rank + 2):
                    supported = False
                    break
        if supported:
            return rank
    return 0


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
    try:
        selected, rule = choose_simplest(candidates, losses)
    except CandidateFailure as error:
        selected, rule = None, {"reason": str(error)}
    if selected is not None:
        ceiling = structural_rank_ceiling(arrays[3], folds, partitions, candidates[selected], candidates)
        rule.update(structural_rank_ceiling=ceiling,
                    rank_ceiling_selected=bool(candidates[selected]["rank"] > 0 and candidates[selected]["rank"] >= ceiling))
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
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(path)
        metadata = dict(self.metadata)
        metadata["models"] = save_model_collection(path.with_suffix(".models"), self.models,
                                                  metadata.get("identity"))
        return write_artifact(path, "evaluation", self.arrays, metadata)

    @classmethod
    def load(cls, path):
        arrays, metadata = read_artifact(path, "evaluation")
        models = load_model_collection(Path(path).parent, metadata["models"], metadata.get("identity"))
        return cls(arrays, metadata, models)


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
              "eligible": arrays[3].copy(), "unavailable": arrays[3].copy(),
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
            report.update(selected=candidates[selected] if selected is not None else None,
                          selected_index=selected, inner=inner)
            if selected is None:
                raise CandidateFailure(inner["rule"]["reason"])
            result = predict_partitioned(arrays, window_ids, fold.train, fold.test, partitions,
                                         candidates[selected], keep_models=True)
            for name in ("prediction", "target", "modeled", "mean_only", "zero_only", "eligible", "unavailable"):
                output[name][fold.test] = getattr(result, name)
            output["window_loss"][fold.test] = result.loss.per_window
            models[fi] = result.models
            for baseline in ("zero", "mean"):
                reference = predict_partitioned(arrays, window_ids, fold.train, fold.test, partitions,
                    {"method": baseline, "rank": 0, "representation": "linear"})
                output[f"{baseline}_baseline_loss"][fold.test] = reference.loss.per_window
            report.update(status="evaluated", loss=result.loss.mean, scored_cells=int(result.target.sum()),
                          eligible_cells=int(result.eligible.sum()), unavailable_cells=int(result.unavailable.sum()),
                          modeled_cells=int(result.modeled.sum()), mean_only_cells=int(result.mean_only.sum()),
                          zero_only_cells=int(result.zero_only.sum()))
        except (CandidateFailure, np.linalg.LinAlgError) as error:
            report.update(status="candidate_failure", reason=str(error))
    metadata = {"guard_windows": guard, "outer_folds": n_outer, "inner_folds": n_inner,
                "feature_shape": list(feature_shape), "feature_axis": axis,
                "candidates": candidates, "folds": reports,
                "complete": all(r["status"] == "evaluated" for r in reports)}
    return Evaluation(output, metadata, models)
