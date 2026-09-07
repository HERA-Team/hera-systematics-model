"""Candidate definitions and held-feature prediction for a fixed time split."""

from dataclasses import dataclass, replace

import numpy as np

from .kernel import fit_kernel
from .masked import fit_masked
from .models import fit_complete, measured_arrays
from .model_io import ConstantModel
from .scoring import CandidateFailure, score_predictions, training_mean


def candidate_grid(max_rank=20, include_kernel=True, representations=None, methods=None):
    representations = representations or ["linear", "noise_weighted", "signed_asinh", "log_ratio"]
    methods = methods or ["complete", "masked"]
    result = [{"rank": 0, "method": method, "representation": "linear"} for method in ("zero", "mean")]
    for representation in representations:
        margins = [1., 3., 10.] if representation == "log_ratio" else [1.]
        for margin in margins:
            for method in methods:
                for rank in range(1, max_rank + 1):
                    result.append({"rank": rank, "method": method, "representation": representation,
                                   "log_margin": margin})
            if include_kernel:
                for rank in range(1, min(10, max_rank) + 1):
                    for bandwidth in (.1, .3, 1., 3., 10.):
                        for alpha in (.1, 1., 10.):
                            result.append({"rank": rank, "method": "kernel", "representation": representation,
                                           "log_margin": margin, "bandwidth": bandwidth, "alpha": alpha})
    return result


@dataclass
class PredictionResult:
    prediction: np.ndarray
    target: np.ndarray
    modeled: np.ndarray
    mean_only: np.ndarray
    zero_only: np.ndarray
    loss: object
    models: list
    eligible: np.ndarray
    unavailable: np.ndarray
    inference: list


def fit_candidate(arrays, train, candidate, predictor, window_ids, cache=None):
    method = candidate["method"]
    if method in ("zero", "mean"):
        return ConstantModel.fit([a[train] for a in arrays], window_ids[train], method)
    arguments = {key: value for key, value in candidate.items() if key != "method"}
    train_arrays = [a[train] for a in arrays]
    if method == "complete":
        key = (tuple(train), candidate["representation"], candidate.get("log_margin", 1.))
        base = None if cache is None else cache.get(key)
        if base is None:
            base = fit_complete(*train_arrays, **{**arguments, "rank": 0}, training_ids=window_ids[train])
            if cache is not None:
                cache[key] = base
        rank = candidate["rank"]
        singular = base.singular_values
        support = np.count_nonzero(singular > (singular[0] * 1e-10 if len(singular) else 0))
        if rank > min(support, len(train) - 2):
            raise CandidateFailure("rank exceeds training numerical support")
        return replace(base, metadata={**base.metadata, "rank": rank})
    if method == "masked":
        return fit_masked(*train_arrays, **arguments, training_ids=window_ids[train])
    if method == "kernel":
        return fit_kernel(*train_arrays, **arguments, predictor=predictor, training_ids=window_ids[train])
    raise ValueError("unknown residual model method")


def predict_partitioned(arrays, window_ids, train, test, partitions, candidate, cache=None,
                        keep_models=False):
    """Predict targets from disjoint features with one fixed training partition."""
    arrays = measured_arrays(*arrays)
    power, ideal, pn, valid = arrays
    train, test = np.asarray(train), np.asarray(test)
    if len(np.intersect1d(train, test)):
        raise ValueError("training and validation rows overlap")
    mean = training_mean((power - ideal)[train], valid[train])
    target = valid[test] & np.isfinite(mean)[None, :]
    prediction = np.full(target.shape, np.nan)
    modeled = np.zeros_like(target)
    covered = np.zeros(power.shape[1], dtype=int)
    models, inference = [], []
    model = None
    for partition_index, partition in enumerate(partitions):
        covered += partition.target
        if not partition.target.any() or not target[:, partition.target].any():
            continue
        if model is None or candidate["method"] == "kernel":
            model = fit_candidate(arrays, train, candidate, partition.predictor, window_ids, cache)
            if keep_models and model is not None:
                models.append(model)
        if candidate["method"] == "zero":
            pred = np.zeros(target.shape)
            scores, diagnostics = np.zeros((len(test), 0)), []
        elif candidate["method"] == "mean":
            pred = np.broadcast_to(mean, target.shape)
            scores, diagnostics = np.zeros((len(test), 0)), []
        else:
            diagnostics = [] if keep_models else None
            pred = np.broadcast_to(mean, target.shape).copy()
            scores = np.zeros((len(test), model.rank))
            # Coefficients are needed only where this partition has a valid
            # target supported by modes. Other targets use the shared mean.
            infer = np.flatnonzero((target & model.feature_mask & partition.target).any(axis=1))
            if len(infer):
                predicted, inferred_scores = model.predict(*(a[test[infer]] for a in arrays),
                    predictor=partition.predictor, diagnostics=diagnostics)
                pred[infer], scores[infer] = predicted, inferred_scores
                if diagnostics is not None:
                    for entry in diagnostics:
                        entry["row"] = int(infer[entry["row"]])
            modeled[:, partition.target] = model.feature_mask[partition.target] & target[:, partition.target]
        if keep_models:
            support = np.zeros(target.shape, bool)
            ranks = np.zeros(len(test), int)
            conditions = np.full(len(test), np.nan)
            for row in diagnostics:
                support[row["row"], row["predictor_features"]] = True
                ranks[row["row"]] = row["effective_rank"]
                conditions[row["row"]] = row["condition_number"]
            inference.append({"scores": scores, "predictor_support": support,
                "effective_rank": ranks, "condition_number": conditions,
                "coefficients_inferred": support.any(axis=1),
                "inactive_score_encoding": "zero placeholder where coefficients_inferred is false",
                "partition_index": partition_index, "model_index": len(models) - 1,
                "type": "kernel_embedding" if candidate["method"] == "kernel" else "linear_coefficients",
                "condition_definition": "retained training kernel eigenvalue ratio" if candidate["method"] == "kernel"
                                        else "observed predictor design singular value ratio",
                "condition_unavailable_reason": "no coefficient solve for rank zero" if candidate["rank"] == 0 else None})
        prediction[:, partition.target] = pred[:, partition.target]
    if not np.all(covered == 1):
        raise ValueError("feature partitions must cover each target exactly once")
    loss = score_predictions(prediction, (power - ideal)[test], pn[test], target)
    zero_only = target & (candidate["method"] == "zero")
    return PredictionResult(prediction, target, modeled, target & ~modeled & ~zero_only,
                            zero_only, loss, models, valid[test].copy(), valid[test] & ~target, inference)
