"""Semantic validation of saved time partitions and predictive support."""

import numpy as np


def validate_evaluation(arrays, metadata, models):
    ids = np.asarray(arrays["window_ids"])
    dimensions = metadata["feature_shape"]
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or np.any(np.diff(ids) <= 0)
            or len(dimensions) != 2 or min(dimensions) < 1 or type(metadata.get("complete")) is not bool):
        raise ValueError("invalid evaluation time or feature identities")
    shape = (len(ids), int(np.prod(dimensions)))
    parts = [np.asarray(arrays[name]) for name in ("feature_targets", "feature_predictors", "feature_guards")]
    if (parts[0].ndim != 2 or parts[0].shape[1] != shape[1]
            or any(part.shape != parts[0].shape or part.dtype.kind != "b" for part in parts)
            or not np.all(parts[0].sum(axis=0) == 1)
            or np.any(parts[0].astype(int) + parts[1] + parts[2] != 1)):
        raise ValueError("feature partitions overlap or fail to cover coordinates")

    def array(name, dtype=None):
        value = np.asarray(arrays[name])
        if value.shape != shape or (dtype is not None and value.dtype.kind != dtype):
            raise ValueError(f"invalid evaluation dimensions or type: {name}")
        return value

    if metadata.get("purpose") == "descriptive_fit":
        if not metadata["complete"]:
            return
        target = array("target", "b")
        prediction = array("training_prediction")
        entries = models.get("descriptive", [])
        if (len(entries) != 1 or not np.array_equal(entries[0].training_ids, ids)
                or arrays["training_scores"].shape != (len(ids), entries[0].rank)
                or not np.isfinite(prediction[target]).all()):
            raise ValueError("descriptive model state does not support saved predictions")
        return

    target = array("target", "b")
    masks = [array(name, "b") for name in ("modeled", "mean_only", "zero_only")]
    eligible, unavailable = array("eligible", "b"), array("unavailable", "b")
    excluded = array("excluded", "b") if "excluded" in arrays else np.zeros(shape, bool)
    if (not np.array_equal(sum(mask.astype(int) for mask in masks), target.astype(int))
            or not np.array_equal(target.astype(int) + unavailable + excluded, eligible.astype(int))
            or not np.isfinite(array("prediction")[target]).all()):
        raise ValueError("prediction coverage masks or finite target support disagree")
    fold_ids = np.asarray(arrays["outer_fold"])
    if fold_ids.shape != (len(ids),) or fold_ids.dtype.kind not in "iu":
        raise ValueError("invalid outer-fold identity array")
    for report in metadata["folds"]:
        index = report["outer_fold"]
        train, test, guard = (np.asarray(report[name], dtype=int) for name in
                              ("train_window_ids", "test_window_ids", "guard_window_ids"))
        combined = np.concatenate([train, test, guard])
        if (len(combined) != len(ids) or not np.array_equal(np.sort(combined), ids)
                or not np.array_equal(ids[fold_ids == index], test)):
            raise ValueError("saved physical-time partitions overlap or omit windows")
        if len(train) and len(test) and np.min(np.abs(train[:, None] - test)) <= metadata["guard_windows"]:
            raise ValueError("saved physical-time guard is violated")
        if report["status"] != "evaluated":
            continue
        entries = models.get(index, models.get(str(index), []))
        if not entries or any(not np.array_equal(model.training_ids, train) for model in entries):
            raise ValueError("saved predictive model training identities disagree")
        for inference in report.get("inference", []):
            prefix = inference["array_prefix"]
            support = np.asarray(arrays[f"{prefix}_predictor_support"])
            model = entries[inference["model_index"]]
            predictor = parts[1][inference["partition_index"]]
            scores = np.asarray(arrays[f"{prefix}_scores"])
            if (support.shape != (len(test), shape[1]) or support.dtype.kind != "b"
                    or np.any(support & ~predictor) or scores.shape != (len(test), model.rank)
                    or not np.isfinite(scores).all()):
                raise ValueError("saved coefficient inference uses invalid predictor support")
