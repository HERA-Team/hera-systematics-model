"""Reproduce saved coefficients, decoded predictions and physical-window losses."""

import numpy as np

from .evaluation_state import validate_evaluation
from .models import measured_arrays
from .reconstruction import decode_scores
from .scoring import score_predictions, training_mean


def require_close(left, right, label, rtol=1e-10, atol=1e-12):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
        raise ValueError(f"saved evaluation replay mismatch: {label}")


def replay_evaluation(arrays, window_ids, evaluation, training_filter=None, rtol=1e-10, noise_atol=1e-10):
    """Recompute inference without refitting modes or repeating model selection."""
    if not np.isfinite(rtol) or not np.isfinite(noise_atol) or min(rtol, noise_atol) < 0:
        raise ValueError("finite nonnegative replay tolerances required")
    arrays = measured_arrays(*arrays)
    saved, metadata = evaluation.arrays, evaluation.metadata
    validate_evaluation(saved, metadata, evaluation.models)
    if (metadata.get("purpose") == "descriptive_fit" or not np.array_equal(window_ids, saved["window_ids"])
            or not np.array_equal(arrays[3], saved["eligible"])):
        raise ValueError("replay requires matching predictive time and support identities")
    lookup = {int(identity): row for row, identity in enumerate(window_ids)}
    reports = []
    for report in metadata["folds"]:
        fold = report["outer_fold"]
        if report["status"] != "evaluated":
            reports.append({"outer_fold": fold, "replayed": False, "reason": report["status"]})
            continue
        train, test = (np.array([lookup[int(value)] for value in report[name]], int)
                       for name in ("train_window_ids", "test_window_ids"))
        filtered = arrays if training_filter is None else training_filter(arrays, train)[0]
        power, ideal, pn, valid = filtered
        mean = training_mean((power - ideal)[train], valid[train])
        target = valid[test] & np.isfinite(mean)
        if (not np.array_equal(target, saved["target"][test])
                or not np.array_equal(arrays[3][test] & ~valid[test], saved["excluded"][test])):
            raise ValueError("replayed target support or exclusions differ")
        prediction = np.full(target.shape, np.nan)
        modeled = np.zeros(target.shape, bool)
        coverage = np.zeros(target.shape, int)
        models = evaluation.models.get(fold, evaluation.models.get(str(fold), []))
        coefficient_rows = 0
        for inference in report["inference"]:
            model = models[inference["model_index"]]
            prefix = inference["array_prefix"]
            partition = inference["partition_index"]
            selected = saved["feature_targets"][partition]
            predictor = saved["feature_predictors"][partition]
            scores = saved[prefix + "_scores"]
            active = (target & model.feature_mask & selected).any(axis=1) if model.rank else np.zeros(len(test), bool)
            if prefix + "_coefficients_inferred" in saved and not np.array_equal(active, saved[prefix + "_coefficients_inferred"]):
                raise ValueError("replayed coefficient rows differ")
            values = np.broadcast_to(mean, target.shape).copy()
            if model.metadata["method"] == "zero":
                values[:] = 0
            elif active.any():
                _, coefficients = model.predict(*(a[test[active]] for a in filtered), predictor=predictor)
                require_close(coefficients, scores[active], "predictor-only coefficients")
                values[active] = decode_scores(model, scores[active], ideal[test[active]], pn[test[active]])
            coefficient_rows += int(active.sum())
            prediction[:, selected] = values[:, selected]
            modeled[:, selected] = target[:, selected] & model.feature_mask[selected]
            coverage[:, selected] += target[:, selected]
        if not np.array_equal(coverage, target.astype(int)) or not np.array_equal(modeled, saved["modeled"][test]):
            raise ValueError("replayed inference partitions or modeled coverage differ")
        error = np.abs(prediction[target] - saved["prediction"][test][target])
        tolerance = rtol * np.abs(saved["prediction"][test][target]) + noise_atol * pn[test][target]
        if not np.isfinite(error).all() or np.any(error > tolerance):
            raise ValueError("saved evaluation replay mismatch: decoded target predictions")
        for label, predicted in (("selected", saved["prediction"][test]),
                                 ("zero", np.zeros(target.shape)), ("mean", np.broadcast_to(mean, target.shape))):
            loss = score_predictions(predicted, (power - ideal)[test], pn[test], target)
            key = "window_loss" if label == "selected" else label + "_baseline_loss"
            require_close(loss.per_window, saved[key][test], label + " physical-window losses", rtol=1e-12, atol=0)
            if label == "selected":
                require_close(loss.mean, report["loss"], "fold mean loss", rtol=1e-12, atol=0)
        reports.append({"outer_fold": fold, "replayed": True, "target_cells": int(target.sum()),
                        "coefficient_rows": coefficient_rows, "maximum_prediction_error": float(error.max())})
    return {"passed": True, "evaluation_complete": metadata["complete"], "folds": reports,
        "prediction_rtol": rtol, "prediction_noise_atol": noise_atol,
        "coefficient_rtol": 1e-10, "coefficient_atol": 1e-12,
        "loss_rtol": 1e-12, "loss_atol": 0., "basis_refitted": False, "selection_repeated": False}
