"""Final descriptive fits selected independently of outer-fold outcomes."""

import numpy as np

from .evaluation import Evaluation, select_within
from .models import measured_arrays
from .prediction import candidate_grid, fit_candidate
from .scoring import CandidateFailure, score_predictions, training_mean
from .splits import feature_partitions


def select_final_fit(arrays, window_ids, feature_shape, candidates=None, guard=12,
                     n_inner=3, axis="delay"):
    """Use inner selection on all available rows, then fit a descriptive model.

    Training reconstructions from this model are not out-of-fold predictions.
    Kernel fits use every eligible predictor feature for descriptive embedding.
    """
    arrays = measured_arrays(*arrays)
    window_ids = np.asarray(window_ids)
    if window_ids.shape != (len(arrays[0]),) or np.prod(feature_shape) != arrays[0].shape[1]:
        raise ValueError("descriptive fit identity dimensions disagree")
    candidates = candidate_grid() if candidates is None else candidates
    partitions = feature_partitions(feature_shape, axis=axis, guard=2 if axis == "delay" else 1)
    rows = np.arange(len(window_ids))
    output = {"window_ids": window_ids.copy(),
              "feature_targets": np.stack([p.target for p in partitions]),
              "feature_predictors": np.stack([p.predictor for p in partitions]),
              "feature_guards": np.stack([p.guard for p in partitions])}
    metadata = {"purpose": "descriptive_fit", "performance_estimator": False,
                "guard_windows": guard, "inner_folds": n_inner, "feature_axis": axis,
                "feature_shape": list(feature_shape), "candidates": candidates, "complete": False}
    models = {}
    try:
        selected, losses, inner = select_within(arrays, window_ids, rows, partitions,
                                                candidates, guard, n_inner)
        output["inner_losses"] = losses
        metadata["inner"] = inner
        if selected is None:
            raise CandidateFailure(inner["rule"]["reason"])
        candidate = candidates[selected]
        metadata.update(selected=candidate, selected_index=selected, inner=inner)
        model = fit_candidate(arrays, rows, candidate, np.ones(arrays[0].shape[1], bool), window_ids)
        prediction, scores = model.predict(*arrays, np.ones(arrays[0].shape[1], bool))
        target = arrays[3] & np.isfinite(training_mean(arrays[0] - arrays[1], arrays[3]))
        score = score_predictions(prediction, arrays[0] - arrays[1], arrays[2], target)
        output.update(training_prediction=prediction, training_scores=scores, target=target,
                      modeled=target & model.feature_mask, training_window_loss=score.per_window)
        ranks = [c["rank"] for c in candidates if c["method"] == candidate["method"]
                 and c["representation"] == candidate["representation"]]
        metadata.update(complete=True, status="fitted", training_loss=score.mean,
                        configured_rank_ceiling=max(ranks),
                        rank_ceiling_selected=bool(candidate["rank"] > 0 and candidate["rank"] == max(ranks)))
        models["descriptive"] = [model]
    except (CandidateFailure, np.linalg.LinAlgError) as error:
        metadata.update(status="candidate_failure", reason=str(error))
    return Evaluation(output, metadata, models)
