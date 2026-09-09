"""Apply the predeclared pilot conclusion rule to saved fold evidence."""

import math

import numpy as np

from .evidence import fold_summary

LABELS = {
    "useful": "useful predictive structure",
    "no_advantage": "no demonstrated advantage over the baseline",
    "insufficient": "insufficient evidence under the declared validation design",
}


def predictive_conclusion(summary):
    """Classify one SPW without changing the predeclared four-fold rule."""
    if summary.get("schema_version") != 1 or "identity" not in summary:
        raise ValueError("versioned evidence with a physical identity is required")
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ValueError("exactly four outer physical-time folds are required")
    if [fold.get("outer_fold") for fold in folds] != list(range(4)):
        raise ValueError("outer physical-time fold identities must be ordered")

    improvements = []
    baseline_choices = []
    complete_folds = []
    unavailable = []
    for fold in folds:
        losses = fold.get("losses", {})
        values = {
            name: losses.get(name, {}).get("mean_window_loss")
            for name in ("selected", "zero", "mean")
        }
        coverage = fold.get("coverage", {})
        target = coverage.get("target", {})
        complete = (
            fold.get("status") == "evaluated"
            and fold.get("scored_windows", 0) > 0
            and coverage.get("eligible_cells", 0) > 0
            and target.get("cells", 0) > 0
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value) for value in values.values())
        )
        if not complete:
            improvements.append(np.nan)
            baseline_choices.append(None)
            unavailable.append({
                "outer_fold": fold["outer_fold"],
                "reason": "fold lacks complete common-plane selected and baseline scores",
            })
            continue
        baseline = "zero" if values["zero"] <= values["mean"] else "mean"
        baseline_choices.append(baseline)
        improvements.append(values[baseline] - values["selected"])
        complete_folds.append(fold["outer_fold"])

    improvement = fold_summary(improvements, 4)
    all_folds = len(complete_folds) == 4
    positive = bool(all_folds and improvement["mean"] > 0)
    exceeds_one_se = bool(
        all_folds
        and improvement["standard_error"] is not None
        and improvement["mean"] > improvement["standard_error"])
    if not all_folds:
        key = "insufficient"
    elif positive and exceeds_one_se:
        key = "useful"
    else:
        key = "no_advantage"

    return {
        "schema_version": 1,
        "identity": summary["identity"],
        "conclusion": LABELS[key],
        "conclusion_key": key,
        "complete_common_plane_folds": complete_folds,
        "unavailable_folds": unavailable,
        "better_baseline_by_fold": baseline_choices,
        "better_baseline_minus_selected": improvement,
        "criteria": {
            "four_complete_outer_folds": all_folds,
            "positive_mean_improvement": positive,
            "mean_improvement_exceeds_one_standard_error": exceeds_one_se,
        },
        "rule": {
            "outer_folds": 4,
            "baseline": "lower loss of zero-residual and learned-mean within each fold",
            "threshold": "positive mean fold improvement exceeding one physical-time-fold standard error",
            "feature_partitions_are_independent_realizations": False,
        },
        "limitations": summary.get("limitations", []),
    }
