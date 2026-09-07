"""Compact numerical evidence from saved physical-time evaluations."""

import numpy as np

from .configuration import AnalysisConfig
from .evaluation_state import validate_evaluation


def fold_summary(values, expected):
    """Summarize equally weighted time folds without pooling target partitions."""
    values = np.asarray(values, float)
    if type(expected) is not int or expected < 1 or values.shape != (expected,):
        raise ValueError("one loss per declared physical-time fold required")
    finite = values[np.isfinite(values)]
    complete = len(finite) == expected
    return {"expected_folds": expected, "available_folds": len(finite), "complete": complete,
        "mean": float(finite.mean()) if complete else None,
        "standard_error": float(finite.std(ddof=1) / np.sqrt(expected)) if complete and expected > 1 else None,
        "unavailable_reason": None if complete and expected > 1 else "incomplete or fewer than two time folds",
        "uncertainty_unit": "physical-time fold", "independent_realizations": False}


def coverage(arrays, rows):
    denominator = int(arrays["eligible"][rows].sum())
    result = {"eligible_cells": denominator}
    for name in ("target", "modeled", "mean_only", "zero_only", "unavailable", "excluded"):
        count = int(arrays[name][rows].sum())
        result[name] = {"cells": count, "fraction_of_eligible": count / denominator if denominator else None,
                        "unavailable_reason": None if denominator else "no eligible cells"}
    return result


def evaluation_evidence(evaluation, descriptive):
    """Separate a final descriptive choice from all frozen outer-fold choices.

    Fold summaries require every requested fold. A failed fold cannot improve
    the aggregate by disappearing. Ratios with a zero baseline are undefined.
    This export does not classify scientific usefulness or calibrate inference.
    """
    for artifact in (evaluation, descriptive):
        validate_evaluation(artifact.arrays, artifact.metadata, artifact.models)
    arrays, metadata = evaluation.arrays, evaluation.metadata
    if (metadata.get("purpose") == "descriptive_fit"
            or descriptive.metadata.get("purpose") != "descriptive_fit"):
        raise ValueError("an outer evaluation and a descriptive fit are required")
    for key in ("identity", "input", "input_metadata"):
        if key not in metadata or metadata[key] != descriptive.metadata.get(key):
            raise ValueError("evidence source or physical identities disagree")
    configurations = [AnalysisConfig(**artifact.metadata["configuration"]).as_dict()
                      for artifact in (evaluation, descriptive)]
    for config in configurations:
        config.pop("guard")
    if configurations[0] != configurations[1]:
        raise ValueError("evidence configurations disagree")
    if not np.array_equal(arrays["window_ids"], descriptive.arrays["window_ids"]):
        raise ValueError("evidence physical time identities disagree")
    losses = {name: [] for name in ("selected", "zero", "mean")}
    folds = []
    for index, report in enumerate(metadata["folds"]):
        rows = arrays["outer_fold"] == index
        record = {"outer_fold": index, "status": report["status"],
            "reason": report.get("reason"), "selected": report.get("selected"),
            "test_windows": int(rows.sum()), "scored_windows": int(arrays["target"][rows].any(axis=1).sum()),
            "coverage": coverage(arrays, rows), "losses": {},
            "rank_ceiling_selected": report.get("inner", {}).get("rule", {}).get("rank_ceiling_selected"),
            "training_filter": report.get("training_filter")}
        for name, key in (("selected", "window_loss"), ("zero", "zero_baseline_loss"), ("mean", "mean_baseline_loss")):
            values = arrays[key][rows]
            scored = arrays["target"][rows].any(axis=1)
            available = report["status"] == "evaluated" and scored.any() and np.isfinite(values[scored]).all()
            value = float(values[scored].mean()) if available else None
            losses[name].append(np.nan if value is None else value)
            record["losses"][name] = {"mean_window_loss": value,
                "unavailable_reason": None if available else "fold lacks verified scored predictions"}
        folds.append(record)
    count = len(folds)
    aggregate = {name: fold_summary(values, count) for name, values in losses.items()}
    comparisons = {}
    for baseline in ("zero", "mean"):
        difference = np.asarray(losses[baseline]) - losses["selected"]
        reference = aggregate[baseline]["mean"]
        selected = aggregate["selected"]["mean"]
        ratio = selected / reference if reference is not None and reference > 0 and selected is not None else None
        comparisons[baseline] = {"baseline_minus_selected": fold_summary(difference, count),
            "selected_to_baseline_loss_ratio": ratio,
            "ratio_unavailable_reason": None if ratio is not None else "baseline is zero or fold evidence is incomplete",
            "folds_with_lower_selected_loss": int(np.count_nonzero(difference > 0)),
            "available_paired_folds": int(np.isfinite(difference).sum())}
    final = descriptive.metadata
    return {"schema_version": 1, "identity": metadata["identity"], "configuration": metadata["configuration"],
        "evaluation_complete": metadata["complete"], "selection_mode": metadata.get("selection_mode", "nested"),
        "descriptive_fit": {"complete": final["complete"], "selected": final.get("selected"),
            "training_loss": final.get("training_loss"), "performance_estimator": False,
            "rank_ceiling_selected": final.get("rank_ceiling_selected"), "reason": final.get("reason")},
        "coverage": coverage(arrays, np.ones(len(arrays["window_ids"]), bool)),
        "folds": folds, "predictive_loss": aggregate, "baseline_comparisons": comparisons,
        "loss_units": f"({metadata['identity']['power_units']})^2",
        "aggregation": "equal physical-window means within each fold, then equal fold means",
        "source_samples": {key: metadata[key] for key in ("input", "input_metadata")},
        "limitations": ["one correlated LST arc", "no independent-realization calibration",
                        "no signal-preservation or inference-readiness test"]}
