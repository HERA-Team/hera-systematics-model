import json

import pytest

from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.cli import main
from hera_systematics_model.conclusions import (
    predictive_conclusion,
    predictive_conclusion_table,
)


def summary(selected=(5, 5, 5, 5), zero=(10, 10, 10, 10), mean=(9, 9, 9, 9)):
    folds = []
    for index in range(4):
        folds.append({
            "outer_fold": index,
            "status": "evaluated",
            "scored_windows": 20,
            "coverage": {
                "eligible_cells": 100,
                "target": {"cells": 25},
            },
            "losses": {
                "selected": {"mean_window_loss": selected[index]},
                "zero": {"mean_window_loss": zero[index]},
                "mean": {"mean_window_loss": mean[index]},
            },
        })
    return {
        "schema_version": 1,
        "identity": {"spw": 6, "power_units": "mK2 Mpc3 / h3"},
        "folds": folds,
        "descriptive_fit": {
            "complete": True,
            "selected": {"representation": "linear", "method": "complete", "rank": 2},
            "rank_ceiling_selected": False,
        },
        "coverage": {"eligible_cells": 100, "target": {"cells": 100},
                     "modeled": {"cells": 80}, "mean_only": {"cells": 20},
                     "unavailable": {"cells": 0}},
        "predictive_loss": {"selected": {"mean": 5}, "zero": {"mean": 10},
                            "mean": {"mean": 9}},
        "limitations": ["one correlated LST arc"],
    }


def test_useful_structure_requires_better_baseline_and_one_fold_standard_error():
    report = predictive_conclusion(summary())
    assert report["conclusion_key"] == "useful"
    assert report["better_baseline"] == "mean"
    assert report["better_baseline_minus_selected"]["mean"] == 4
    assert all(report["criteria"].values())
    canonical_json(report)


@pytest.mark.parametrize("selected", [
    (8, 10, 8, 10),
    (5, 11, 5, 11),
])
def test_complete_evidence_without_predeclared_margin_reports_no_advantage(selected):
    report = predictive_conclusion(summary(selected=selected))
    assert report["conclusion_key"] == "no_advantage"
    assert report["criteria"]["four_complete_outer_folds"]
    assert not all(report["criteria"].values())
    canonical_json(report)


def test_missing_common_plane_fold_reports_insufficient_evidence():
    evidence = summary()
    evidence["folds"][2]["coverage"]["target"]["cells"] = 0
    report = predictive_conclusion(evidence)
    assert report["conclusion_key"] == "insufficient"
    assert report["complete_common_plane_folds"] == [0, 1, 3]
    assert report["better_baseline_minus_selected"]["mean"] is None
    assert report["unavailable_folds"] == [{
        "outer_fold": 2,
        "reason": "fold lacks complete common-plane selected and baseline scores",
    }]
    canonical_json(report)


def test_better_baseline_is_selected_by_complete_fold_mean_without_oracle_switching():
    evidence = summary(
        selected=(5, 5, 5, 5),
        zero=(1, 10, 10, 10),
        mean=(9, 9, 9, 9),
    )
    report = predictive_conclusion(evidence)
    assert report["better_baseline"] == "zero"
    assert report["better_baseline_minus_selected"]["mean"] == 2.75
    assert report["conclusion_key"] == "useful"


def test_conclusion_rejects_changed_fold_identity_or_count():
    evidence = summary()
    evidence["folds"][1]["outer_fold"] = 3
    with pytest.raises(ValueError, match="identities"):
        predictive_conclusion(evidence)
    with pytest.raises(ValueError, match="exactly four"):
        predictive_conclusion({**summary(), "folds": summary()["folds"][:3]})


def test_conclusion_command_binds_saved_summary(tmp_path):
    source = tmp_path / "summary.json"
    output = tmp_path / "conclusion.json"
    source.write_text(json.dumps(summary()))
    assert main(["conclude", "--summary", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["conclusion_key"] == "useful"
    assert report["input"]["path"] == str(source)
    assert report["input"]["bytes"] == source.stat().st_size


def test_conclusion_table_preserves_each_spectral_window():
    summaries = []
    for spw in reversed(range(14)):
        evidence = summary()
        evidence["identity"] = {**evidence["identity"], "spw": spw}
        evidence["descriptive_fit"]["selected"]["rank"] = spw % 3
        summaries.append(evidence)
    report = predictive_conclusion_table(summaries)
    assert report["spectral_windows"] == list(range(14))
    assert [row["spw"] for row in report["rows"]] == list(range(14))
    assert [row["selected"]["rank"] for row in report["rows"]] == [i % 3 for i in range(14)]
    assert not report["aggregation_across_spectral_windows"]
    canonical_json(report)


def test_conclusion_table_rejects_duplicate_or_incomplete_windows():
    summaries = []
    for spw in range(14):
        evidence = summary()
        evidence["identity"] = {**evidence["identity"], "spw": spw}
        summaries.append(evidence)
    summaries[-1]["identity"]["spw"] = 12
    with pytest.raises(ValueError, match="unique"):
        predictive_conclusion_table(summaries)
    with pytest.raises(ValueError, match="exactly 14"):
        predictive_conclusion_table(summaries[:-1])


def test_conclusion_table_command_binds_all_summary_files(tmp_path):
    paths = []
    for spw in range(14):
        evidence = summary()
        evidence["identity"] = {**evidence["identity"], "spw": spw}
        path = tmp_path / f"spw-{spw:02d}.json"
        path.write_text(json.dumps(evidence))
        paths.append(path)
    output = tmp_path / "primary-table.json"
    arguments = ["conclude-table", "--summaries", *map(str, paths),
                 "--output", str(output)]
    assert main(arguments) == 0
    report = json.loads(output.read_text())
    assert len(report["inputs"]) == 14
    assert [row["spw"] for row in report["rows"]] == list(range(14))
    assert [item["path"] for item in report["inputs"]] == list(map(str, paths))
