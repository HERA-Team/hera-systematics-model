import json

import pytest
from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.cli import main
from hera_systematics_model.conclusions import predictive_conclusion


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
        "limitations": ["one correlated LST arc"],
    }


def test_useful_structure_requires_better_baseline_and_one_fold_standard_error():
    report = predictive_conclusion(summary())
    assert report["conclusion_key"] == "useful"
    assert report["better_baseline_by_fold"] == ["mean"] * 4
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
