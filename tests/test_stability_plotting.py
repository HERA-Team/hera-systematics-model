import json

import numpy as np
import pytest

from hera_systematics_model.stability import bootstrap_stability
from hera_systematics_model.stability_plotting import plot_stability
from test_models import low_rank_data


def measured(rank=2):
    arrays, metadata = bootstrap_stability(low_rank_data(), np.arange(40),
        {"method": "complete", "representation": "linear", "rank": rank}, n_replicates=3)
    metadata["identity"] = {"spw": 0}
    return arrays, metadata


def test_bootstrap_figures_preserve_failed_draw_denominator(tmp_path):
    arrays, metadata = measured()
    metadata["records"][1] = {"replicate": 1, "status": "unavailable", "reason": "synthetic unsupported draw"}
    products = plot_stability(arrays, metadata, tmp_path / "figures")
    assert len(products) == 3 and all(product["bytes"] > 1000 for product in products)
    report = json.loads((tmp_path / "figures/figures.json").read_text())["bootstrap_measurements"]
    assert report["requested_replicates"] == 3 and report["evaluated_replicates"] == 2
    assert report["feature_support_counts"] == [2] * 30
    assert len(report["failed_draws"]) == 1
    assert all(component["sign_aligned_cosine"]["n"] == 2 for component in report["components"])


def test_zero_rank_bootstrap_does_not_fabricate_components(tmp_path):
    arrays, metadata = measured(rank=0)
    plot_stability(arrays, metadata, tmp_path / "figures")
    report = json.loads((tmp_path / "figures/figures.json").read_text())["bootstrap_measurements"]
    assert report["components"] == []


def test_reordered_bootstrap_records_fail_before_writing(tmp_path):
    arrays, metadata = measured()
    metadata["records"].reverse()
    with pytest.raises(ValueError, match="reordered"):
        plot_stability(arrays, metadata, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
