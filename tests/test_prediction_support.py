import numpy as np

from hera_systematics_model.evaluation import evaluate_nested
from hera_systematics_model.prediction import predict_partitioned
from hera_systematics_model.splits import feature_partitions
from test_evaluation import series


def test_row_without_valid_targets_needs_no_coefficient_inference(tmp_path):
    arrays = list(series())
    arrays[3][0] = False
    candidates = [{"method": "mean", "representation": "linear", "rank": 0},
                  {"method": "complete", "representation": "linear", "rank": 2}]
    result = evaluate_nested(arrays, np.arange(100), (1, 30), candidates, guard=3)
    assert result.metadata["complete"]
    assert result.metadata["folds"][0]["selected"]["rank"] == 2
    assert not result.arrays["target"][0].any()
    assert np.isnan(result.arrays["window_loss"][0])
    for entry in result.metadata["folds"][0]["inference"]:
        prefix = entry["array_prefix"]
        assert not result.arrays[prefix + "_coefficients_inferred"][0]
        np.testing.assert_equal(result.arrays[prefix + "_scores"][0], 0)
        assert not result.arrays[prefix + "_predictor_support"][0].any()
    result.save(tmp_path / "evaluation.npz")


def test_mean_only_target_band_does_not_infer_unused_coefficients():
    arrays = list(series())
    arrays[3][0, -6:] = False
    result = predict_partitioned(arrays, np.arange(100), np.arange(75), np.arange(75, 100),
        feature_partitions((1, 30)), {"method": "complete", "representation": "linear", "rank": 2}, keep_models=True)
    assert result.target.all() and result.mean_only[:, -6:].all()
    np.testing.assert_allclose(result.prediction[:, -6:], np.broadcast_to(arrays[0][1:75, -6:].mean(axis=0), (25, 6)))
    assert not result.inference[-1]["coefficients_inferred"].any()
    np.testing.assert_equal(result.inference[-1]["scores"], 0)
    assert result.inference[0]["coefficients_inferred"].all()


def test_empty_geometric_target_bands_do_not_create_failed_kernel_inference():
    arrays = list(series())
    arrays[3][:, :12] = False
    result = predict_partitioned(arrays, np.arange(100), np.arange(75), np.arange(75, 100),
        feature_partitions((1, 30)), {"method": "kernel", "representation": "linear", "rank": 2}, keep_models=True)
    assert len(result.models) == 3
    assert [entry["partition_index"] for entry in result.inference] == [2, 3, 4]
    assert result.target[:, 12:].all() and not result.target[:, :12].any()
