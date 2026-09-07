from dataclasses import replace

import numpy as np
import pytest

from test_samples import paired

from hera_systematics_model.diagnostics import regional_losses


def test_regional_losses_keep_geometry_and_unavailable_denominators():
    truth = np.ones((4, 6))
    target = np.ones(truth.shape, bool)
    target[:, :2] = False
    prediction = np.zeros(truth.shape)
    prediction[~target] = np.nan
    regions = {"full": np.ones((2, 3), bool), "short": np.array([[True] * 3, [False] * 3]),
               "empty": np.zeros((2, 3), bool)}
    result = regional_losses(prediction, truth, np.ones_like(truth), target, np.ones_like(target), regions)
    assert result["full"]["scored_cells"] == 16
    assert result["full"]["eligible_cells"] == 24
    assert result["short"]["scored_cells"] == 4
    assert result["short"]["eligible_cells"] == 12
    assert result["full"]["window_loss"]["mean"] == 1
    assert result["empty"]["window_loss"]["mean"] is None
    assert result["empty"]["window_loss"]["reason"]



@pytest.mark.parametrize("group,delay", [(1, None), (None, 2)])
def test_sample_views_preserve_physical_contributors_and_weights(paired, group, delay):
    from hera_systematics_model.views import sample_view, analysis_view

    viewed = sample_view(paired, group, delay)
    expected, shape, identity = analysis_view(paired, group, delay)
    actual, _, actual_identity = analysis_view(viewed)
    assert viewed.corrupted.shape[1:] == shape
    assert identity == actual_identity
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_equal(viewed.contributor_counts, 1)
    if group is not None:
        assert viewed.baseline_ids.tolist() == ["0:2"]
        assert viewed.baseline_group.tolist() == [0]
    else:
        assert viewed.baseline_ids.tolist() == paired.baseline_ids.tolist()
    viewed.corrupted[:] = 99
    np.testing.assert_equal(paired.corrupted, 0)


@pytest.mark.parametrize("group,delay", [(1, None), (None, 2)])
def test_localized_diagnostics_keep_coordinates_and_region_denominators(paired, group, delay):
    from hera_systematics_model.configuration import AnalysisConfig
    from hera_systematics_model.diagnostics import residual_diagnostics
    from hera_systematics_model.evaluation import evaluate_nested
    from hera_systematics_model.fitting import select_final_fit
    from hera_systematics_model.views import analysis_view

    ids = np.arange(120)
    shape = (120, 8, 15)
    sample = replace(paired, corrupted=np.zeros(shape), ideal=np.ones(shape), pn=np.ones(shape),
        valid=np.ones(shape, bool), weights=np.full((*shape, 2), .5),
        group_ids=np.array([str(i) for i in range(8)]), baseline_ids=np.array([f"0:{i+1}" for i in range(8)]),
        baseline_group=np.arange(8), baseline_length_m=np.arange(1, 9) * 10., kperp=np.arange(1, 9) * .1,
        delay_s=np.arange(1, 16) * 1e-7, kparallel=np.arange(1, 16) * .05,
        window_ids=ids, time_jd=2450000. + (ids + .5) * 270. / 86400., lst_rad=(6.1 + ids * .02) % (2 * np.pi))
    config = AnalysisConfig(group=group, delay=delay, max_rank=0, include_kernel=False)
    arrays, feature_shape, identity = analysis_view(sample, group, delay)
    axis = "group" if delay is not None else "delay"
    fit = select_final_fit(arrays, ids, feature_shape, config.candidates(), axis=axis)
    evaluation = evaluate_nested(arrays, ids, feature_shape, config.candidates(), axis=axis)
    for artifact in (fit, evaluation):
        assert artifact.metadata["complete"]
        artifact.metadata.update(identity=identity, configuration=config.as_dict())
    output, metadata = residual_diagnostics(sample, fit, evaluation, n_surrogates=9)
    assert output["mean_residual"].shape == feature_shape
    assert output["contributor_counts"].shape == (120, *feature_shape, 2)
    assert metadata["regions"]["full"]["eligible_cells"] == np.prod((120, *feature_shape))
    assert len(metadata["baseline_inventory"]) == feature_shape[0]
    assert metadata["identity"] == identity
    evaluation.metadata["configuration"]["region"] = "horizon"
    with pytest.raises(ValueError, match="configurations disagree"):
        residual_diagnostics(sample, fit, evaluation)
