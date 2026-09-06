import copy

import numpy as np
import pytest

from hera_systematics_model.evaluation import evaluate_nested
from hera_systematics_model.evaluation_state import validate_evaluation
from test_evaluation import series


@pytest.fixture
def evaluated():
    return evaluate_nested(series(), np.arange(100), (1, 30),
        [{"method": "complete", "rank": 2, "representation": "linear"}], guard=3)


def test_semantic_validation_rejects_overlapping_coverage(evaluated):
    evaluated.arrays["mean_only"][0, 0] = True
    with pytest.raises(ValueError, match="coverage"):
        validate_evaluation(evaluated.arrays, evaluated.metadata, evaluated.models)


def test_semantic_validation_rejects_target_cells_used_as_predictors(evaluated):
    prefix = evaluated.metadata["folds"][0]["inference"][0]["array_prefix"]
    evaluated.arrays[f"{prefix}_predictor_support"][:, 0] = True
    with pytest.raises(ValueError, match="inference"):
        validate_evaluation(evaluated.arrays, evaluated.metadata, evaluated.models)


def test_semantic_validation_rejects_changed_training_window_id(evaluated):
    evaluated.models[0][0].training_ids = evaluated.models[0][0].training_ids.copy()
    evaluated.models[0][0].training_ids[0] = 0
    with pytest.raises(ValueError, match="training identities"):
        validate_evaluation(evaluated.arrays, evaluated.metadata, evaluated.models)


def test_semantic_validation_rejects_dropped_time_partition(evaluated):
    evaluated.metadata["folds"][0]["guard_window_ids"] = []
    with pytest.raises(ValueError, match="partitions"):
        validate_evaluation(evaluated.arrays, evaluated.metadata, evaluated.models)
