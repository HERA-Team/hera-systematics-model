from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from hera_systematics_model.spectral_labels import canonicalize_time_average_labels


def spectrum(rows=6):
    return SimpleNamespace(Nbltpairs=1, Nspws=14, Npols=1,
        blpair_array=np.array([100101100101]), labels=["dset0", "dset1"],
        label_1_array=np.zeros((14, rows, 1), int), label_2_array=np.ones((14, rows, 1), int),
        history="captured history", data_array=np.array([0., -1., np.nan]),
        noise_array=np.array([1., 2., np.inf]), time_array=np.array([2459861.4]))


def test_repeated_interleave_labels_collapse_without_changing_measurements():
    value = spectrum()
    before = deepcopy(value)
    report = canonicalize_time_average_labels(value)
    assert report["passed"] and report["changed"]
    assert not report["numerical_payload_modified"]
    for name in ("label_1_array", "label_2_array"):
        assert getattr(value, name).shape == (14, 1, 1)
        np.testing.assert_array_equal(report["label_arrays"][name]["input_labels"], getattr(before, name))
    for name in ("data_array", "noise_array", "time_array", "blpair_array"):
        np.testing.assert_array_equal(getattr(value, name), getattr(before, name))
    assert value.history.startswith(before.history)


def test_already_canonical_labels_leave_history_unchanged():
    value = spectrum(rows=1)
    assert not canonicalize_time_average_labels(value)["changed"]
    assert value.history == "captured history"


def test_distinct_labels_fail_without_partial_mutation():
    value = spectrum()
    value.label_2_array[3, 2, 0] = 0
    before = deepcopy(value)
    with pytest.raises(ValueError, match="distinct labels"):
        canonicalize_time_average_labels(value)
    np.testing.assert_array_equal(value.label_1_array, before.label_1_array)
    np.testing.assert_array_equal(value.label_2_array, before.label_2_array)
    assert value.history == before.history


@pytest.mark.parametrize("field,value", [
    ("Nbltpairs", 2), ("blpair_array", np.array([1, 2])),
    ("label_1_array", np.zeros((14, 0, 1), int)),
    ("label_1_array", np.zeros((13, 6, 1), int)),
    ("label_1_array", np.zeros((14, 6, 2), int)),
    ("label_1_array", np.zeros((14, 6, 1), float)),
    ("label_1_array", np.full((14, 6, 1), -1, int)),
    ("label_1_array", np.full((14, 6, 1), 2, int)),
])
def test_inconsistent_label_metadata_is_rejected(field, value):
    obj = spectrum()
    setattr(obj, field, value)
    with pytest.raises(ValueError):
        canonicalize_time_average_labels(obj)
