import numpy as np
import pytest
import sys
from types import SimpleNamespace

from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.ideal_io import baseline_key, choose_source_files, redundant_baseline_map, source_polarizations, verify_mapping_geometry


def test_source_selection_is_deterministic_across_wrap_and_input_order():
    knots = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    inventory = [{"path": str(index), "lst_rad": float(value)} for index, value in enumerate(knots)]
    target = np.array([6.23, .02, .10])
    expected = choose_source_files(inventory, target, buffer_rad=.05)
    assert choose_source_files(inventory[::-1], target, buffer_rad=.05) == expected
    assert expected == ["98", "99", "0", "1", "2", "3"]
    assert baseline_key((6, 34)) == "6_34"


def test_undersupported_source_selection_is_explicit():
    inventory = [{"path": str(i), "lst_rad": float(i)} for i in range(3)]
    with pytest.raises(ValueError):
        choose_source_files(inventory, [.5, .6], .1)


def test_mapping_records_absent_baselines_and_integer_identities(monkeypatch):
    class Groups:
        @classmethod
        def from_antpos(cls, antpos):
            assert set(antpos) == {0, 1, 2}
            return cls()
        def get_reds_in_bl_set(self, pair, **kwargs):
            return {(np.int64(0), np.int64(1))} if pair == (0, 1) else set()
    monkeypatch.setitem(sys.modules, "hera_cal.red_groups", SimpleNamespace(RedundantGroups=Groups))
    reference = SimpleNamespace(telescope=SimpleNamespace(get_enu_antpos=lambda: np.eye(3), antenna_numbers=np.arange(3)),
                                get_antpairs=lambda: [(np.int64(0), np.int64(1)), (1, 2)])
    source = SimpleNamespace(get_antpairs=lambda: [(0, 1)], telescope=reference.telescope)
    result = redundant_baseline_map(reference, source)
    assert result["0_1"]["source_pair"] == [0, 1]
    assert result["0_1"]["stored_pair"] == [0, 1]
    assert not result["0_1"]["conjugate"]
    assert result["1_2"]["exclusion"] == "source_baseline_absent"
    canonical_json(result)
    source.get_antpairs = lambda: [(1, 0)]
    reversed_map = redundant_baseline_map(reference, source)
    assert reversed_map["0_1"]["source_pair"] == [0, 1]
    assert reversed_map["0_1"]["stored_pair"] == [1, 0]
    assert reversed_map["0_1"]["conjugate"]


def test_reversed_baselines_exchange_cross_polarization_indices():
    source = [-5, -7, -8, -6]
    assert source_polarizations(source, [-5, -6, -7, -8]) == [0, 3, 1, 2]
    assert source_polarizations(source, [-5, -6, -7, -8], True) == [0, 3, 2, 1]
    assert source_polarizations([-1, -2, -3, -4], [-1, -2, -3, -4], True) == [0, 1, 3, 2]
    with pytest.raises(ValueError, match="polarization"):
        source_polarizations([-5, -7], [-7], True)


def test_actual_source_vector_identity_is_checked_independently_of_labels():
    def model(positions):
        return SimpleNamespace(telescope=SimpleNamespace(antenna_numbers=[0, 1], get_enu_antpos=lambda: positions))
    reference = model(np.array([[0., 0., 0.], [10., 0., 0.]]))
    source = model(np.array([[100., 0., 0.], [110., 0., 0.]]))
    mapping = {"0_1": {"reference_pair": [0, 1], "source_pair": [0, 1], "stored_pair": [0, 1], "conjugate": False}}
    checked = verify_mapping_geometry(reference, source, mapping)
    assert checked["0_1"]["vector_difference_m"] == 0
    assert checked["0_1"]["source_vector_enu_m"] == [10., 0., 0.]
    with pytest.raises(ValueError, match="vectors differ"):
        verify_mapping_geometry(reference, model(np.array([[0., 0., 0.], [11., 0., 0.]])), mapping)
    with pytest.raises(ValueError, match="absent"):
        verify_mapping_geometry(reference, source, {"0_2": {**mapping["0_1"], "reference_pair": [0, 2]}})
    with pytest.raises(ValueError, match="conjugation"):
        verify_mapping_geometry(reference, source, {"0_1": {**mapping["0_1"], "stored_pair": [1, 0]}})
