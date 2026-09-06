import numpy as np
import pytest
import sys
from types import SimpleNamespace

from hera_systematics_model.artifacts import canonical_json
from hera_systematics_model.ideal_io import baseline_key, choose_source_files, redundant_baseline_map


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
    source = SimpleNamespace(get_antpairs=lambda: [(0, 1)])
    result = redundant_baseline_map(reference, source)
    assert result["0_1"]["source_pair"] == [0, 1]
    assert result["1_2"]["exclusion"] == "source_baseline_absent"
    canonical_json(result)
