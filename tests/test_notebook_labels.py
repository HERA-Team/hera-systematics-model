from copy import deepcopy

import nbformat
import pytest

from hera_systematics_model.notebook_labels import instrument_labels


def document(source):
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])


def test_label_hook_runs_after_average_before_consuming_result():
    original = document("uvp_avg_all = average(\n    time_avg=True)\nconsume(uvp_avg_all)")
    original.cells.append(nbformat.v4.new_code_cell("%matplotlib inline"))
    before = deepcopy(original)
    result, metadata = instrument_labels(original)
    source = result.cells[0].source
    compile(source, "instrumented", "exec")
    assert source.index("time_avg=True)") < source.index("canonicalize_time_average_labels(uvp_avg_all)")
    assert source.index("write_json_exclusive(Path") < source.index("consume(uvp_avg_all)")
    assert metadata["report"] == "label-metadata.json"
    assert original == before


@pytest.mark.parametrize("source", ["unrelated = 1", "uvp_avg_all = first\nuvp_avg_all = second",
                                   "def nested():\n    uvp_avg_all = first"])
def test_missing_or_ambiguous_final_average_fails(source):
    with pytest.raises(ValueError, match="exactly one"):
        instrument_labels(document(source))
