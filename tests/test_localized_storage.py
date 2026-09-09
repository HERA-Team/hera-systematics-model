import copy

import pytest

from hera_systematics_model.localized_storage import project_localized_storage


def inputs():
    tasks = [{"slice": f"slice-{axis}-{index}", "spw": 0, "axis": axis,
              "method": "complete", "features": 8,
              "configuration_digest": str(index) * 64}
             for axis in ("group", "delay") for index in range(3)]
    measurements = [{"slice": task["slice"],
                     "configuration_digest": task["configuration_digest"],
                     "retained_bytes": 100 if task["axis"] == "group" else 200}
                    for task in tasks if task["configuration_digest"] == "0" * 64]
    return tasks, measurements


def test_remaining_only_and_single_contingency():
    tasks, measurements = inputs()
    result = project_localized_storage(tasks, measurements, 1000, 100)
    assert result["projected_additional_bytes"] == 700
    assert result["projected_peak_bytes"] == 1840
    assert result["within_cap"]
    assert result == project_localized_storage(tasks[::-1], measurements[::-1], 1000, 100)


def test_larger_second_measurement_sets_family_estimate():
    tasks, measurements = inputs()
    measurements.append({"slice": tasks[1]["slice"],
                         "configuration_digest": tasks[1]["configuration_digest"],
                         "retained_bytes": 400})
    result = project_localized_storage(tasks, measurements, 1000)
    assert result["projected_additional_bytes"] == 800


@pytest.mark.parametrize("case", ["missing_family", "duplicate_measurement", "unknown",
                                 "configuration", "zero", "dimensions", "duplicate_task"])
def test_incomplete_or_mismatched_evidence_fails(case):
    tasks, measurements = inputs()
    if case == "missing_family":
        measurements.pop()
    elif case == "duplicate_measurement":
        measurements.append(copy.deepcopy(measurements[0]))
    elif case == "unknown":
        measurements[0]["slice"] = "absent"
    elif case == "configuration":
        measurements[0]["configuration_digest"] = "f" * 64
    elif case == "zero":
        measurements[0]["retained_bytes"] = 0
    elif case == "dimensions":
        tasks[1]["features"] = 9
    else:
        tasks.append(copy.deepcopy(tasks[0]))
    with pytest.raises(ValueError):
        project_localized_storage(tasks, measurements, 1000)


def test_cap_and_impossible_retained_footprints():
    tasks, measurements = inputs()
    assert not project_localized_storage(tasks, measurements, 1500000000000)["within_cap"]
    with pytest.raises(ValueError, match="exceed"):
        project_localized_storage(tasks, measurements, 299)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            project_localized_storage(tasks, measurements, 1000, invalid)
