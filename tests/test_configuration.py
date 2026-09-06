import json

import pytest

from hera_systematics_model.configuration import AnalysisConfig, capture_runtime, digest_json, file_identity


def test_configuration_digest_is_order_independent_and_complete(tmp_path):
    config = AnalysisConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.as_dict()))
    assert AnalysisConfig.load(path) == config
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
    assert digest_json(config.as_dict()) != digest_json(AnalysisConfig(guard=8).as_dict())
    assert file_identity(path)["bytes"] > 0


@pytest.mark.parametrize("values", [{"guard": 4}, {"max_rank": 21}, {"methods": ["masked", "masked"]},
    {"representations": ["unknown"]}, {"group": 1, "delay": 2}, {"include_kernel": "false"}])
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        AnalysisConfig(**values)


def test_runtime_captures_executed_sources_and_versions():
    result = capture_runtime()
    assert "models.py" in result["package_sources"]
    assert any(name == "numpy" for name, _ in result["distributions"])
    assert result["source_digest"] == digest_json(result["package_sources"])
