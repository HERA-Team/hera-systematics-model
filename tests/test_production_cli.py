import json
from pathlib import Path

import pytest

from hera_systematics_model.cli import main
from hera_systematics_model.production import read_run
from test_production import task


def test_production_cli_captures_configuration_and_refuses_replacement(tmp_path, capsys):
    source, config = tmp_path / "source.dat", tmp_path / "configuration.json"
    source.write_bytes(b"measured input")
    config.write_text('{"spws": [0, 1]}')
    arguments = ["production", "init", "--root", str(tmp_path / "runs"), "--code-commit", "a" * 40,
                 "--config", str(config), "--input", str(source), "--projected-bytes", "1000"]
    assert main(arguments) == 0
    run = Path(json.loads(capsys.readouterr().out)["run"])
    assert read_run(run)["identity"]["configuration"] == {"spws": [0, 1]}
    assert next((run / "snapshots").iterdir()).read_text() == config.read_text()
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    definition = tmp_path / "task.json"
    definition.write_text(json.dumps(task()))
    assert main(["production", "define", "--run", str(run), "--definition", str(definition)]) == 0
    with pytest.raises(SystemExit) as error:
        main(["production", "define", "--run", str(run), "--definition", str(definition)])
    assert error.value.code == 2
