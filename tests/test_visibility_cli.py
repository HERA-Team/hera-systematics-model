import json

import pytest

from hera_systematics_model.cli import main
from hera_systematics_model.visibility_cli import read_paths
from test_visibility_inventory import create


def test_inventory_command_uses_explicit_files_and_exclusive_output(tmp_path, capsys):
    data, files, output = [tmp_path / name for name in ("data.uvh5", "files.json", "inventory.json")]
    create(data)
    files.write_text(json.dumps([str(data)]))
    command = ["inventory", "--files", str(files), "--output", str(output)]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["metadata_only"]
    assert json.loads(output.read_text())["files"][0]["cross_baselines"] == 2
    with pytest.raises(SystemExit) as error:
        main(command)
    assert error.value.code == 2


def test_visibility_file_lists_reject_implicit_or_duplicate_inputs(tmp_path):
    path = tmp_path / "files.json"
    for value in [[], ["a", "a"], {"a": "b"}, [1]]:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="unique file paths"):
            read_paths(path)
