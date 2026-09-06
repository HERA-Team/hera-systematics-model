"""The base installation must not require the radio I/O dependencies."""

import subprocess
import sys

from hera_systematics_model import __version__
from hera_systematics_model.cli import main


def test_import_without_radio_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import hera_systematics_model; "
         "assert 'hera_pspec' not in sys.modules; "
         "assert 'pyuvdata' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_installed_command():
    result = subprocess.run(
        [str(__import__('pathlib').Path(sys.executable).with_name('hera-systematics')),
         "--version"], capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == __version__


def test_help(capsys):
    assert main([]) == 0
    assert "Cylindrical" in capsys.readouterr().out
