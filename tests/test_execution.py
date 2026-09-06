import os
import subprocess
import sys

import pytest

from hera_systematics_model.execution import execute_checked, storage_monitor


def test_storage_exhaustion_stops_writer_and_retains_output(tmp_path):
    evidence = {}
    check = storage_monitor(tmp_path, tmp_path, 100, evidence)
    command = [sys.executable, "-c", "import time; open('retained', 'wb').write(b'x' * 200); time.sleep(30)"]
    with open(os.devnull, "wb") as stream:
        with pytest.raises(ValueError, match="reservation"):
            execute_checked(command, tmp_path, os.environ, stream, stream, 5, check, interval=.02)
    assert (tmp_path / "retained").read_bytes() == b"x" * 200
    assert evidence["maximum_sampled_task_bytes"] == 200


def test_timeout_propagates_without_accepting_background_work(tmp_path):
    with open(os.devnull, "wb") as stream:
        with pytest.raises(subprocess.TimeoutExpired):
            execute_checked([sys.executable, "-c", "import time; time.sleep(30)"],
                tmp_path, os.environ, stream, stream, .05, lambda: None, interval=.02)


def test_final_short_lived_output_is_checked(tmp_path):
    check = storage_monitor(tmp_path, tmp_path, 100, {})
    with open(os.devnull, "wb") as stream:
        with pytest.raises(ValueError, match="reservation"):
            execute_checked([sys.executable, "-c", "open('retained', 'wb').write(b'x' * 200)"],
                tmp_path, os.environ, stream, stream, 5, check)
