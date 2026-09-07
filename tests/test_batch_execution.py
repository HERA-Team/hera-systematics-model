import json
from pathlib import Path
import sys

import pytest

from hera_systematics_model.batch_execution import run_bounded_commands, validate_batch


def task(name, script):
    return {"name": name, "command": [sys.executable, "-c", script], "environment": {"OMP_NUM_THREADS": "1"}}


def test_concurrent_children_keep_separate_working_directories_and_state(tmp_path):
    script = ("from pathlib import Path\nimport os,time\n"
        "Path('ready').write_text(str(os.getpid()))\n"
        "for _ in range(200):\n"
        " if len(list(Path('..').glob('*/ready')))==2:break\n"
        " time.sleep(.01)\n"
        "else:raise RuntimeError('concurrent peer never started')\n"
        "print(Path.cwd().name)\n")
    result = run_bounded_commands([task("a", script), task("b", script)], tmp_path / "batch", 2, 4, 32768)
    assert result["all_commands_exited_zero"] and not result["products_verified"]
    assert [r["name"] for r in result["records"]] == ["a", "b"]
    for name in ("a", "b"):
        assert (tmp_path / "batch" / name / "stdout.log").read_text().strip() == name
    assert (tmp_path / "batch/a/ready").read_text() != (tmp_path / "batch/b/ready").read_text()


def test_failure_stops_new_launches_and_preserves_nonzero_status(tmp_path):
    result = run_bounded_commands([task("a", "print('retained');raise SystemExit(7)"),
        task("b", "raise AssertionError('must not run')")], tmp_path / "batch", 1, 2, 16384)
    assert not result["all_commands_exited_zero"]
    assert result["records"][0]["exit_code"] == 7
    assert result["records"][1]["status"] == "not_started"
    assert not (tmp_path / "batch/b").exists()
    assert json.loads((tmp_path / "batch/batch-result.json").read_text()) == result
    with pytest.raises(FileExistsError):
        run_bounded_commands([task("a", "pass")], tmp_path / "batch", 1, 2, 16384)


@pytest.mark.parametrize("workers,cpus,memory", [(4, 7, 65536), (4, 8, 65535), (5, 16, 131072)])
def test_reservations_reject_oversubscription(workers, cpus, memory):
    with pytest.raises(ValueError, match="allocation|allocated"):
        validate_batch([task("a", "pass")], workers, cpus, memory)


def test_duplicate_physical_task_names_are_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        run_bounded_commands([task("a", "pass"), task("a", "pass")], tmp_path / "batch", 1, 2, 16384)
    assert not (tmp_path / "batch").exists()
