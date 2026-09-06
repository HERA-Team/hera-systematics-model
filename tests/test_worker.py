import json
import sys

import pytest

from hera_systematics_model.production import create_run, define_task
from hera_systematics_model.worker import run_task, verified_receipt
from test_production import task


def setup_task(tmp_path, code):
    run = create_run(tmp_path / "runs", "a" * 40, {}, [])
    specification = task()
    specification["command"] = [sys.executable, "-c", code]
    define_task(run, specification)
    return run


def test_success_requires_output_and_resume_does_not_rewrite_it(tmp_path):
    run = setup_task(tmp_path, "import numpy as np; np.savez('spectrum.npz', power=np.arange(3))")
    first = run_task(run, "baseline-1", require_slurm=False)
    product = run / "products/baseline-1/spectrum.npz"
    stamp = product.stat().st_mtime_ns
    second = run_task(run, "baseline-1", require_slurm=False)
    assert first == second
    assert product.stat().st_mtime_ns == stamp
    product.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verified_receipt(run, "baseline-1")


@pytest.mark.parametrize("code", ["raise SystemExit(7)", "pass", "open('spectrum.npz', 'w').write('broken')"])
def test_command_failure_absence_and_corruption_cannot_succeed(tmp_path, code):
    run = setup_task(tmp_path, code)
    with pytest.raises(ValueError):
        run_task(run, "baseline-1", require_slurm=False)
    assert not (run / "products/baseline-1/success.json").exists()
    assert (run / "products/baseline-1/failure.json").exists()


def test_stale_success_marker_cannot_be_reused_with_changed_configuration(tmp_path):
    run = setup_task(tmp_path, "import numpy as np; np.savez('spectrum.npz', power=np.arange(3))")
    run_task(run, "baseline-1", require_slurm=False)
    path = run / "tasks/baseline-1.json"
    spec = json.loads(path.read_text())
    spec["environment"] = {"OMP_NUM_THREADS": "2"}
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="stale"):
        verified_receipt(run, "baseline-1")
