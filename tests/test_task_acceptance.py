import json
import sys

import pytest

from hera_systematics_model import scheduler
from hera_systematics_model.configuration import file_identity
from hera_systematics_model.production import create_run, define_task
from hera_systematics_model.task_acceptance import validate_catalog, verify_catalog
from hera_systematics_model.worker import run_task


@pytest.fixture
def accepted_task(tmp_path, monkeypatch):
    source = tmp_path / 'input.txt'
    source.write_text('original')
    run = create_run(tmp_path / 'runs', 'a' * 40, {}, [source])
    define_task(run, {
        'name': 'check', 'command': [sys.executable, '-c',
            "import json,numpy as np;np.savez('data.npz',x=np.arange(3));"
            "open('verification.json','w').write(json.dumps({'passed':True}))"],
        'environment': {}, 'inputs': [file_identity(source)],
        'outputs': [{'path': 'data.npz', 'kind': 'npz'},
                    {'path': 'verification.json', 'kind': 'file'}],
        'resources': {'cpus': 1, 'memory_mib': 1024, 'hours': 1}, 'projected_bytes': 1000000,
    })
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    run_task(run, 'check', require_slurm=False)
    (run / 'submissions').mkdir()
    (run / 'submissions/check.json').write_text(json.dumps({'job_id': '42', 'returncode': 0}))
    monkeypatch.setattr(scheduler.subprocess, 'check_output',
        lambda *a, **k: '42|COMPLETED|0:0|1|1G|00:01:00|\n42.batch|COMPLETED|0:0|1|1G|00:01:00|\n')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'schema_version': 1, 'tasks': [
        {'run': str(run), 'task': 'check', 'job_id': '42'}]}))
    return run, source, catalog, tmp_path / 'accepted.json'


def test_batch_reopens_outputs_and_preserves_producer(accepted_task):
    run, _, catalog, output = accepted_task
    original = (run / 'products/check/success.json').read_bytes()
    result = verify_catalog(catalog, output)
    assert result['passed'] and result['task_count'] == 1
    assert result['tasks'][0]['fresh_input_and_output_hashes']
    assert (run / 'products/check/success.json').read_bytes() == original
    with pytest.raises(FileExistsError):
        verify_catalog(catalog, output)


@pytest.mark.parametrize('changed', ['input', 'output', 'job', 'step', 'receipt_job', 'verification'])
def test_batch_rejects_changed_data_or_failed_scheduler(accepted_task, monkeypatch, changed):
    run, source, catalog, output = accepted_task
    if changed == 'input':
        source.write_text('modified')
    elif changed == 'output':
        (run / 'products/check/data.npz').write_bytes(b'broken')
    elif changed == 'job':
        (run / 'submissions/check.json').write_text(json.dumps({'job_id': '43', 'returncode': 0}))
    elif changed == 'step':
        monkeypatch.setattr(scheduler.subprocess, 'check_output', lambda *a, **k:
            '42|COMPLETED|0:0|1|1G|00:01:00|\n42.batch|FAILED|1:0|1|1G|00:01:00|\n')
    else:
        path = run / 'products/check/success.json'
        receipt = json.loads(path.read_text())
        if changed == 'receipt_job':
            receipt['job_id'] = '43'
        else:
            verification = run / 'products/check/verification.json'
            verification.write_text('{"passed": false}')
            receipt['products'][1].update(file_identity(verification))
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_catalog(catalog, output)
    assert not output.exists()


@pytest.mark.parametrize('tasks', [[],
    [{'run': '/a', 'task': '../escape', 'job_id': '1'}],
    [{'run': 'relative', 'task': 'a', 'job_id': '1'}],
    [{'run': '/a', 'task': 'a', 'job_id': '1'}] * 2,
    [{'run': '/a', 'task': 'a', 'job_id': '1'}, {'run': '/b', 'task': 'b', 'job_id': '1'}],
])
def test_catalog_refuses_ambiguous_or_escaped_identities(tasks):
    with pytest.raises(ValueError):
        validate_catalog({'schema_version': 1, 'tasks': tasks})
