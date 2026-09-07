import json

import numpy as np
import pytest

from hera_systematics_model.product_manifest import create_product_manifest
from hera_systematics_model.worker import verify_product


def products(tmp_path):
    np.savez(tmp_path / 'model.npz', components=np.arange(6).reshape(2, 3))
    (tmp_path / 'model.json').write_text('{"rank":2}')
    specifications = [{'path': 'model.npz', 'kind': 'npz'}, {'path': 'model.json', 'kind': 'file'}]
    create_product_manifest(tmp_path, specifications, 'products.json')
    return {'path': 'products.json', 'kind': 'manifest'}


def test_manifest_reopens_models_and_detects_mutation_and_absence(tmp_path):
    specification = products(tmp_path)
    accepted = verify_product(tmp_path, specification)
    assert len(accepted['structure']['products']) == 2
    assert accepted['structure']['products'][0]['structure']['components']['shape'] == [2, 3]
    original = (tmp_path / 'model.json').read_bytes()
    (tmp_path / 'model.json').write_text('{"rank":3}')
    with pytest.raises(ValueError, match='changed'):
        verify_product(tmp_path, specification)
    (tmp_path / 'model.json').write_bytes(original)
    assert verify_product(tmp_path, specification) == accepted
    (tmp_path / 'model.npz').unlink()
    with pytest.raises(ValueError, match='missing'):
        verify_product(tmp_path, specification)


def test_manifest_rejects_object_payload_before_binding(tmp_path):
    np.savez(tmp_path / 'model.npz', unsafe=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match='Object arrays'):
        create_product_manifest(tmp_path, [{'path': 'model.npz', 'kind': 'npz'}], 'products.json')
    assert not (tmp_path / 'products.json').exists()


@pytest.mark.parametrize('change', ['duplicate', 'recursive', 'escape', 'missing_hash', 'version', 'empty'])
def test_manifest_rejects_invalid_inventory(tmp_path, change):
    specification = products(tmp_path)
    path = tmp_path / 'products.json'
    value = json.loads(path.read_text())
    if change == 'duplicate':
        value['products'].append(value['products'][0])
    elif change == 'recursive':
        value['products'][0]['kind'] = 'manifest'
    elif change == 'escape':
        value['products'][0]['path'] = '../model.npz'
    elif change == 'missing_hash':
        del value['products'][0]['sha256']
    elif change == 'version':
        value['schema_version'] = True
    else:
        value['products'] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_product(tmp_path, specification)


def test_manifest_rejects_external_symlink_and_overwrite(tmp_path):
    directory = tmp_path / 'task'
    directory.mkdir()
    (tmp_path / 'external').write_text('external')
    (directory / 'link').symlink_to(tmp_path / 'external')
    with pytest.raises(ValueError, match='escaped'):
        create_product_manifest(directory, [{'path': 'link', 'kind': 'file'}], 'products.json')
    products(directory)
    with pytest.raises(FileExistsError):
        create_product_manifest(directory, [{'path': 'model.json', 'kind': 'file'}], 'products.json')


def test_task_receipt_and_resume_verify_dynamic_models(tmp_path):
    import sys
    from test_production import task
    from hera_systematics_model.production import create_run, define_task
    from hera_systematics_model.worker import run_task, verified_receipt

    run = create_run(tmp_path / 'runs', 'a' * 40, {}, [])
    specification = task()
    specification['outputs'] = [{'path': 'products.json', 'kind': 'manifest'}]
    specification['command'] = [sys.executable, '-c',
        "import numpy as np\n"
        "from hera_systematics_model.product_manifest import create_product_manifest\n"
        "np.savez('model.npz', components=np.arange(6).reshape(2,3))\n"
        "create_product_manifest('.', [{'path':'model.npz','kind':'npz'}], 'products.json')\n"]
    define_task(run, specification)
    accepted = run_task(run, 'baseline-1', require_slurm=False)
    assert verified_receipt(run, 'baseline-1') == accepted
    assert run_task(run, 'baseline-1', require_slurm=False) == accepted
    np.savez(run / 'products/baseline-1/model.npz', components=np.zeros((2, 3)))
    with pytest.raises(ValueError, match='changed'):
        verified_receipt(run, 'baseline-1')
    with pytest.raises(ValueError, match='changed'):
        run_task(run, 'baseline-1', require_slurm=False)
