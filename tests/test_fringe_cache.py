from copy import deepcopy

import pytest

from hera_systematics_model.fringe_cache import resolve_cache_aliases


def correspondence():
    return {'0_326': {'reference_pair': [0, 326], 'source_pair': [196, 2],
            'stored_pair': [196, 2], 'conjugate': False, 'exclusion': None,
            'reference_vector_enu_m': [80., -155., 1.],
            'source_vector_enu_m': [80., -155., 1.],
            'vector_difference_m': 0., 'vector_tolerance_m': 1e-6}}


@pytest.mark.parametrize('reverse', [False, True])
def test_alias_preserves_cache_index_and_reader_orientation(reverse):
    groups = {7: [[2, 196] if reverse else [196, 2]]}
    mapping = correspondence()
    original = deepcopy((groups, mapping))
    result = resolve_cache_aliases(groups, mapping)
    assert result[0]['cache_index'] == 7
    assert result[0]['cache_pair'] == ([326, 0] if reverse else [0, 326])
    assert result[0]['reversed'] is reverse
    assert (groups, mapping) == original


def test_existing_reference_mapping_is_preserved():
    assert resolve_cache_aliases({9: [[326, 0]], 7: [[196, 2]]}, correspondence()) == []


@pytest.mark.parametrize('mutation', ['vector', 'nan', 'orientation', 'unsupported', 'difference', 'tolerance'])
def test_invalid_geometry_or_source_fails(mutation):
    mapping = correspondence(); e = mapping['0_326']
    if mutation == 'vector': e['source_vector_enu_m'][0] += 1
    if mutation == 'nan': e['source_vector_enu_m'][0] = float('nan')
    if mutation == 'orientation': e['conjugate'] = True
    if mutation == 'unsupported': e['source_pair'] = None
    if mutation == 'difference': e['vector_difference_m'] = .1
    if mutation == 'tolerance': e['vector_tolerance_m'] = 1.
    with pytest.raises(ValueError):
        resolve_cache_aliases({7: [[196, 2]]}, mapping)


@pytest.mark.parametrize('groups', [{}, {7: [[196, 2], [196, 2]]},
                                   {7: [[196, 2]], 8: [[2, 196]]}])
def test_missing_duplicate_or_ambiguous_cache_fails(groups):
    with pytest.raises(ValueError):
        resolve_cache_aliases(groups, correspondence())


def cache_files(tmp_path):
    import json
    import h5py
    import numpy as np
    source, mapping = tmp_path / 'cache.h5', tmp_path / 'mapping.json'
    with h5py.File(source, 'w') as f:
        f.attrs['version'] = 'original'
        f.create_dataset('metadata/baseline_dimension', data=2)
        f.create_dataset('metadata/baseline_groups/0', data=[[196, 2]])
        f.create_dataset('metadata/frequencies_MHz', data=[100., 150.])
        d = f.create_dataset('erh_mode_power_spectrum', data=np.arange(12.).reshape(6, 2, 1))
        d.attrs['baseline_dimension'] = 2
    mapping.write_text(json.dumps({'baseline_mapping': correspondence()}))
    return source, mapping


def test_copy_binds_inputs_preserves_payload_and_rejects_overwrite(tmp_path):
    import h5py
    import numpy as np
    from hera_systematics_model.configuration import file_identity
    from hera_systematics_model.fringe_cache import copy_with_aliases
    source, mapping = cache_files(tmp_path)
    identities = file_identity(source), file_identity(mapping)
    output = tmp_path / 'extended.h5'
    result = copy_with_aliases(source, mapping, output, *identities)
    assert result['passed'] and len(result['aliases']) == 1
    assert result['spectral_values_changed'] is False
    with h5py.File(output) as f:
        np.testing.assert_array_equal(f['erh_mode_power_spectrum'][:], np.arange(12.).reshape(6, 2, 1))
        assert f['metadata/baseline_groups/0'][:].tolist() == [[196, 2], [0, 326]]
        assert f.attrs['version'] == 'original'
    assert (file_identity(source), file_identity(mapping)) == identities
    with pytest.raises(FileExistsError):
        copy_with_aliases(source, mapping, output, *identities)


def test_copy_rejects_wrong_input_hash_before_output(tmp_path):
    from hera_systematics_model.configuration import file_identity
    from hera_systematics_model.fringe_cache import copy_with_aliases
    source, mapping = cache_files(tmp_path)
    output = tmp_path / 'extended.h5'
    with pytest.raises(ValueError, match='identity'):
        copy_with_aliases(source, mapping, output, {**file_identity(source), 'sha256': '0'*64}, file_identity(mapping))
    assert not output.exists()


def test_copy_detects_changed_payload_and_retains_failed_product(tmp_path, monkeypatch):
    import h5py
    import shutil
    from hera_systematics_model.configuration import file_identity
    from hera_systematics_model.fringe_cache import copy_with_aliases
    source, mapping = cache_files(tmp_path)
    original = shutil.copyfileobj
    def corrupt(reader, writer, **kwargs):
        original(reader, writer, **kwargs)
        writer.flush()
        with h5py.File(writer.name, 'r+') as f:
            f['erh_mode_power_spectrum'][0, 0, 0] = -999
    monkeypatch.setattr(shutil, 'copyfileobj', corrupt)
    output = tmp_path / 'extended.h5'
    with pytest.raises(ValueError, match='original values'):
        copy_with_aliases(source, mapping, output, file_identity(source), file_identity(mapping))
    assert output.exists() and not output.with_suffix('.aliases.json').exists()
