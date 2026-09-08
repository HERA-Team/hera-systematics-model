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
