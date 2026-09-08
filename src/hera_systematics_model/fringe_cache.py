"""Resolve missing fringe-rate cache identities from verified baseline geometry."""

import numpy as np


def _pair(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(a) is not int or a < 0 for a in value)):
        raise ValueError("invalid physical baseline pair")
    return tuple(value)


def resolve_cache_aliases(groups, mapping, tolerance_m=1e-6):
    """Return additions only; retain existing cache indices and orientation.

    ``mapping`` contains geometry-verified ideal source correspondences. A
    reversed cached source receives the reversed reference pair, so the reader's
    existing fringe-rate sign reversal remains applicable. No spectrum is changed.
    The caller must bind the consumed mapping and cache to accepted input hashes.
    """
    if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 1e-6:
        raise ValueError("invalid geometry tolerance")
    lookup = {}
    for index, pairs in groups.items():
        if type(index) is not int or index < 0:
            raise ValueError("invalid cache index")
        for value in pairs:
            pair = _pair(value)
            if pair in lookup:
                raise ValueError("duplicate cache baseline identity")
            lookup[pair] = index
    additions = []
    references = set()
    for key, entry in sorted(mapping.items()):
        pair = _pair(entry['reference_pair'])
        if key != f'{pair[0]}_{pair[1]}' or pair in references or pair[::-1] in references:
            raise ValueError("inconsistent reference identity")
        references.add(pair)
        if pair[0] == pair[1] or pair in lookup or pair[::-1] in lookup:
            continue
        if entry.get('exclusion') is not None or entry.get('source_pair') is None:
            raise ValueError("missing supported source correspondence")
        source = _pair(entry['source_pair'])
        stored = _pair(entry['stored_pair'])
        conjugate = entry['conjugate']
        if type(conjugate) is not bool or source != (stored[::-1] if conjugate else stored):
            raise ValueError("source orientation disagrees with stored pair")
        vectors = [np.asarray(entry[name], dtype=float) for name in
                   ('reference_vector_enu_m', 'source_vector_enu_m')]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("invalid baseline geometry")
        difference = float(np.linalg.norm(vectors[0] - vectors[1]))
        declared = entry['vector_tolerance_m']
        if (not np.isfinite(declared) or not 0 < declared <= tolerance_m
                or difference > declared
                or not np.isclose(difference, entry['vector_difference_m'], rtol=1e-6, atol=1e-12)):
            raise ValueError("source and reference geometry disagree")
        if source in lookup and source[::-1] in lookup:
            raise ValueError("ambiguous cached source orientation")
        reverse = source not in lookup
        cached = source[::-1] if reverse else source
        if cached not in lookup:
            raise ValueError("source baseline absent from cache")
        alias = pair[::-1] if reverse else pair
        additions.append({'reference_pair': list(pair), 'source_pair': list(source),
                          'cached_source_pair': list(cached), 'cache_pair': list(alias),
                          'cache_index': lookup[cached], 'reversed': reverse,
                          'vector_difference_m': difference,
                          'reference_vector_enu_m': vectors[0].tolist(),
                          'source_vector_enu_m': vectors[1].tolist(),
                          'vector_tolerance_m': declared})
    return additions


def copy_with_aliases(source, mapping_file, output, expected_source, expected_mapping):
    """Create an exclusive cache copy bound to caller-verified input identities.

    Only baseline-group membership datasets may gain rows. Spectra, coordinates,
    existing group rows, dataset types, and all original attributes are verified.
    Partial files are retained on failure, without a successful sidecar.
    """
    import json
    import shutil
    from pathlib import Path
    import h5py
    from .configuration import file_identity
    from .production import write_json_exclusive

    source, mapping_file, output = map(Path, (source, mapping_file, output))
    sidecar = output.with_suffix('.aliases.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    if file_identity(source) != expected_source or file_identity(mapping_file) != expected_mapping:
        raise ValueError('cache or mapping input identity differs')
    mapping = json.loads(mapping_file.read_text())['baseline_mapping']
    prefix = 'metadata/baseline_groups/'
    with h5py.File(source, 'r') as handle:
        groups = {int(k): v[:].tolist() for k, v in handle['metadata/baseline_groups'].items()}
        spectrum = handle['erh_mode_power_spectrum']
        if spectrum.ndim != 3 or any(i >= spectrum.shape[2] for i in groups):
            raise ValueError('cache group index outside spectrum axis')
        if int(handle['metadata/baseline_dimension'][()]) != 2:
            raise ValueError('unsupported cache baseline axis')
        aliases = resolve_cache_aliases(groups, mapping)
    additions = {}
    for entry in aliases:
        additions.setdefault(prefix + str(entry['cache_index']), []).append(entry['cache_pair'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, output.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    with h5py.File(output, 'r+') as handle:
        for name, rows in additions.items():
            old = handle[name]
            attrs, dtype = dict(old.attrs), old.dtype
            values = np.concatenate([old[:], np.asarray(rows, dtype=dtype)])
            del handle[name]
            new = handle.create_dataset(name, data=values)
            new.attrs.update(attrs)
    checked = []
    with h5py.File(source, 'r') as original, h5py.File(output, 'r') as copied:
        def equal(a, b):
            a, b = np.asarray(a), np.asarray(b)
            return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)

        def compare(name, obj):
            actual = copied[name] if name else copied
            if set(obj.attrs) != set(actual.attrs) or any(not equal(v, actual.attrs[k]) for k, v in obj.attrs.items()):
                raise ValueError('cache copy changed attributes')
            if isinstance(obj, h5py.Group):
                if set(obj) != set(actual):
                    raise ValueError('cache copy changed object inventory')
                return
            shape = (obj.shape[0] + len(additions[name]), 2) if name in additions else obj.shape
            if actual.shape != shape or actual.dtype != obj.dtype:
                raise ValueError('cache copy changed dataset structure')
            for start in range(0, obj.shape[0], 8) if obj.shape else [None]:
                key = () if start is None else slice(start, min(start + 8, obj.shape[0]))
                if not equal(obj[key], actual[key]):
                    raise ValueError('cache copy changed original values')
            if name in additions and not equal(actual[obj.shape[0]:], additions[name]):
                raise ValueError('cache alias rows differ')
            checked.append(name)
        compare('', original)
        original.visititems(compare)
        result_groups = {int(k): v[:].tolist() for k, v in copied['metadata/baseline_groups'].items()}
        if resolve_cache_aliases(result_groups, mapping):
            raise ValueError('cache alias coverage remains incomplete')
    if file_identity(source) != expected_source or file_identity(mapping_file) != expected_mapping:
        raise ValueError('cache or mapping changed during copying')
    result = {'schema_version': 1, 'passed': True, 'source': expected_source,
              'mapping': expected_mapping, 'output': file_identity(output), 'aliases': aliases,
              'verified_original_datasets': checked, 'spectral_values_changed': False}
    write_json_exclusive(sidecar, result)
    return result
