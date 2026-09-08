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
