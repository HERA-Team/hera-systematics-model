"""Export measured native membership from interleaved averaging state."""

import itertools

import numpy as np

from .window_membership import WindowMemberships


def memberships_from_streams(native_times, stream_times, averaged_times, output_times,
                             baseline_id, grid, samples_per_stream, metadata,
                             require_shared_windows=True):
    """Bind actual spectral centroids to the native rows consumed by averaging.

    Native time lookup is exact. Centroid arithmetic is checked within two
    floating-point spacings to allow equivalent orders of summation. This
    tolerance is never used to join rows or assign native identities.
    """
    native_times, output_times = np.asarray(native_times), np.asarray(output_times)
    streams = [np.asarray(value) for value in stream_times]
    averages = [np.asarray(value) for value in averaged_times]
    if (native_times.ndim != 1 or not len(native_times) or not np.isfinite(native_times).all()
            or np.any(np.diff(native_times) <= 0) or output_times.ndim != 1
            or not len(output_times) or not np.isfinite(output_times).all()
            or np.any(np.diff(output_times) <= 0)):
        raise ValueError("ordered native and output physical times are required")
    if (type(samples_per_stream) is not int or samples_per_stream < 1
            or len(streams) < 2 or len(streams) != len(averages)):
        raise ValueError("invalid interleaved averaging configuration")
    nrows = len(output_times)
    positions = {float(time): i for i, time in enumerate(native_times)}
    indices = []
    tolerance = 2 * np.max(np.abs(np.spacing(native_times)))
    for stream, average in zip(streams, averages):
        if (stream.ndim != 1 or not len(stream) or not np.isfinite(stream).all()
                or np.any(np.diff(stream) <= 0) or average.shape != (nrows,)
                or int(np.ceil(len(stream) / samples_per_stream)) != nrows):
            raise ValueError("averaging stream lengths or physical order disagree")
        if any(float(time) not in positions for time in stream):
            raise ValueError("stream time has no exact native identity")
        ids = np.array([positions[float(time)] for time in stream])
        expected = np.array([np.mean(stream[i:i + samples_per_stream])
                             for i in range(0, len(stream), samples_per_stream)])
        if not np.allclose(average, expected, rtol=0, atol=tolerance):
            raise ValueError("averaged stream centroid differs from consumed native samples")
        indices.append(ids)
    flattened = np.concatenate(indices)
    if len(np.unique(flattened)) != len(flattened):
        raise ValueError("native samples occur in multiple interleaves")
    # Unique cross-products of streams are averaged incoherently. Each stream
    # appears equally often, including when auto-products are also retained.
    pair_centroids = [(averages[a] + averages[b]) / 2
                      for a, b in itertools.combinations(range(len(streams)), 2)]
    if not np.allclose(output_times, np.mean(pair_centroids, axis=0), rtol=0, atol=tolerance):
        raise ValueError("spectral centroid differs from interleave membership")
    width = samples_per_stream * len(streams)
    members = np.full((nrows, width), -1, dtype=np.int64)
    for row in range(nrows):
        selected = np.sort(np.concatenate([ids[row * samples_per_stream:(row + 1) * samples_per_stream]
                                          for ids in indices]))
        members[row, :len(selected)] = selected
    ids = grid.assign(output_times)
    if require_shared_windows:
        for identity, row in zip(ids, members):
            if not np.array_equal(row, np.arange(identity * width, (identity + 1) * width)):
                raise ValueError("native averaging membership is not aligned to the shared reference grid")
    result = WindowMemberships(np.repeat(baseline_id, nrows), output_times, ids, members, native_times,
        {**metadata, "n_interleaves": len(streams), "averaging_configuration": {
            **metadata.get("averaging_configuration", {}), "samples_per_stream": samples_per_stream,
            "native_samples_per_window": width, "shared_native_windows": require_shared_windows}})
    return result


def merge_memberships(memberships, baseline_ids, centroid_jd, spectrum_source):
    """Rebind exact single-baseline exports to a validated merged spectral file."""
    from .artifacts import canonical_json

    if not memberships:
        raise ValueError("native membership inputs are required")
    first = memberships[0]
    for value in memberships[1:]:
        if (value.native_grid_digest != first.native_grid_digest
                or value.metadata["n_interleaves"] != first.metadata["n_interleaves"]
                or canonical_json(value.metadata["averaging_configuration"])
                != canonical_json(first.metadata["averaging_configuration"])):
            raise ValueError("native membership inputs use different averaging grids")
    width = max(value.native_ids.shape[1] for value in memberships)
    combined = WindowMemberships(np.concatenate([value.baseline_ids for value in memberships]),
        np.concatenate([value.centroid_jd for value in memberships]),
        np.concatenate([value.window_ids for value in memberships]),
        np.concatenate([np.pad(value.native_ids, ((0, 0), (0, width - value.native_ids.shape[1])),
                               constant_values=-1) for value in memberships]), first.native_time_jd,
        {**first.metadata, "spectrum_source": spectrum_source,
         "constituent_spectra": [value.metadata["spectrum_source"] for value in memberships]})
    if len(baseline_ids) != len(combined.baseline_ids):
        raise ValueError("merged spectrum row count differs from native membership exports")
    ids, native = combined.lookup(baseline_ids, centroid_jd)
    return WindowMemberships(np.asarray(baseline_ids), np.asarray(centroid_jd), ids, native,
                             combined.native_time_jd, combined.metadata)
