"""Common-contributor averaging followed by verified symmetric delay folding."""

import numpy as np

from .records import WindowGrid, matched_indices
from .samples import PairedSamples


def delay_pairs(delay_s, kparallel):
    """Pair each strictly positive bin; omit DC and unpaired negative endpoints."""
    delay_s, kparallel = np.asarray(delay_s), np.asarray(kparallel)
    positive = np.flatnonzero(delay_s > 0)
    if not len(positive):
        raise ValueError("no positive delay bins")
    negative = []
    for index in positive:
        match = np.flatnonzero(np.isclose(delay_s, -delay_s[index], rtol=1e-10, atol=1e-15))
        if len(match) != 1:
            raise ValueError("positive delay lacks a unique negative counterpart")
        if not np.isclose(kparallel[match[0]], -kparallel[index], rtol=1e-10, atol=1e-12):
            raise ValueError("delay-side kparallel mismatch")
        negative.append(match[0])
    return np.column_stack([negative, positive]).astype(int)


def build_paired(corrupted, ideal, quorum=0.95):
    """Average both branches with the same corrupted-derived weights.

    Noise propagation treats contributing baselines and the two delay sides as
    independent. Ideal noise metadata does not enter either branch's weights.
    """
    if not 0 < quorum <= 1:
        raise ValueError("quorum must lie in (0, 1]")
    ci, ii, matching = matched_indices(corrupted, ideal)
    pairs = delay_pairs(corrupted.delay_s, corrupted.kparallel)
    baselines = sorted(set(corrupted.baseline_ids) & set(ideal.baseline_ids))
    bl_position = {baseline: pos for pos, baseline in enumerate(baselines)}
    first = {baseline: int(np.flatnonzero(corrupted.baseline_ids == baseline)[0])
             for baseline in baselines}
    groups = sorted({corrupted.group_ids[first[b]] for b in baselines})
    lengths = {group: float(np.mean([corrupted.baseline_length_m[first[b]] for b in baselines
                                    if corrupted.group_ids[first[b]] == group])) for group in groups}
    groups.sort(key=lambda group: (lengths[group], group))
    group_position = {group: pos for pos, group in enumerate(groups)}
    membership = np.array([group_position[corrupted.group_ids[first[b]]] for b in baselines])
    good = (corrupted.valid[ci][:, pairs] & ideal.valid[ii][:, pairs]
            & np.isfinite(corrupted.pn[ci][:, pairs]) & (corrupted.pn[ci][:, pairs] > 0))
    row_usable = good.any(axis=(1, 2))
    all_windows = np.unique(corrupted.window_ids)
    windows = np.array([wid for wid in all_windows
                        if np.count_nonzero((corrupted.window_ids[ci] == wid) & row_usable)
                        >= int(np.ceil(quorum * len(baselines)))], dtype=int)
    if not len(windows):
        raise ValueError("no paired windows meet quorum")
    shape = (len(windows), len(groups), len(pairs))
    pc, pi, pn = (np.full(shape, np.nan) for _ in range(3))
    valid = np.zeros(shape, dtype=bool)
    weights = np.zeros((len(windows), len(baselines), len(pairs), 2))
    lst = np.empty(len(windows))
    for ti, wid in enumerate(windows):
        rows = np.flatnonzero(corrupted.window_ids[ci] == wid)
        lst[ti] = np.angle(np.mean(np.exp(1j * corrupted.lst_rad[ci[rows]]))) % (2 * np.pi)
        for gi, group in enumerate(groups):
            selected = rows[corrupted.group_ids[ci[rows]] == group]
            if not len(selected):
                continue
            measured = good[selected]
            errors = corrupted.pn[ci[selected]][:, pairs]
            safe_errors = np.where(measured, errors, np.inf)
            smallest = safe_errors.min(axis=0)
            has_side = np.isfinite(smallest)
            ratio = np.zeros_like(safe_errors)
            np.divide(np.where(has_side, smallest, 0), safe_errors, out=ratio, where=measured)
            raw_weights = ratio ** 2
            total = raw_weights.sum(axis=0)
            normalized = np.divide(raw_weights, total, out=np.zeros_like(raw_weights), where=total > 0)
            cell_valid = has_side.all(axis=-1)
            folded_weights = normalized * .5 * cell_valid[None, :, None]
            for ri, row in enumerate(selected):
                bi = bl_position[corrupted.baseline_ids[ci[row]]]
                weights[ti, bi] = folded_weights[ri]
            for output, record, indices in ((pc, corrupted, ci), (pi, ideal, ii)):
                data = record.power[indices[selected]][:, pairs]
                average = (np.where(measured, data, 0) * folded_weights).sum(axis=(0, 2))
                output[ti, gi] = np.where(cell_valid, average, np.nan)
            variance = ((np.where(measured, errors, 0) * folded_weights) ** 2).sum(axis=(0, 2))
            pn[ti, gi] = np.where(cell_valid, np.sqrt(variance), np.nan)
            valid[ti, gi] = cell_valid
    grid = WindowGrid(corrupted.metadata["window_anchor_jd"], corrupted.metadata["window_seconds"])
    group_kperp = [float(np.mean([corrupted.kperp[first[b]] for b in baselines
                                  if corrupted.group_ids[first[b]] == group])) for group in groups]
    metadata = {**corrupted.metadata,
                "sources": {"corrupted": corrupted.metadata["sources"], "ideal": ideal.metadata["sources"]},
                "noise_model": "corrupted_diagonal_independent_baselines_and_delay_sides",
                "quorum": quorum, "matching": matching,
                "excluded_window_ids": sorted(set(all_windows.tolist()) - set(windows.tolist())),
                "omitted_delay_s": [float(value) for idx, value in enumerate(corrupted.delay_s)
                                    if idx not in pairs.ravel()]}
    return PairedSamples(
        corrupted=pc, ideal=pi, pn=pn, valid=valid, window_ids=windows,
        time_jd=grid.centers(windows), lst_rad=lst, group_ids=np.asarray(groups),
        baseline_length_m=np.array([lengths[group] for group in groups]),
        delay_s=corrupted.delay_s[pairs[:, 1]], kperp=np.asarray(group_kperp),
        kparallel=corrupted.kparallel[pairs[:, 1]], baseline_ids=np.asarray(baselines),
        baseline_group=membership, weights=weights, metadata=metadata,
    )
