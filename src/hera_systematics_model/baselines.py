"""Baseline membership, loading localization and matched-coordinate similarity."""

import numpy as np

from .statistics import summary


def group_delay_shares(component, shape):
    values = np.asarray(component).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("finite component required")
    energy = values ** 2
    total = energy.sum()
    if total == 0:
        return None, None
    return energy.sum(axis=1) / total, energy.sum(axis=0) / total


def baseline_inventory(samples):
    """Report every physical contributor, including its measured weight support."""
    result = []
    residual = samples.residual
    for group, identity in enumerate(samples.group_ids):
        indices = np.flatnonzero(samples.baseline_group == group)
        valid = samples.valid[:, group]
        result.append({"group_id": str(identity), "length_m": float(samples.baseline_length_m[group]),
            "kperp": float(samples.kperp[group]), "valid_cells": int(valid.sum()), "total_cells": int(valid.size),
            "corrupted": summary(samples.corrupted[:, group][valid]),
            "ideal": summary(samples.ideal[:, group][valid]), "residual": summary(residual[:, group][valid]),
            "noise": summary(samples.pn[:, group][valid]),
            "baselines": [{"baseline_id": str(samples.baseline_ids[i]),
                "positive_weight_cells": int(np.count_nonzero(samples.weights[:, i])),
                "total_weight": float(samples.weights[:, i].sum())} for i in indices]})
    return result


def mode_localization(samples, components, representation, threshold=.3):
    """Squared loading shares are reported in the selected representation."""
    components = np.asarray(components)
    shape = samples.corrupted.shape[1:]
    if components.ndim != 2 or components.shape[1] != np.prod(shape):
        raise ValueError("mode and sample feature axes disagree")
    result = []
    for mode, component in enumerate(components):
        groups, delays = group_delay_shares(component, shape)
        if groups is None:
            result.append({"mode": mode, "available": False, "reason": "zero loading energy"})
            continue
        ordering = sorted(range(len(groups)), key=lambda i: (-groups[i], str(samples.group_ids[i])))
        result.append({"mode": mode, "available": True, "representation": representation,
            "effective_groups": float(1 / (groups @ groups)), "effective_delays": float(1 / (delays @ delays)),
            "kparallel_threshold": threshold, "kparallel_units": "h Mpc^-1",
            "loading_share_above_threshold": float(delays[samples.kparallel > threshold].sum()),
            "groups": [{"group_id": str(samples.group_ids[i]), "loading_share": float(groups[i]),
                "baseline_ids": samples.baseline_ids[samples.baseline_group == i].tolist()} for i in ordering],
            "delay_loading_shares": delays.tolist()})
    return result


def match_coordinates(left, right, delay_atol=1e-14):
    """Match physical group identities and native delay centers without interpolation."""
    for name in ("polarization", "power_units", "cosmology"):
        if left.metadata[name] != right.metadata[name]:
            raise ValueError(f"cross-window metadata mismatch: {name}")
    left_groups = {value: i for i, value in enumerate(left.group_ids)}
    right_groups = {value: i for i, value in enumerate(right.group_ids)}
    groups = sorted(left_groups.keys() & right_groups.keys())
    pairs = []
    for group in groups:
        li, ri = left_groups[group], right_groups[group]
        if not np.isclose(left.baseline_length_m[li], right.baseline_length_m[ri], rtol=1e-10, atol=1e-8):
            raise ValueError("matched group has different physical geometry")
        for ld, delay in enumerate(left.delay_s):
            rd = np.flatnonzero(np.isclose(right.delay_s, delay, rtol=1e-10, atol=delay_atol))
            if len(rd) > 1:
                raise ValueError("ambiguous matching delay centers")
            if len(rd):
                pairs.append((li * len(left.delay_s) + ld, ri * len(right.delay_s) + rd[0]))
    if not pairs:
        raise ValueError("no matched physical group/delay coordinates")
    indices = np.asarray(pairs, int)
    if len(np.unique(indices[:, 1])) != len(indices):
        raise ValueError("cross-window coordinates are not one-to-one")
    return indices


def cross_window_similarity(left, right, left_components, right_components, left_mask, right_mask):
    """Keep signed-mode and squared-loading cosine similarities separate."""
    indices = match_coordinates(left, right)
    left_components, right_components = np.asarray(left_components), np.asarray(right_components)
    masks = [np.asarray(mask) for mask in (left_mask, right_mask)]
    for samples, components, mask in zip((left, right), (left_components, right_components), masks):
        if (components.ndim != 2 or components.shape[1] != np.prod(samples.corrupted.shape[1:])
                or mask.shape != (components.shape[1],) or mask.dtype.kind != "b"):
            raise ValueError("component physical identity dimensions disagree")
    use = masks[0][indices[:, 0]] & masks[1][indices[:, 1]]
    a, b = left_components[:, indices[use, 0]], right_components[:, indices[use, 1]]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite loading on shared support")
    def cosine(x, y):
        denominator = np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=1)[None, :]
        return np.divide(x @ y.T, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
    return {"signed_similarity": cosine(a, b), "squared_loading_similarity": cosine(a ** 2, b ** 2),
            "coordinate_indices": indices, "scored_coordinate_mask": use,
            "matched_cells": int(use.sum()), "left_cells": len(left_mask), "right_cells": len(right_mask)}
