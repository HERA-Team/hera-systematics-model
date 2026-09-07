"""Conservative delay-bin comparisons for descriptive mode diagnostics."""

import numpy as np

from .baselines import match_groups, pairwise_cosine


def center_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2 or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise ValueError("at least two ordered finite delay centers are required")
    middle = (centers[1:] + centers[:-1]) / 2
    return np.r_[centers[0] - (centers[1] - centers[0]) / 2, middle,
                 centers[-1] + (centers[-1] - centers[-2]) / 2]


def overlap_weights(source_edges, target_edges):
    """Integrate piecewise-constant source profiles over fully covered target bins."""
    source, target = np.asarray(source_edges, float), np.asarray(target_edges, float)
    if any(value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all()
           or np.any(np.diff(value) <= 0) for value in (source, target)):
        raise ValueError("ordered finite bin edges are required")
    overlap = np.maximum(0., np.minimum(target[1:, None], source[None, 1:])
                         - np.maximum(target[:-1, None], source[None, :-1]))
    tolerance = 32 * np.finfo(float).eps * max(np.max(np.abs(source)), np.max(np.abs(target)))
    overlap[overlap <= tolerance] = 0.
    weights = overlap / np.diff(target)[:, None]
    if not np.allclose(weights.sum(axis=1), 1., rtol=1e-12, atol=0):
        raise ValueError("target bins extend outside source support")
    return weights


def common_delay_bins(left, right):
    """Use full bins at least as wide as either native grid, within shared support."""
    a, b = center_edges(left), center_edges(right)
    lower, upper = max(a[0], b[0]), min(a[-1], b[-1])
    width = max(np.max(np.diff(a)), np.max(np.diff(b)))
    tolerance = 32 * np.finfo(float).eps * max(np.max(np.abs(a)), np.max(np.abs(b)))
    count = int(np.floor((upper - lower + tolerance) / width))
    if count < 1:
        raise ValueError("no complete common delay bin")
    edges = lower + np.arange(count + 1) * width
    return a, b, edges, overlap_weights(a, edges), overlap_weights(b, edges)


def rebin_profiles(values, support, weights):
    """Require every contributing native cell; absent support is never zero power."""
    values, support, weights = np.asarray(values, float), np.asarray(support), np.asarray(weights, float)
    if (values.ndim != 3 or support.shape != values.shape[1:] or support.dtype.kind != "b"
            or weights.ndim != 2 or weights.shape[1] != values.shape[2]
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1., rtol=1e-12, atol=0)
            or not np.isfinite(values[:, support]).all()):
        raise ValueError("mode profiles and rebin support disagree")
    available = (~support).astype(int) @ (weights > 0).T == 0
    averaged = np.einsum("rgd,bd->rgb", np.where(support, values, 0.), weights)
    return np.where(available, averaged, np.nan), available


def rebinned_mode_similarity(left, right, left_components, right_components, left_mask, right_mask):
    """Compare signed profiles and separately integrated squared loadings.

    This descriptive mapping assumes a constant profile within each native
    delay bin. It changes neither fits nor their native validation targets.
    It does not supply common spectral window functions or independent cells.
    """
    groups = match_groups(left, right)
    a, b, edges, wa, wb = common_delay_bins(left.delay_s, right.delay_s)
    signed, squared, masks = [], [], []
    for samples, components, mask, column, weights in zip((left, right), (left_components, right_components),
                                                          (left_mask, right_mask), (0, 1), (wa, wb)):
        components, mask = np.asarray(components), np.asarray(mask)
        shape = samples.corrupted.shape[1:]
        if (components.ndim != 2 or components.shape[1] != np.prod(shape)
                or mask.shape != (np.prod(shape),) or mask.dtype.kind != "b"):
            raise ValueError("mode dimensions disagree with physical sample coordinates")
        values = components.reshape((-1, *shape))[:, groups[:, column]]
        support = mask.reshape(shape)[groups[:, column]]
        profile, available = rebin_profiles(values, support, weights)
        energy, _ = rebin_profiles(values ** 2, support, weights)
        signed.append(profile)
        squared.append(energy)
        masks.append(available)
    use = masks[0] & masks[1]
    arrays = {"signed_similarity": pairwise_cosine(signed[0][:, use], signed[1][:, use]),
        "squared_loading_similarity": pairwise_cosine(squared[0][:, use], squared[1][:, use]),
        "left_source_edges_s": a, "right_source_edges_s": b, "common_edges_s": edges,
        "left_overlap_weights": wa, "right_overlap_weights": wb, "group_indices": groups,
        "left_signed_profiles": signed[0], "right_signed_profiles": signed[1],
        "left_squared_profiles": squared[0], "right_squared_profiles": squared[1],
        "scored_coordinate_mask": use, "left_supported_bins": masks[0], "right_supported_bins": masks[1]}
    metadata = {"matched_cells": int(use.sum()), "common_grid_cells": int(use.size),
        "left_native_cells": int(np.size(left_mask)), "right_native_cells": int(np.size(right_mask)),
        "left_supported_native_cells": int(np.count_nonzero(left_mask)),
        "right_supported_native_cells": int(np.count_nonzero(right_mask)),
        "matched_group_ids": left.group_ids[groups[:, 0]].tolist(),
        "bin_width_s": float(edges[1] - edges[0]), "common_delay_width_s": float(edges[-1] - edges[0]),
        "comparison": "overlap-weighted piecewise-constant native delay profiles",
        "squared_loading_operation": "square native loadings before conservative averaging",
        "uses_validation_for_selection": False,
        "limitations": ["bin-constant descriptive approximation", "correlated native cells",
                        "no common spectral window-function calculation", "incomplete edge bins excluded"]}
    return arrays, metadata
