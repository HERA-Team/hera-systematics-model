"""Array views for full-plane, fixed-baseline and fixed-delay analysis."""

import numpy as np

from .prediction import candidate_grid
from .evaluation import evaluate_nested


def analysis_view(samples, group=None, delay=None):
    """Preserve time identities while selecting exactly one cylindrical view."""
    if group is not None and delay is not None:
        raise ValueError("choose a baseline slice or a delay slice, not both")
    ng, nd = samples.corrupted.shape[1:]
    if group is not None and not 0 <= group < ng:
        raise ValueError("group index out of bounds")
    if delay is not None and not 0 <= delay < nd:
        raise ValueError("delay index out of bounds")
    gs = slice(None) if group is None else slice(group, group + 1)
    ds = slice(None) if delay is None else slice(delay, delay + 1)
    shape = (ng if group is None else 1, nd if delay is None else 1)
    arrays = tuple(getattr(samples, name)[:, gs, ds].reshape(len(samples.window_ids), -1)
                   for name in ("corrupted", "ideal", "pn", "valid"))
    identity = {"spw": samples.metadata["spw"], "polarization": samples.metadata["polarization"],
                "power_units": samples.metadata["power_units"], "cosmology": samples.metadata["cosmology"],
                "group_ids": samples.group_ids[gs].tolist(), "kperp": samples.kperp[gs].tolist(),
                "delay_s": samples.delay_s[ds].tolist(), "kparallel": samples.kparallel[ds].tolist()}
    return arrays, shape, identity


def sample_view(samples, group=None, delay=None):
    """Select diagnostic coordinates and contributors without changing weights."""
    from dataclasses import replace

    analysis_view(samples, group, delay)
    gs = slice(None) if group is None else slice(group, group + 1)
    ds = slice(None) if delay is None else slice(delay, delay + 1)
    baselines = np.ones(len(samples.baseline_ids), bool) if group is None else samples.baseline_group == group
    values = {name: getattr(samples, name)[:, gs, ds].copy()
              for name in ("corrupted", "ideal", "pn", "valid")}
    values.update({name: getattr(samples, name)[gs].copy()
                   for name in ("group_ids", "baseline_length_m", "kperp")})
    values.update({name: getattr(samples, name)[ds].copy() for name in ("delay_s", "kparallel")})
    return replace(samples, **values, baseline_ids=samples.baseline_ids[baselines].copy(),
        baseline_group=(samples.baseline_group[baselines].copy() if group is None
                        else np.zeros(int(baselines.sum()), int)),
        weights=samples.weights[:, baselines, ds].copy())


def evaluate_slice(samples, representation, group=None, delay=None, max_rank=20, guard=12):
    arrays, shape, identity = analysis_view(samples, group, delay)
    candidates = candidate_grid(max_rank, include_kernel=False, representations=[representation])
    result = evaluate_nested(arrays, samples.window_ids, shape, candidates, guard=guard,
                             axis="group" if delay is not None else "delay")
    result.metadata["identity"] = identity
    return result


def geometry_masks(samples, horizon_delay_s, buffer_ns=500.):
    """Make geometric regions without consulting power amplitudes."""
    horizon_delay_s = np.asarray(horizon_delay_s)
    if horizon_delay_s.shape != (len(samples.group_ids),) or not np.isfinite(horizon_delay_s).all():
        raise ValueError("one finite horizon delay per baseline group required")
    if np.any(horizon_delay_s < 0) or not np.isfinite(buffer_ns) or buffer_ns < 0:
        raise ValueError("negative horizon or invalid buffer")
    delays = samples.delay_s[None, :]
    return {"full": np.ones(samples.corrupted.shape[1:], bool),
            "horizon": delays > horizon_delay_s[:, None],
            "horizon_buffer": delays > horizon_delay_s[:, None] + buffer_ns * 1e-9}
