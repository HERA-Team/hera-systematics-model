"""Rank-revealing solves with finite, serializable conditioning evidence."""

import numpy as np

from .scoring import CandidateFailure


def solve_observed(design, values, rcond=1e-10):
    """Solve only supplied observations and expose the numerical support."""
    design, values = np.asarray(design), np.asarray(values)
    if (design.ndim != 2 or values.ndim not in (1, 2) or len(values) != len(design)
            or not np.isfinite(design).all() or not np.isfinite(values).all()):
        raise ValueError("finite observed design and response required")
    coefficients, _, effective, singular = np.linalg.lstsq(design, values, rcond=rcond)
    supported = effective == design.shape[1]
    condition = float(singular[0] / singular[-1]) if supported and len(singular) else None
    evidence = {"observations": len(design), "parameters": design.shape[1],
                "effective_rank": int(effective), "rcond": rcond,
                "singular_values": singular.tolist(), "condition_number": condition,
                "condition_unavailable_reason": None if condition is not None else "rank deficient or empty design"}
    if not supported:
        raise CandidateFailure("rank-deficient observed solve", evidence)
    return coefficients, evidence
