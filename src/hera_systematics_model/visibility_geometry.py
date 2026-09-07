"""Explicit physical geometry correction for unprojected visibility metadata."""

import numpy as np


def recalculate_unprojected_uvws(metadata, tolerance_m=1e-6):
    """Recalculate each row from antenna positions without changing visibility phases.

    This operation is restricted to metadata-only, unprojected data. Phased
    observations require a separately specified phase and geometry procedure.
    """
    if metadata.data_array is not None:
        raise ValueError("UVW correction requires metadata-only input")
    kinds = {metadata.phase_center_catalog[int(identity)]["cat_type"]
             for identity in np.unique(metadata.phase_center_id_array)}
    if kinds != {"unprojected"}:
        raise ValueError("UVW correction requires entirely unprojected coordinates")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("positive finite geometry tolerance required")
    stored = metadata.uvw_array.copy()
    if not np.isfinite(stored).all():
        raise ValueError("input UVW coordinates are nonfinite")
    rectangular = metadata.blts_are_rectangular
    try:
        # Calculate from every actual row identity, independently of any
        # rectangular ordering optimization in the input metadata.
        metadata.blts_are_rectangular = False
        metadata.set_uvws_from_antenna_positions(update_vis=False)
    finally:
        metadata.blts_are_rectangular = rectangular
    metadata.check(strict_uvw_antpos_check=True)
    differences = np.linalg.norm(metadata.uvw_array - stored, axis=1)
    return {"policy": "recalculate_unprojected", "rows": len(differences),
        "changed_rows": int(np.count_nonzero(differences > tolerance_m)),
        "maximum_uvw_change_m": float(np.max(differences)), "change_tolerance_m": tolerance_m,
        "phase_center_types": sorted(kinds), "visibility_phase_applied": False,
        "physical_geometry_check": True}
