"""Instrument captured spectral notebooks with explicit averaging-grid exports."""

import ast
import copy
import json
from pathlib import Path
import re

import numpy as np

from .configuration import file_identity
from .records import WindowGrid


def load_native_grid(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1 or value.get("policy") not in ("shared", "retained"):
        raise ValueError("unsupported native averaging grid schema or policy")
    native = np.asarray(value["native_time_jd"], dtype=float)
    grid = WindowGrid(value["anchor_jd"], value["window_seconds"])
    width = value["native_samples_per_window"]
    if (native.ndim != 1 or not len(native) or not np.isfinite(native).all()
            or np.any(np.diff(native) <= 0) or type(width) is not int or width <= 0
            or not value.get("sources")):
        raise ValueError("invalid native reference grid")
    if value["policy"] == "shared":
        if len(native) % width or not np.array_equal(grid.assign(native), np.arange(len(native)) // width):
            raise ValueError("native times do not occupy complete shared windows")
    return value


def shared_time_slice(input_times, native_times):
    """Use one full reference arc; never infer boundaries from visibility values."""
    if not np.array_equal(np.asarray(input_times), np.asarray(native_times)):
        raise ValueError("single-baseline native times differ from the common reference arc")
    return slice(0, len(native_times))


def instrument_averaging(document, grid_path):
    """Modify only the identified time-boundary assignment and append an export."""
    import nbformat

    grid_path = str(Path(grid_path).resolve())
    settings = load_native_grid(grid_path)
    document = copy.deepcopy(document)
    assignments = []
    for index, cell in enumerate(document.cells):
        if cell.cell_type != "code" or not re.search(r"(?m)^tslice[ \t]*=", cell.source):
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tslice" for t in node.targets):
                assignments.append((index, node.lineno, node.end_lineno))
    if len(assignments) != 1:
        raise ValueError("exactly one top-level native time-boundary assignment is required")
    index, start, end = assignments[0]
    if settings["policy"] == "shared":
        lines = document.cells[index].source.splitlines()
        lines[start - 1:end] = ["tslice = shared_time_slice(single_bl_times, _hsm_native_grid['native_time_jd'])"]
        document.cells[index].source = "\n".join(lines)
    setup = ("from hera_systematics_model.notebook_averaging import load_native_grid, shared_time_slice\n"
             f"_hsm_native_grid = load_native_grid({grid_path!r})\n")
    document.cells.insert(index, nbformat.v4.new_code_cell(setup))
    export = f'''from pathlib import Path
from hera_systematics_model.averaging_export import memberships_from_streams
from hera_systematics_model.configuration import file_identity
from hera_systematics_model.records import WindowGrid
_hsm_grid = WindowGrid(_hsm_native_grid['anchor_jd'], _hsm_native_grid['window_seconds'])
if Navg * NINTERLEAVE != _hsm_native_grid['native_samples_per_window']:
    raise ValueError('executed averaging width differs from the captured reference grid')
_hsm_pair = f'{{ANTPAIR[0]}}_{{ANTPAIR[1]}}'
_hsm_members = memberships_from_streams(
    _hsm_native_grid['native_time_jd'], [d.times for d in deint_filt_data],
    [d.times for d in deint_avg_data], interleaved_uvp.time_avg_array,
    _hsm_pair + ':' + _hsm_pair, _hsm_grid, Navg,
    {{'spectrum_source': file_identity(OUT_PSPEC_FILE),
      'native_time_source': file_identity({grid_path!r}),
      'averaging_configuration': {{'target_seconds': float(TARGET_AVERAGING_TIME),
          'actual_seconds': float(AVERAGING_TIME), 'interleave_auto_products': bool(INCLUDE_INTERLEAVE_AUTO_PS),
          'phase_to_common_lst': True, 'grid_policy': _hsm_native_grid['policy']}}}},
    require_shared_windows=_hsm_native_grid['policy'] == 'shared')
_hsm_members.save(str(Path(OUT_PSPEC_FILE).with_name('window-memberships.npz')))
'''
    document.cells.append(nbformat.v4.new_code_cell(export))
    return document, {"native_grid": file_identity(grid_path), "policy": settings["policy"],
                      "boundary_assignment_replaced": settings["policy"] == "shared"}
