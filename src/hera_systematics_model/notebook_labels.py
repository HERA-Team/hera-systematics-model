"""Insert explicit label normalization after the final time average."""

import ast
import copy
import re


def instrument_labels(document):
    """Require one identified final average and retain the original notebook."""
    document = copy.deepcopy(document)
    matches = []
    for index, cell in enumerate(document.cells):
        if cell.cell_type != "code" or not re.search(r"(?m)^uvp_avg_all[ \t]*=", cell.source):
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "uvp_avg_all" for target in node.targets):
                matches.append((index, node.end_lineno))
    if len(matches) != 1:
        raise ValueError("exactly one final time-average assignment is required")
    index, end = matches[0]
    insertion = [
        "from pathlib import Path",
        "from hera_systematics_model.spectral_labels import canonicalize_time_average_labels",
        "from hera_systematics_model.production import write_json_exclusive",
        "_hsm_label_report = canonicalize_time_average_labels(uvp_avg_all)",
        "write_json_exclusive(Path(OUT_TAVG_PSPEC_FILE).with_name('label-metadata.json'), _hsm_label_report)",
    ]
    lines = document.cells[index].source.splitlines()
    lines[end:end] = insertion
    document.cells[index].source = "\n".join(lines)
    return document, {"policy": "collapse_identical_labels_after_final_time_average",
                      "report": "label-metadata.json"}
