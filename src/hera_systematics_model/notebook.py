"""Execute a captured spectral notebook with explicit output and kernel paths."""

import argparse
import ast
import json
import os
from pathlib import Path
import sys

from .configuration import file_identity
from .production import write_json_exclusive


SPECTRAL_MODULES = ["numpy", "scipy", "astropy", "h5py", "pyuvdata", "hera_cal", "hera_pspec",
                    "hera_filters", "hera_qm", "hera_notebook_templates"]


def notebook_parameters(notebook, configuration, single_baseline, output_dir):
    """Validate overrides against the captured parameter cell."""
    document = json.loads(Path(notebook).read_text())
    cells = [cell for cell in document["cells"] if "parameters" in cell.get("metadata", {}).get("tags", [])]
    if len(cells) != 1:
        raise ValueError("exactly one tagged parameter cell required")
    names = set()
    for node in ast.parse("".join(cells[0]["source"])).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    if not set(configuration) <= names:
        raise ValueError("configuration contains unknown notebook parameters")
    output_dir = Path(output_dir).resolve()
    parameters = {**configuration, "SINGLE_BL_FILE": str(Path(single_baseline).resolve()),
                  "OUT_PSPEC_FILE": str(output_dir / "spectrum.pspec.h5"),
                  "OUT_TAVG_PSPEC_FILE": str(output_dir / "spectrum.tavg.pspec.h5")}
    if not set(parameters) <= names or not parameters.get("SAVE_RESULTS", True):
        raise ValueError("notebook cannot write the required spectral products")
    return parameters


def execute_spectrum(notebook, configuration, single_baseline, output_dir, native_grid=None):
    """Execute with this interpreter; errors propagate to the compute worker."""
    import papermill
    import nbformat

    output_dir = Path(output_dir).resolve()
    parameters = notebook_parameters(notebook, configuration, single_baseline, output_dir)
    document = nbformat.read(notebook, as_version=4)
    averaging = None
    if native_grid is not None:
        from .notebook_averaging import instrument_averaging

        document, averaging = instrument_averaging(document, native_grid)
    instrumentation = ("from hera_systematics_model.configuration import capture_imports\n"
        "from hera_systematics_model.production import write_json_exclusive\n"
        "from pathlib import Path\n"
        f"write_json_exclusive(Path({str(output_dir / 'import-runtime.json')!r}), capture_imports({SPECTRAL_MODULES!r}))\n")
    document.cells.append(nbformat.v4.new_code_cell(instrumentation))
    instrumented = output_dir / "instrumented.ipynb"
    with instrumented.open("x") as stream:
        nbformat.write(document, stream)
    kernel_root = output_dir / "jupyter"
    kernel = kernel_root / "kernels" / "hsm-worker"
    kernel.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(kernel / "kernel.json", {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Spectrum worker", "language": "python"})
    write_json_exclusive(output_dir / "execution.json", {"parameters": parameters,
        "notebook": file_identity(notebook), "instrumented_notebook": file_identity(instrumented),
        "single_baseline": file_identity(single_baseline), "python": sys.executable,
        "native_averaging": averaging})
    previous = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + previous if previous else "")
    try:
        papermill.execute_notebook(str(instrumented), str(output_dir / "spectrum.ipynb"), parameters=parameters,
            kernel_name="hsm-worker", cwd=str(output_dir), progress_bar=False, log_output=True)
    finally:
        if previous is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one captured spectral notebook")
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--single-baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--native-grid", help="Captured shared or retained native averaging grid JSON")
    args = parser.parse_args(argv)
    execute_spectrum(args.notebook, json.loads(Path(args.configuration).read_text()), args.single_baseline,
                     args.output_dir, native_grid=args.native_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
