"""Deterministic spectral batches with bounded concurrency and verified reuse."""

import json
from pathlib import Path
import re
import sys

from .batch_execution import run_bounded_commands, validate_batch
from .configuration import capture_imports, capture_runtime, digest_json
from .notebook import SPECTRAL_MODULES
from .input_verification import VerifiedInputs
from .notebook_averaging import load_native_grid
from .production import write_json_exclusive
from .spectral_receipts import accept_spectral_entry, reuse_spectral_entry
from .spectrum_selection import baseline_pair_identity


def spectral_inventory(value):
    required = {"schema_version", "code_commit", "notebook", "native_grid", "configuration", "baselines"}
    if (not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1
            or not re.match(r"[0-9a-f]{40}\Z", value["code_commit"])
            or not isinstance(value["configuration"], dict) or not value["baselines"]):
        raise ValueError("invalid spectral batch inventory")
    if not {"EFIELD_HEALPIX_BEAM_FILE", "FR_SPECTRA_FILE"} <= set(value["configuration"]):
        raise ValueError("explicit beam and fringe-rate inputs are required")
    codes = []
    for entry in value["baselines"]:
        if not {"baseline_pair_code", "file", "auto"} <= set(entry) or set(entry) - {"baseline_pair_code", "file", "auto", "reuse"}:
            raise ValueError("baseline inventory requires physical code, visibility and auto paths")
        code = entry["baseline_pair_code"]
        first, second = baseline_pair_identity(code).split(":")
        a, b = map(int, first.split("_"))
        if first != second or a == b:
            raise ValueError("single cross-baseline spectra are required")
        for path in (entry["file"], entry["auto"]):
            if not Path(path).is_absolute():
                raise ValueError("absolute visibility input paths required")
        expected_auto = Path(entry["file"]).parent / "zen.LST.baseline.0_0.sum.uvh5"
        if Path(entry["auto"]).resolve() != expected_auto.resolve():
            raise ValueError("auto path differs from the notebook's adjacent autocorrelation input")
        codes.append(code)
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate physical baseline in spectral inventory")
    return sorted(value["baselines"], key=lambda entry: entry["baseline_pair_code"])


def run_spectral_batch(inventory_path, directory, workers, allocated_cpus, allocated_memory_mib):
    """Run explicit baseline identities; accept each zero-exit product separately."""
    inventory_path, directory = Path(inventory_path).resolve(), Path(directory).resolve()
    inventory = json.loads(inventory_path.read_text())
    entries = spectral_inventory(inventory)
    reservations = validate_batch([{"name": "resource-check", "command": [sys.executable], "environment": {}}],
                                  workers, allocated_cpus, allocated_memory_mib)
    directory.mkdir(parents=True, exist_ok=False)
    # Scientific imports can expose bundled distribution metadata. Capture the
    # same import context as the executed notebook before constructing identities.
    imported = capture_imports(SPECTRAL_MODULES)
    runtime = imported["runtime"]
    write_json_exclusive(directory / "import-runtime.json", imported)
    write_json_exclusive(directory / "runtime.json", runtime)
    configuration_path = directory / "configuration.json"
    write_json_exclusive(configuration_path, inventory["configuration"])
    results, tasks, identities = {}, [], {}
    with VerifiedInputs(directory / "input-verification.json") as inputs:
        inputs.identity(inventory_path)
        shared = {"notebook": inputs.identity(inventory["notebook"]),
            "native_grid": inputs.identity(inventory["native_grid"]),
            "beam": inputs.identity(inventory["configuration"]["EFIELD_HEALPIX_BEAM_FILE"]),
            "fringe_rate_cache": inputs.identity(inventory["configuration"]["FR_SPECTRA_FILE"])}
        grid = load_native_grid(inventory["native_grid"])
        if grid["policy"] != "shared":
            raise ValueError("production spectral batches require the shared native grid")
        native, width = grid["native_time_jd"], grid["native_samples_per_window"]
        for entry in entries:
            code = entry["baseline_pair_code"]
            identity = {"code_commit": inventory["code_commit"], "source_digest": runtime["source_digest"],
                "runtime_digest": digest_json(runtime),
                "configuration": inventory["configuration"], "baseline_pair_code": code,
                "native_grid_digest": digest_json(native), "inputs": {**shared,
                    "single_baseline": inputs.identity(entry["file"]), "auto": inputs.identity(entry["auto"])}}
            identities[code] = identity
            if entry.get("reuse"):
                folder = Path(entry["reuse"]).resolve()
                report = reuse_spectral_entry(folder, identity, native, width, inputs)
                for product in report["products"]:
                    if inputs.identity(product["path"]) != {key: product[key] for key in ("path", "bytes", "sha256")}:
                        raise ValueError("reused product changed while registering batch inputs")
                results[code] = {"baseline_pair_code": code, "directory": str(folder), "reused": True,
                                 "passed": True, "receipt": inputs.identity(folder / "spectrum-verification.json")}
                continue
            tasks.append({"name": f"baseline-{code}", "command": [sys.executable, "-m", "hera_systematics_model.notebook",
                "--notebook", shared["notebook"]["path"], "--configuration", str(configuration_path),
                "--single-baseline", identity["inputs"]["single_baseline"]["path"],
                "--native-grid", shared["native_grid"]["path"], "--output-dir", "{output_dir}"],
                "environment": {"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
                                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}})
        write_json_exclusive(directory / "entry-identities.json", {str(code): identity for code, identity in identities.items()})
        def accept_completed(record, folder):
            code = int(record["name"].removeprefix("baseline-"))
            results[code] = {"baseline_pair_code": code, "directory": str(folder), "reused": False,
                             "passed": False, "command_status": record["status"], "exit_code": record["exit_code"]}
            if record["status"] == "exited_zero":
                try:
                    accepted = accept_spectral_entry(folder, identities[code], native, width, inputs)
                    for product in accepted["products"]:
                        if inputs.identity(product["path"]) != {key: product[key] for key in ("path", "bytes", "sha256")}:
                            raise ValueError("new product changed while registering batch inputs")
                    results[code].update(passed=True, receipt=inputs.identity(folder / "spectrum-verification.json"))
                except (ValueError, OSError, KeyError) as error:
                    results[code]["verification_error"] = str(error)
                    return {**record, "status": "verification_failed", "verification_error": str(error)}
            return record

        if tasks:
            commands = run_bounded_commands(tasks, directory / "commands", workers, allocated_cpus,
                                           allocated_memory_mib, on_completion=accept_completed)
            for record in commands["records"]:
                code = int(record["name"].removeprefix("baseline-"))
                if code not in results:
                    results[code] = {"baseline_pair_code": code, "passed": False, "reused": False,
                                     "command_status": record["status"], "exit_code": record["exit_code"]}
        if capture_runtime()["source_digest"] != runtime["source_digest"]:
            raise ValueError("spectral batch source changed during execution")
    report = {"schema_version": 1, "passed": all(record["passed"] for record in results.values()),
        "code_commit": inventory["code_commit"], "source_digest": runtime["source_digest"],
        "resources": reservations, "baselines": [results[entry["baseline_pair_code"]] for entry in entries]}
    write_json_exclusive(directory / "batch-verification.json", report)
    return report
