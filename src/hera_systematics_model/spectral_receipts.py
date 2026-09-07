"""Per-baseline spectral acceptance and exact-identity reuse records."""

import json
from pathlib import Path
import re

import numpy as np

from .configuration import digest_json, file_identity
from .production import write_json_exclusive
from .spectrum_layout import inspect_spectrum
from .spectrum_selection import baseline_pair_identity
from .window_membership import WindowMemberships
from .worker import verify_product


def spectral_products(directory, baseline_pair_code, native_times, native_width):
    """Read all required payloads and reconcile their baseline and native rows."""
    import h5py

    directory = Path(directory).resolve()
    baseline = baseline_pair_identity(baseline_pair_code)
    native = np.asarray(native_times, float)
    if (type(native_width) is not int or native_width < 1 or native.ndim != 1
            or not len(native) or len(native) % native_width or not np.isfinite(native).all()
            or np.any(np.diff(native) <= 0)):
        raise ValueError("complete ordered native windows required")
    products, counts = [], {}
    execution = json.loads((directory / "execution.json").read_text())
    if "label_metadata" in execution:
        if execution["label_metadata"] != {"policy": "collapse_identical_labels_after_final_time_average",
                                           "report": "label-metadata.json"}:
            raise ValueError("unknown spectral label normalization")
        products.append(verify_product(directory, {"path": "label-metadata.json", "kind": "file"}))
        labels = json.loads((directory / "label-metadata.json").read_text())
        if (labels.get("schema_version") != 1 or labels.get("passed") is not True
                or labels.get("numerical_payload_modified") is not False):
            raise ValueError("spectral label normalization report failed")
    for filename, name in (("spectrum.pspec.h5", "interleave_averaged"),
                           ("spectrum.tavg.pspec.h5", "time_and_interleave_averaged")):
        group_name = "stokespol/" + name
        products.append(verify_product(directory, {"path": filename, "kind": "hdf5", "required_paths": [group_name]}))
        with h5py.File(directory / filename) as handle:
            group = handle[group_name]
            layout = inspect_spectrum(group)
            if not np.all(group["blpair_array"][()] == baseline_pair_code):
                raise ValueError("spectrum contains an unexpected physical baseline")
            counts[name] = layout["counts"]
            counts[name]["spectral_windows"] = []
            for spw in range(14):
                power, noise = group[f"data_spw{spw}"][()], group[f"stats_P_N_{spw}"][()]
                counts[name]["spectral_windows"].append({"spw": spw,
                    "polarization_codes": group.attrs["polpair_array"].tolist(),
                    "finite_power_per_polarization": np.isfinite(power).sum(axis=(0, 1)).tolist(),
                    "finite_positive_noise_per_polarization": (np.isfinite(noise) & (noise.real > 0)
                        & (noise.imag == 0)).sum(axis=(0, 1)).tolist()})
            if name == "interleave_averaged":
                centroids = group["time_avg_array"][()]
    membership = WindowMemberships.load(directory / "window-memberships.npz")
    if membership.metadata["spectrum_source"] != file_identity(directory / "spectrum.pspec.h5"):
        raise ValueError("membership belongs to a different spectral payload")
    ids, members = membership.lookup(np.repeat(baseline, len(centroids)), centroids)
    if (not np.array_equal(membership.native_time_jd, native)
            or not np.array_equal(ids, np.arange(len(native) // native_width))
            or not np.array_equal(members, np.arange(len(native)).reshape(-1, native_width))):
        raise ValueError("spectrum does not cover the exact requested native windows")
    for filename, kind in (("window-memberships.npz", "npz"), ("window-memberships.json", "file"),
                           ("execution.json", "file"), ("import-runtime.json", "file"),
                           ("instrumented.ipynb", "file"), ("spectrum.ipynb", "file")):
        products.append(verify_product(directory, {"path": filename, "kind": kind}))
    notebook = json.loads((directory / "spectrum.ipynb").read_text())
    if any(output.get("output_type") == "error" for cell in notebook["cells"] for output in cell.get("outputs", [])):
        raise ValueError("executed notebook contains an error output")
    return products, counts


def validate_entry_identity(directory, identity, native_times, input_context=None):
    """Check recorded roles against consumed files and notebook execution state."""
    required = {"code_commit", "source_digest", "runtime_digest", "configuration", "inputs", "baseline_pair_code", "native_grid_digest"}
    if (set(identity) != required or not re.match(r"[0-9a-f]{40}\Z", identity["code_commit"])
            or not re.match(r"[0-9a-f]{64}\Z", identity["source_digest"])
            or not re.match(r"[0-9a-f]{64}\Z", identity["runtime_digest"])
            or identity["native_grid_digest"] != digest_json(np.asarray(native_times).tolist())
            or not isinstance(identity["configuration"], dict) or not isinstance(identity["inputs"], dict)
            or not {"notebook", "single_baseline", "auto", "native_grid", "beam", "fringe_rate_cache"} <= set(identity["inputs"])):
        raise ValueError("spectral entry identity is incomplete or has a different native grid")
    for item in identity["inputs"].values():
        actual = file_identity(item["path"]) if input_context is None else input_context.identity(item["path"])
        if actual != item:
            raise ValueError("spectral entry consumed input changed")
    directory = Path(directory).resolve()
    execution = json.loads((directory / "execution.json").read_text())
    parameters = {**identity["configuration"], "SINGLE_BL_FILE": identity["inputs"]["single_baseline"]["path"],
        "OUT_PSPEC_FILE": str(directory / "spectrum.pspec.h5"),
        "OUT_TAVG_PSPEC_FILE": str(directory / "spectrum.tavg.pspec.h5")}
    runtime = json.loads((directory / "import-runtime.json").read_text())["runtime"]
    if (execution["parameters"] != parameters or execution["notebook"] != identity["inputs"]["notebook"]
            or execution["single_baseline"] != identity["inputs"]["single_baseline"]
            or runtime["source_digest"] != identity["source_digest"] or digest_json(runtime) != identity["runtime_digest"]):
        raise ValueError("spectral execution used different inputs, configuration or code")


def accept_spectral_entry(directory, identity, native_times, native_width, input_context=None):
    """Bind verified outputs to complete code, configuration and consumed inputs."""
    validate_entry_identity(directory, identity, native_times, input_context)
    products, counts = spectral_products(directory, identity["baseline_pair_code"], native_times, native_width)
    report = {"schema_version": 1, "identity": identity, "identity_digest": digest_json(identity),
              "passed": True, "products": products, "counts": counts}
    write_json_exclusive(Path(directory) / "spectrum-verification.json", report)
    return report


def reuse_spectral_entry(directory, identity, native_times, native_width, input_context=None):
    """Return only unchanged verified products from exactly the same inputs."""
    path = Path(directory) / "spectrum-verification.json"
    report = json.loads(path.read_text())
    if (report.get("schema_version") != 1 or report.get("passed") is not True
            or report.get("identity") != identity or report.get("identity_digest") != digest_json(identity)):
        raise ValueError("spectral reuse has different code, inputs or configuration")
    validate_entry_identity(directory, identity, native_times, input_context)
    products, counts = spectral_products(directory, identity["baseline_pair_code"], native_times, native_width)
    if report["products"] != products or report["counts"] != counts:
        raise ValueError("spectral reuse products changed after acceptance")
    return report
