"""Bind independently verified merged spectra to exact native sample exports."""

from pathlib import Path

from .artifacts import read_artifact
from .averaging_export import merge_memberships
from .input_verification import VerifiedInputs
from .spectrum_selection import baseline_pair_identity
from .window_membership import WindowMemberships


def merge_membership_files(spectrum, exports, output):
    """Require one exact export for every measured single-baseline merge input."""
    import h5py

    spectrum, output = Path(spectrum).resolve(), Path(output).resolve()
    merge_sidecar = spectrum.with_suffix(".merge.npz")
    exports = [Path(path).resolve(strict=True) for path in exports]
    if not exports or len(set(exports)) != len(exports):
        raise ValueError("unique native membership exports are required")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    report_path = output.with_suffix(".inputs.json")
    with VerifiedInputs(report_path) as inputs:
        for path in (merge_sidecar, merge_sidecar.with_suffix(".json")):
            inputs.identity(path)
        _, metadata = read_artifact(merge_sidecar, "spectral-merge")
        identity = inputs.identity(spectrum)
        if metadata.get("passed") is not True or metadata.get("output") != identity:
            raise ValueError("merged spectrum has no matching verified merge receipt")
        expected = {item["path"]: {key: item[key] for key in ("path", "bytes", "sha256")}
                    for item in metadata["inputs"]}
        memberships, consumed = [], set()
        for path in exports:
            for product in (path, path.with_suffix(".json")):
                inputs.identity(product)
            value = WindowMemberships.load(path)
            source = value.metadata["spectrum_source"]
            if source.get("path") not in expected or source != expected[source["path"]]:
                raise ValueError("native membership export belongs to a different source spectrum")
            if source["path"] in consumed:
                raise ValueError("duplicate source spectrum in native membership exports")
            consumed.add(source["path"])
            memberships.append(value)
        if consumed != set(expected):
            raise ValueError("native memberships do not cover all merged input spectra")
        with h5py.File(spectrum, "r") as handle:
            group = handle[metadata["group"]]
            baselines = [baseline_pair_identity(code) for code in group["blpair_array"][()]]
            result = merge_memberships(memberships, baselines, group["time_avg_array"][()], identity)
    result.metadata.update(merge_artifact={"path": str(merge_sidecar)},
                           membership_input_verification={"path": str(report_path)})
    result.save(output)
    return result
