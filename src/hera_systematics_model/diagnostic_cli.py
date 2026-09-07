"""File-backed diagnostic, stability and plotting commands."""

import json
from pathlib import Path

import numpy as np

from .artifacts import read_artifact, write_artifact
from .configuration import capture_runtime, file_identity
from .evaluation import Evaluation
from .samples import PairedSamples
from .views import analysis_view


def input_identities(paths):
    return [file_identity(item) for path in paths for item in (Path(path), Path(path).with_suffix(".json"))]


def run_diagnostics(args):
    from .diagnostics import residual_diagnostics

    arrays, metadata = residual_diagnostics(PairedSamples.load(args.samples), Evaluation.load(args.fit),
        Evaluation.load(args.evaluation), args.surrogates, args.seed)
    metadata.update(runtime=capture_runtime(), inputs=input_identities([args.samples, args.fit, args.evaluation]),
                    complete=metadata["evaluation_complete"])
    write_artifact(args.output, "diagnostics", arrays, metadata)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": metadata["complete"]}))
    return 0 if metadata["complete"] else 2


def run_stability(args):
    from .stability import bootstrap_stability

    samples = PairedSamples.load(args.samples)
    fit = Evaluation.load(args.fit)
    arrays, _, identity = analysis_view(samples)
    if (fit.metadata.get("purpose") != "descriptive_fit" or not fit.metadata.get("complete")
            or fit.metadata.get("identity") != identity
            or not np.array_equal(fit.arrays["window_ids"], samples.window_ids)):
        raise ValueError("stability inputs must match a complete descriptive fit")
    output, metadata = bootstrap_stability(arrays, samples.window_ids, fit.metadata["selected"],
        n_replicates=args.replicates, block_length=args.block_length, seed=args.seed)
    output["window_ids"] = samples.window_ids.copy()
    metadata.update(identity=identity, purpose="block_stability", runtime=capture_runtime(),
                    inputs=input_identities([args.samples, args.fit]))
    write_artifact(args.output, "diagnostics", output, metadata)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": metadata["complete"]}))
    return 0 if metadata["complete"] else 2


def run_plot(args):
    from .plotting import plot_diagnostics
    from .cross_plotting import plot_cross_spw

    arrays, metadata = read_artifact(args.artifact, "diagnostics")
    renderers = {"residual_diagnostics": plot_diagnostics, "cross_spw_diagnostics": plot_cross_spw}
    if metadata.get("purpose") not in renderers:
        raise ValueError("a supported diagnostic artifact is required")
    products = renderers[metadata["purpose"]](arrays, metadata, args.output_dir)
    print(json.dumps({"figures": len(products), "output_dir": str(Path(args.output_dir).resolve())}))
    return 0


def run_cross_spw(args):
    from .cross_spw import cross_spw_diagnostics

    arrays, metadata = cross_spw_diagnostics(PairedSamples.load(args.left_samples),
        PairedSamples.load(args.right_samples), Evaluation.load(args.left_fit), Evaluation.load(args.right_fit))
    metadata.update(runtime=capture_runtime(), inputs=input_identities(
        [args.left_samples, args.right_samples, args.left_fit, args.right_fit]))
    write_artifact(args.output, "diagnostics", arrays, metadata)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": metadata["complete"],
                      "mode_similarity_available": metadata["mode_similarity"]["available"]}))
    return 0


def add_commands(commands):
    cross = commands.add_parser("cross-spw", help="Compare saved modes on matched physical coordinates")
    for name in ("left-samples", "right-samples", "left-fit", "right-fit", "output"):
        cross.add_argument("--" + name, required=True)
    cross.set_defaults(function=run_cross_spw)
    diagnostic = commands.add_parser("diagnostics", help="Measure residual and mode diagnostics")
    diagnostic.add_argument("--samples", required=True)
    diagnostic.add_argument("--fit", required=True)
    diagnostic.add_argument("--evaluation", required=True)
    diagnostic.add_argument("--output", required=True)
    diagnostic.add_argument("--surrogates", type=int, default=1000)
    diagnostic.add_argument("--seed", type=int, default=0)
    diagnostic.set_defaults(function=run_diagnostics)
    stability = commands.add_parser("stability", help="Refit selected models with physical block resampling")
    stability.add_argument("--samples", required=True)
    stability.add_argument("--fit", required=True)
    stability.add_argument("--output", required=True)
    stability.add_argument("--replicates", type=int, default=500)
    stability.add_argument("--block-length", type=int, choices=(8, 12, 16), default=12)
    stability.add_argument("--seed", type=int, default=0)
    stability.set_defaults(function=run_stability)
    plot = commands.add_parser("plot", help="Render standalone residual diagnostic figures")
    plot.add_argument("artifact")
    plot.add_argument("--output-dir", required=True)
    plot.set_defaults(function=run_plot)
