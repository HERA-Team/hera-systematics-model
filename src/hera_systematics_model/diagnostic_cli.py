"""File-backed diagnostic, stability and plotting commands."""

import json
from pathlib import Path

import numpy as np

from .artifacts import read_artifact, write_artifact
from .configuration import AnalysisConfig, capture_runtime, file_identity
from .evaluation import Evaluation
from .samples import PairedSamples
from .views import analysis_view


def input_identities(paths):
    return [file_identity(item) for path in paths for item in (Path(path), Path(path).with_suffix(".json"))]


def run_diagnostics(args):
    from .diagnostics import residual_diagnostics

    samples = PairedSamples.load(args.samples)
    fit, evaluation = Evaluation.load(args.fit), Evaluation.load(args.evaluation)
    for artifact in (fit, evaluation):
        if (artifact.metadata.get("input") != file_identity(args.samples)
                or artifact.metadata.get("input_metadata") != file_identity(Path(args.samples).with_suffix(".json"))):
            raise ValueError("diagnostic source samples differ from fitted or evaluated inputs")
    arrays, metadata = residual_diagnostics(samples, fit, evaluation, args.surrogates, args.seed)
    metadata.update(runtime=capture_runtime(), inputs=input_identities([args.samples, args.fit, args.evaluation]),
                    complete=metadata["evaluation_complete"])
    write_artifact(args.output, "diagnostics", arrays, metadata)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": metadata["complete"]}))
    return 0 if metadata["complete"] else 2


def run_stability(args):
    from .stability import bootstrap_stability

    samples = PairedSamples.load(args.samples)
    fit = Evaluation.load(args.fit)
    config = AnalysisConfig(**fit.metadata["configuration"])
    arrays, _, identity = analysis_view(samples, config.group, config.delay)
    if (fit.metadata.get("purpose") != "descriptive_fit" or not fit.metadata.get("complete")
            or fit.metadata.get("input") != file_identity(args.samples)
            or fit.metadata.get("input_metadata") != file_identity(Path(args.samples).with_suffix(".json"))
            or fit.metadata.get("identity") != identity
            or not np.array_equal(fit.arrays["window_ids"], samples.window_ids)):
        raise ValueError("stability inputs must match a complete descriptive fit")
    output, metadata = bootstrap_stability(arrays, samples.window_ids, fit.metadata["selected"],
        n_replicates=args.replicates, block_length=args.block_length, seed=args.seed,
        training_filter=config.training_filter(samples))
    output["window_ids"] = samples.window_ids.copy()
    metadata.update(identity=identity, configuration=config.as_dict(), purpose="block_stability", runtime=capture_runtime(),
                    inputs=input_identities([args.samples, args.fit]))
    write_artifact(args.output, "diagnostics", output, metadata)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": metadata["complete"]}))
    return 0 if metadata["complete"] else 2


def run_plot(args):
    from .plotting import plot_diagnostics
    from .cross_plotting import plot_cross_spw
    from .stability_plotting import plot_stability

    arrays, metadata = read_artifact(args.artifact, "diagnostics")
    renderers = {"residual_diagnostics": plot_diagnostics, "cross_spw_diagnostics": plot_cross_spw,
                 "block_stability": plot_stability}
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


def run_summary(args):
    from .evidence import evaluation_evidence
    from .production import write_json_exclusive

    result = evaluation_evidence(Evaluation.load(args.evaluation), Evaluation.load(args.fit))
    result["inputs"] = input_identities([args.evaluation, args.fit])
    result["runtime"] = capture_runtime()
    write_json_exclusive(Path(args.output), result)
    complete = result["evaluation_complete"] and result["descriptive_fit"]["complete"]
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": complete}))
    return 0 if complete else 2


def add_commands(commands):
    summary = commands.add_parser("summary", help="Export numerical evidence across physical-time folds")
    for name in ("fit", "evaluation", "output"):
        summary.add_argument("--" + name, required=True)
    summary.set_defaults(function=run_summary)
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
