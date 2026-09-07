"""Command-line interface for residual analysis."""

import argparse
import json
from pathlib import Path

from . import __version__


def run_analysis(args):
    from .configuration import AnalysisConfig, capture_runtime, digest_json, file_identity
    from .evaluation import Evaluation, evaluate_guard_sensitivity, evaluate_nested
    from .fitting import select_final_fit
    from .samples import PairedSamples
    from .views import analysis_view

    samples = PairedSamples.load(args.samples)
    config = AnalysisConfig.load(args.config) if args.config else AnalysisConfig()
    arrays, shape, identity = analysis_view(samples, config.group, config.delay)
    selection_source = getattr(args, "selection_from", None)
    if selection_source is not None:
        primary = Evaluation.load(selection_source)
        primary_config = AnalysisConfig(**primary.metadata["configuration"]).as_dict()
        compared_config = config.as_dict()
        primary_config.pop("guard")
        compared_config.pop("guard")
        if (primary_config != compared_config or primary.metadata["identity"] != identity
                or primary.metadata["input"] != file_identity(args.samples)
                or primary.metadata["input_metadata"] != file_identity(Path(args.samples).with_suffix(".json"))):
            raise ValueError("frozen selection requires identical samples, coordinates and configuration except guard")
        result = evaluate_guard_sensitivity(primary, arrays, samples.window_ids, config.guard,
                                            training_filter=config.training_filter(samples))
        result.metadata["selection_input"] = file_identity(selection_source)
        result.metadata["selection_input_metadata"] = file_identity(Path(selection_source).with_suffix(".json"))
    else:
        function = evaluate_nested if args.command == "evaluate" else select_final_fit
        result = function(arrays, samples.window_ids, shape, config.candidates(), guard=config.guard,
                          axis="group" if config.delay is not None else "delay", training_filter=config.training_filter(samples))
    result.metadata.update(identity=identity, configuration=config.as_dict(),
        configuration_digest=digest_json(config.as_dict()), runtime=capture_runtime(),
        input=file_identity(args.samples), input_metadata=file_identity(Path(args.samples).with_suffix(".json")))
    result.save(args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": result.metadata["complete"]}))
    return 0 if result.metadata["complete"] else 2


def run_pair(args):
    from .alignment import build_paired
    from .configuration import capture_runtime, file_identity
    from .records import SpectrumRecords

    corrupted, ideal = SpectrumRecords.load(args.corrupted), SpectrumRecords.load(args.ideal)
    samples = build_paired(corrupted, ideal)
    samples.metadata.update(runtime=capture_runtime(), record_inputs=[file_identity(p)
        for name in (args.corrupted, args.ideal) for p in (Path(name), Path(name).with_suffix(".json"))])
    samples.save(args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "shape": list(samples.corrupted.shape)}))
    return 0


def run_replay(args):
    from .configuration import AnalysisConfig, file_identity
    from .evaluation import Evaluation
    from .evaluation_replay import replay_evaluation
    from .production import write_json_exclusive
    from .samples import PairedSamples
    from .views import analysis_view

    samples, evaluation = PairedSamples.load(args.samples), Evaluation.load(args.evaluation)
    config = AnalysisConfig(**evaluation.metadata["configuration"])
    arrays, _, identity = analysis_view(samples, config.group, config.delay)
    if (identity != evaluation.metadata["identity"] or file_identity(args.samples) != evaluation.metadata["input"]
            or file_identity(Path(args.samples).with_suffix(".json")) != evaluation.metadata["input_metadata"]):
        raise ValueError("replay source samples or physical feature identities differ")
    result = replay_evaluation(arrays, samples.window_ids, evaluation, config.training_filter(samples))
    result["inputs"] = [file_identity(p) for name in (args.samples, args.evaluation)
                        for p in (Path(name), Path(name).with_suffix(".json"))]
    write_json_exclusive(Path(args.output), result)
    print(json.dumps({"passed": result["passed"], "evaluation_complete": result["evaluation_complete"]}))
    return 0 if result["evaluation_complete"] else 2


def run_verify(args):
    from .artifacts import read_artifact
    from .evaluation import Evaluation
    from .model_io import load_model
    from .records import SpectrumRecords
    from .samples import PairedSamples
    from .spectrum_merge import verify_merge_receipt
    from .window_membership import WindowMemberships

    path = Path(args.artifact)
    kind = json.loads(path.with_suffix(".json").read_text()).get("kind")
    loaders = {"paired-samples": PairedSamples.load, "spectrum-records": SpectrumRecords.load,
               "fitted-model": load_model, "evaluation": Evaluation.load,
               "diagnostics": lambda p: read_artifact(p, "diagnostics"), "window-memberships": WindowMemberships.load,
               "spectral-merge": verify_merge_receipt}
    if kind not in loaders:
        raise ValueError("unsupported artifact kind")
    product = loaders[kind](path)
    _, metadata = read_artifact(path, kind)
    complete = metadata.get("complete", True)
    print(json.dumps({"artifact": str(path.resolve()), "kind": kind, "verified": True, "complete": complete}))
    return 0 if complete else 2


def main(argv=None):
    """Run a command and return its process exit status."""
    parser = argparse.ArgumentParser(
        prog="hera-systematics", description="Cylindrical power-spectrum analysis"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    from .diagnostic_cli import add_commands
    add_commands(commands)
    from .production_cli import add_commands as add_production
    add_production(commands)
    from .visibility_cli import add_commands as add_visibility
    add_visibility(commands)
    from .spectrum_cli import add_commands as add_spectra
    add_spectra(commands)
    replay = commands.add_parser("replay", help="Reproduce saved inference and losses without refitting modes")
    for name in ("samples", "evaluation", "output"):
        replay.add_argument("--" + name, required=True)
    replay.set_defaults(function=run_replay)
    pair = commands.add_parser("pair", help="Join and fold two spectrum-record artifacts")
    pair.add_argument("--corrupted", required=True)
    pair.add_argument("--ideal", required=True)
    pair.add_argument("--output", required=True)
    pair.set_defaults(function=run_pair)
    for name in ("evaluate", "fit"):
        analysis = commands.add_parser(name, help="Run nested evaluation" if name == "evaluate" else "Select a descriptive fit")
        analysis.add_argument("--samples", required=True)
        analysis.add_argument("--config")
        analysis.add_argument("--output", required=True)
        if name == "evaluate":
            analysis.add_argument("--selection-from", help="Freeze outer choices from a primary guard-12 evaluation")
        analysis.set_defaults(function=run_analysis)
    verify = commands.add_parser("verify", help="Verify artifact structure, state and hashes")
    verify.add_argument("artifact")
    verify.set_defaults(function=run_verify)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.function(args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"hera-systematics: {error}\n")
