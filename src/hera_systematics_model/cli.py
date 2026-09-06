"""Command-line interface for residual analysis."""

import argparse
import json
from pathlib import Path

from . import __version__


def run_analysis(args):
    from .configuration import AnalysisConfig, capture_runtime, digest_json, file_identity
    from .evaluation import evaluate_nested
    from .fitting import select_final_fit
    from .samples import PairedSamples
    from .views import analysis_view

    samples = PairedSamples.load(args.samples)
    config = AnalysisConfig.load(args.config) if args.config else AnalysisConfig()
    arrays, shape, identity = analysis_view(samples, config.group, config.delay)
    function = evaluate_nested if args.command == "evaluate" else select_final_fit
    result = function(arrays, samples.window_ids, shape, config.candidates(), guard=config.guard,
                      axis="group" if config.delay is not None else "delay")
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


def run_verify(args):
    from .artifacts import read_artifact
    from .evaluation import Evaluation
    from .model_io import load_model
    from .records import SpectrumRecords
    from .samples import PairedSamples

    path = Path(args.artifact)
    kind = json.loads(path.with_suffix(".json").read_text()).get("kind")
    loaders = {"paired-samples": PairedSamples.load, "spectrum-records": SpectrumRecords.load,
               "fitted-model": load_model, "evaluation": Evaluation.load,
               "diagnostics": lambda p: read_artifact(p, "diagnostics")}
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
