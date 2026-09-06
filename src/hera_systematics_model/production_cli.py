"""Command interfaces for immutable definitions and accounted Slurm tasks."""

import json
from pathlib import Path

from .production import create_run, define_task, write_json_exclusive


def run_initialize(args):
    configuration = json.loads(Path(args.config).read_text())
    if not isinstance(configuration, dict):
        raise ValueError("production configuration must be an object")
    path = create_run(args.root, args.code_commit, configuration, args.input,
                      snapshots=[args.config, *args.snapshot], projected_additional_bytes=args.projected_bytes)
    print(json.dumps({"run": str(path)}))
    return 0


def run_define(args):
    task = json.loads(Path(args.definition).read_text())
    path = define_task(args.run, task)
    print(json.dumps({"task_definition": str(path)}))
    return 0


def run_submit(args):
    from .scheduler import submit_task

    result = submit_task(args.run, args.task, args.python, args.package_source, args.partition)
    print(json.dumps(result, allow_nan=False))
    return 0


def run_acceptance(args):
    from .scheduler import verify_task_acceptance

    result = verify_task_acceptance(args.run, args.task)
    if args.output:
        write_json_exclusive(args.output, result)
    print(json.dumps(result, allow_nan=False))
    return 0


def add_commands(commands):
    production = commands.add_parser("production", help="Define, submit and verify immutable compute tasks")
    operations = production.add_subparsers(dest="operation", required=True)
    initialize = operations.add_parser("init", help="Capture input identities and configuration")
    initialize.add_argument("--root", required=True)
    initialize.add_argument("--code-commit", required=True)
    initialize.add_argument("--config", required=True)
    initialize.add_argument("--input", action="append", required=True)
    initialize.add_argument("--snapshot", action="append", default=[])
    initialize.add_argument("--projected-bytes", type=int, required=True)
    initialize.set_defaults(function=run_initialize)
    define = operations.add_parser("define", help="Reserve an immutable task definition")
    define.add_argument("--run", required=True)
    define.add_argument("--definition", required=True)
    define.set_defaults(function=run_define)
    submit = operations.add_parser("submit", help="Submit within aggregate resource limits")
    submit.add_argument("--run", required=True)
    submit.add_argument("--task", required=True)
    submit.add_argument("--python", required=True)
    submit.add_argument("--package-source", required=True)
    submit.add_argument("--partition", default="hera")
    submit.set_defaults(function=run_submit)
    acceptance = operations.add_parser("verify", help="Require scheduler success and verified products")
    acceptance.add_argument("--run", required=True)
    acceptance.add_argument("--task", required=True)
    acceptance.add_argument("--output")
    acceptance.set_defaults(function=run_acceptance)
