"""Spectral inventory, verified merges and physical record conversion."""

import json
from pathlib import Path


def run_merge(args):
    from .spectrum_merge import merge_spectra
    from .spectrum_memberships import merge_membership_files

    inventory = json.loads(Path(args.inventory).read_text())
    if inventory.get("schema_version") != 1 or not inventory.get("inputs"):
        raise ValueError("unsupported or empty spectral inventory")
    entries = inventory["inputs"]
    if any(set(item) != {"spectrum", "memberships", "baseline_pair_code"} for item in entries):
        raise ValueError("spectral inputs require file, membership and physical baseline identities")
    report = merge_spectra([item["spectrum"] for item in entries], args.output,
                          [item["baseline_pair_code"] for item in entries])
    members = merge_membership_files(args.output, [item["memberships"] for item in entries], args.memberships_output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "counts": report["counts"],
                      "native_rows": len(members.baseline_ids), "passed": True}))
    return 0


def run_groups(args):
    from .production import write_json_exclusive
    from .spectrum_io import reference_groups

    result = reference_groups(args.spectrum, tolerance_m=args.tolerance_m)
    write_json_exclusive(Path(args.output), result)
    return 0


def run_records(args):
    from .notebook_averaging import load_native_grid
    from .records import WindowGrid
    from .spectrum_io import read_records

    native = load_native_grid(args.native_grid)
    grid = WindowGrid(native["anchor_jd"], native["window_seconds"])
    records = read_records(args.spectrum, args.spw, grid, json.loads(Path(args.grouping).read_text()),
                           args.role, polarization=args.polarization, memberships=args.memberships)
    records.save(args.output)
    return 0


def run_batch(args):
    import os
    from .spectral_batch import run_spectral_batch

    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("spectral batch execution requires a scheduler allocation")
    result = run_spectral_batch(args.inventory, args.output_dir, args.workers,
        int(os.environ["SLURM_CPUS_PER_TASK"]), int(os.environ["SLURM_MEM_PER_NODE"]))
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "passed": result["passed"],
                      "baselines": len(result["baselines"])}))
    return 0 if result["passed"] else 2


def add_commands(commands):
    parser = commands.add_parser("spectra", help="Inventory, merge and convert spectral products")
    actions = parser.add_subparsers(dest="spectral_command", required=True)
    batch = actions.add_parser("batch", help="Run a bounded baseline batch with explicit verified reuse")
    batch.add_argument("--inventory", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    batch.set_defaults(function=run_batch)
    merge = actions.add_parser("merge", help="Merge a complete baseline inventory and native exports")
    for name in ("inventory", "output", "memberships-output"):
        merge.add_argument("--" + name, required=True)
    merge.set_defaults(function=run_merge)
    groups = actions.add_parser("groups", help="Measure length groups from a corrupted reference")
    groups.add_argument("--spectrum", required=True)
    groups.add_argument("--output", required=True)
    groups.add_argument("--tolerance-m", type=float, default=1.)
    groups.set_defaults(function=run_groups)
    records = actions.add_parser("records", help="Export one spectral window with exact physical identities")
    for name in ("spectrum", "native-grid", "grouping", "memberships", "output"):
        records.add_argument("--" + name, required=True)
    records.add_argument("--role", choices=["corrupted", "ideal"], required=True)
    records.add_argument("--spw", type=int, choices=range(14), required=True)
    records.add_argument("--polarization", default="pI")
    records.set_defaults(function=run_records)
