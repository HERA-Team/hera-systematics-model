#!/usr/bin/env bash
# Submit one immutable task through aggregate resource and storage checks.
# Pass --run, --task, --python and --package-source; use --help for details.
set -euo pipefail
HSM_SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$HSM_SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${HSM_PYTHON:-python3}" -m hera_systematics_model production submit "$@"
