# Inventory scripts

The read-only reconnaissance that produced `manifests/h6c_idr2_inventory.json`
(+ `.csv`). Nothing here mutates the surveyed storage.

| File | Role |
|---|---|
| `inventory_commands.sh` | The exact shell commands used to enumerate products on NRAO storage. File counts are `-maxdepth 1` and there is exactly one `du -sh` per product family, deliberately avoiding heavy recursive walks on Lustre. Rerunning these and comparing against the manifest is the reproducibility check. |
| `extract_metadata.py` | Read-only metadata extraction, run remotely by piping the script to the cluster Python (`ssh … 'python -' < extract_metadata.py`). Opens uvh5 / pspec headers with `h5py` (no bulk data reads) and prints one JSON object per file (JSON Lines) to stdout. |
| `build_manifest.py` | Assembles the manifest from the JSON Lines metadata dump plus the counts and sizes recorded by the shell commands. Writes `h6c_idr2_inventory.json` and the flattened `h6c_idr2_inventory.csv` (output path as the first argument, defaulting next to the script). |

## Regenerating

1. Run the enumeration in `inventory_commands.sh` (requires SSH access to the
   NRAO hosts; every command is non-mutating).
2. Run `extract_metadata.py` remotely against the cluster Python environment
   and capture its JSON Lines output.
3. Run `build_manifest.py <output.json>`; the CSV is written alongside.

The regenerated manifest records its own `inventory_date`; the field schema is
described in `manifests/README.md` and inside the JSON's `_schema` block.
