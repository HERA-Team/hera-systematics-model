# Inventory scripts

These made `manifests/h6c_idr2_inventory.json` (and the csv). Everything is
read-only; nothing touches the surveyed data.

- `inventory_commands.sh` — the shell commands used to count and size the
  products on NRAO disk. Counts are one directory level deep and there is
  one `du -sh` per product family, to keep the load on Lustre low. Rerun
  these and compare with the manifest to check it still matches the disk.
- `extract_metadata.py` — opens file headers (h5py only, no data reads) and
  prints one json object per file. Meant to be piped to the cluster python
  over ssh: `ssh ... 'python -' < extract_metadata.py`.
- `build_manifest.py` — puts the metadata dump and the counts together into
  the json and csv. First argument is the output json path (defaults to the
  script directory).

The field meanings are in `manifests/README.md` and inside the json itself
(the `_schema` entry).
