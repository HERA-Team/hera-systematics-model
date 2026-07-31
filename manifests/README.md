# Manifests

Machine-readable inventory of the H6C IDR2 simulation data products on NRAO
storage, as surveyed read-only on the `inventory_date` recorded inside the
file. Produced by `scripts/inventory/` (see its README for regeneration).

## h6c_idr2_inventory.json

Top-level fields:

| Field | Meaning |
|---|---|
| `_schema` | Self-description of the per-product fields (kept inside the file so the manifest documents itself) |
| `inventory_date` | Survey date (ISO) |
| `method` | How the survey was performed |
| `base_path_release` | Release-snapshot root on Lustre |
| `base_path_release_storage_target` | The NFS storage target behind the release symlinks |
| `base_path_working` | Pipeline working-area root |
| `total_sizes` | `du -sh` per product family |
| `environments` | The pipeline and analysis Python environments observed |
| `data_facts` | Short, verifiable file-level statements (counts, completeness, storage placement) |
| `products` | One entry per product family — see `_schema` |

Each `products[]` entry carries a stable `product_id`, a `role`
(e.g. `ideal_sim`, `mock_component`, `sky_model`, `lstbinned_mock`,
`upstream_pspec`, `deprecated`), the absolute `path`, `file_format`,
`n_files` (counted at `-maxdepth 1`; `null` = not counted),
`filename_convention`, `sky_components`, `corruptions`, `generator`
(software provenance read from file histories), and `sample_metadata` —
header fields read from one representative file (dimensions, frequency range,
units, and similar).

Deprecated products are retained in the manifest with `role: "deprecated"`
rather than removed, so supersession stays visible.

## h6c_idr2_inventory.csv

A flattened one-row-per-product view of the JSON (identifier, role, path,
format, counts, sky components, corruptions, generator) for spreadsheet use.
The JSON is authoritative; the CSV omits `sample_metadata`.
