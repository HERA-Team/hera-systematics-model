# Manifests

A list of the H6C IDR2 simulation products on NRAO disk, surveyed read-only
on the date in `inventory_date`. Built by `scripts/inventory/`.

`h6c_idr2_inventory.json` contains:

- `_schema` — describes the per-product fields (kept inside the file so it
  explains itself)
- `inventory_date`, `method` — when and how the survey was done
- `base_path_release`, `base_path_release_storage_target`,
  `base_path_working` — the three root paths involved
- `total_sizes` — `du -sh` per product family
- `environments` — the python environments found (pipeline and analysis)
- `data_facts` — short factual statements about counts and completeness
- `products` — one entry per product family

Each product entry has a stable `product_id`, a `role` (ideal_sim,
mock_component, sky_model, lstbinned_mock, upstream_pspec, deprecated, ...),
the `path`, `file_format`, `n_files` (counted one level deep; null means
not counted), `filename_convention`, `sky_components`, `corruptions`,
`generator` (software versions read from the file histories), and
`sample_metadata` — header values from one example file (sizes, frequency
range, units and so on).

Products that were replaced stay in the list with `role: "deprecated"`
instead of being deleted, so it stays clear what replaced what.

`h6c_idr2_inventory.csv` is the same list flattened to one row per product,
for spreadsheets. It leaves out `sample_metadata`; the json is the full
version.
