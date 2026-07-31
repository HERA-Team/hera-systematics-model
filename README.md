# hera-systematics-model

Analysis code and machine-readable manifests for building a low-rank model of
systematic contamination in HERA cylindrical power spectra
$P(k_\perp, k_\parallel)$, intended for use as an additive nuisance term in
cosmological inference.

## Layout

| Path | Contents |
|---|---|
| `manifests/` | Machine-readable inventory of the H6C IDR2 simulation products (JSON + CSV) |
| `scripts/inventory/` | The read-only commands and scripts that produced the inventory |
| `scripts/analysis/` | The aligned-sample / PCA / held-out-validation pipeline |
| `results/` | Compact summaries and provenance JSON of completed analysis runs |

Each directory carries its own `README.md` with file formats and usage. Bulk
inputs and outputs — UVPSpec containers, aligned sample matrices, per-mode
NPZ files — live on NRAO storage, not in the repository; `results/` holds
only the small JSON summaries needed to interpret a run.

## Environment

Cluster runs use a Python environment providing `numpy`, `h5py`,
`hera_pspec`, `astropy`, and `scipy`. `matplotlib` is required only by the
plotting script, which is normally run off-cluster against copied NPZ files.
The exact package versions used by a given run are recorded in that run's
provenance JSON (`hera_pspec_version`, command line, input checksums by
size/mtime).

## Typical flow

```bash
# on the cluster login host, from a copy of scripts/analysis/
./submit_aligned_pca.sh              # build aligned samples + run all PCA variants + held-out comparison
SKIP_BUILD=1 ./submit_aligned_pca.sh # re-run the analysis only, reusing existing samples

# anywhere with the output NPZ files
python plot_aligned_modes.py --pca-dir <dir> --label sum --outdir figures
```

See `scripts/analysis/README.md` for what each stage produces and the NPZ /
JSON schemas.
