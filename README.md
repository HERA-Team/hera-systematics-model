# hera-systematics-model

Code for building a PCA model of systematic contamination in HERA cylindrical
power spectra P(kperp, kpara). The idea is to learn the shapes that
systematics leave in the power spectrum from simulations, so they can be
included as nuisance terms in cosmological inference instead of being masked
or ignored.

What is where:

- `manifests/` - list of the H6C IDR2 simulation products on NRAO disk (json + csv)
- `scripts/inventory/` - the scripts that made that list
- `scripts/analysis/` - the analysis pipeline: align, average, PCA, plots, held-out tests
- `results/` - small summary files from finished runs
- `Progress.md` - log of completed work

Big files (pspec containers, sample matrices, per-mode npz files) stay on
NRAO storage. Only code, manifests and small summaries are kept here.

You need python with numpy, h5py, hera_pspec, astropy and scipy. matplotlib
is only needed for the plotting script, which is normally run away from the
cluster on copied npz files. The exact package versions used by a run are
recorded in that run's provenance json.

To run the full chain on the cluster:

```bash
./submit_aligned_pca.sh              # build samples + all PCA variants + held-out tests
SKIP_BUILD=1 ./submit_aligned_pca.sh # redo the analysis only, keeping existing samples
```

To plot from the output npz files (works anywhere):

```bash
python plot_aligned_modes.py --pca-dir <dir> --label sum --outdir figures
```

See `scripts/analysis/README.md` for what each script does and what is in
the output files.
