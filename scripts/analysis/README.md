# Analysis scripts

The chain:

```text
baselines_merged.pspec.h5  (one per branch)
    |
    v
build_aligned_samples.py  ->  <label>.aligned.spwNN.npz  (+ provenance json)
    |
    +-> run_pca_aligned.py  ->  <label>.pca.spwNN.npz  (+ summary json)
    |       |
    |       +-> plot_aligned_modes.py  ->  figures
    |
    +-> holdout.py  ->  <label>.holdout.json
```

`submit_aligned_pca.sh` runs all of this on the cluster. `cylindrical.py`
and `residuals.py` are shared modules. `check_residuals.py` tests
`residuals.py` — just run it, it prints PASS/FAIL lines.

## build_aligned_samples.py

The merged pspec files do not have the same set of times for every baseline
pair. The upstream pipeline averaged the data into ~270 s windows on a
common grid, but flagging shifts the average time inside each window by up
to half a window, differently for each baseline. So rows cannot be matched
by row number. This script assigns every row to its window
(`floor((t - anchor) / window_length)`) and builds matrices where row i
really is the same time window for every baseline pair.

A window is kept when at least `--quorum` (default 0.95) of the baseline
pairs have it. Pairs that miss a kept window simply add no weight there.
`--quorum 1.0` means every pair must have the window.

Baseline pairs of the same length are then averaged together with $1/P_N^2$
weights, and the delay axis is folded (the $+\tau$ and $-\tau$ bins
averaged). The noise of each averaged cell comes out as $1/\sqrt{\sum w}$
and is saved as well.

One npz per spectral window, with:

| key | shape | what it is |
|---|---|---|
| `matrix` | (Ntimes, Ngroups*Ndly) | the aligned, averaged, folded power |
| `pn_eff` | same | noise per cell (inf where a cell had no data) |
| `valid` | same, bool | true where both fold halves had data |
| `cube_shape` | (3,) | (Ntimes, Ngroups, Ndly), for reshaping |
| `time_grid`, `lst_grid` | (Ntimes,) | window centres, JD and LST (radians) |
| `window_counts` | (Ntimes,) | how many baseline pairs contributed per window |
| `blp_lens` | (Ngroups,) | baseline length of each group [m] |
| `dlys` | (Ndly,) | folded delays [s] |
| `kperps`, `kparas` | (Ngroups,), (Ndly,) | $k_\perp$ and $k_\parallel$ [h/Mpc]; kperps is NaN if the input had no cosmology attached |
| `group_reps` | (Ngroups,) | one representative baseline-pair id per group |

The provenance json next to it records the input file, all settings, drop
counts, the hera_pspec version, the exact command, and the runtime.

## run_pca_aligned.py

PCA per spectral window: subtract the mean over times, then numpy SVD.

Options:

- `--mask`: cut the plane down before the PCA. `min-delay` drops delay bins
  below `--min-delay-ns`. `above-wedge` keeps only cells above the horizon
  line plus `--wedge-buffer-ns`. The saved components are still on the full
  grid (zero outside the mask), so the coordinate arrays still apply.
- `--whiten pn-median`: divide each cell by its median noise before the PCA.
  Without this the leading components just track the noisiest baseline
  groups — the longest baselines have few redundant partners, so their
  noise is orders of magnitude above the rest. The scale used is saved as
  `feature_scale` so you can get back to power units.
- `--missing`: what to do with cells that never had data. `zero` (default)
  leaves them as zeros; `drop-features` takes them out of the fit.
- Contrast mode (`--subtract-dir` / `--subtract-label`): PCA of one branch
  minus another, matched in time.

Output npz per spectral window: `mean`, `components` (top 40),
`singular_values`, `explained_variance_ratio`, `scores`, `n_for_threshold`
(components needed for 99% of the variance, or whatever
`--variance-threshold` is), `feature_mask`, `feature_scale` (whitened runs
only), `valid`, and the coordinate arrays copied from the input. The
summary json lists, per spectral window, the matrix size, the settings
used, and the top-5 variance ratios.

## plot_aligned_modes.py

Draws the PCA output in $(k_\perp, k_\parallel)$: the mean map, the leading modes
(signed, diverging colour scale), a scree plot, the mode scores against
LST, and a map of how often each cell had data. The horizon line and a
delay buffer line are drawn on every panel.

If $k_\perp$ is missing from the input it is computed here: $k_\parallel/\tau$
is a constant for each spectral window, that constant fixes the redshift,
and the redshift gives $k_\perp$ from the baseline length. The redshifts are
written to `<label>.coords_report.json` so they can be checked.

## cylindrical.py

The shared geometry: redshift from the $k_\parallel/\tau$ constant, $k_\perp$
from baseline length, the horizon slope, and `wedge_mask()` for picking cells
above the horizon (with an optional buffer given as a delay). Uses the
astropy Planck15 cosmology. No plotting imports, so it is safe to use on
the cluster.

## residuals.py and check_residuals.py

The candidate ways of turning a (systematic, ideal) pair of power spectra
into a residual, each with an inverse:

- `linear`: $P_s - P_i$
- `signed_asinh`: $\mathrm{asinh}(P_s/s) - \mathrm{asinh}(P_i/s)$, where $s$
  defaults to the cell noise
- `log_ratio`: $\log(P_s + P_0) - \log(P_i + P_0)$ with a floor $P_0$; cells
  that are not positive after the floor come out NaN
- `noise_weighted`: $(P_s - P_i) / P_N$

Power can be negative in noise-dominated cells (these are cross powers), so
the transforms return NaN where they are not defined instead of making
something up. The inverses exist so reconstruction errors can be compared
in one common space (power units). Note that asinh does not keep shapes: a
contamination pattern gets multiplied by $1/\sqrt{s^2 + P^2}$ across the
plane.

`check_residuals.py` tests all of this on synthetic data with known
answers:

```bash
python check_residuals.py
```

## holdout.py

Held-out comparison of the residual choices. The data is split into blocks
of neighbouring LST — not randomly, because neighbouring windows are
correlated and a random split would cheat. For each block: fit the PCA on
the rest, project the held-out block, go back to power units, and compute a
noise-weighted squared error over the cells above the horizon that the fit
actually covered. If no ideal branch is given (`--ideal-dir`), the mean of
the training rows is used as the reference, and the output says so.

Output: `<label>.holdout.json` with the error against number of components,
and the best point, per spectral window and per residual choice.

## submit_aligned_pca.sh

The Slurm driver. For each branch it submits one short high-memory job to
build the samples (the merged file is about 31 GB in memory) and one small
job that runs every PCA variant plus `holdout.py` and the branch
difference. BLAS threads are limited to the allocated cpus.

Settings via environment variables (defaults in brackets): `QUORUM` (0.95),
`MIN_DELAY_LIST` ("200 300"), `WEDGE_BUFFER_NS` (500), `SKIP_BUILD` (0; set
to 1 to skip rebuilding the samples), `BUILD_MEM` (64G), `ANALYSIS_MEM`
(8G), `ACCTG_FREQ` (5, so even short jobs report their real peak memory).
