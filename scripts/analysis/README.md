# Analysis pipeline

From merged UVPSpec containers to time-aligned sample matrices, PCA bases,
figures, and held-out validation numbers.

```text
baselines_merged.pspec.h5  (PSpecContainer, one per branch)
   │
   ├─ build_aligned_samples.py ──► <label>.aligned.spwNN.npz   (+ <label>.provenance.json)
   │        │
   │        ├─ run_pca_aligned.py ──► <label>.pca.spwNN.npz    (+ <label>.pca_summary.json)
   │        │        │
   │        │        └─ plot_aligned_modes.py ──► figures + <label>.coords_report.json
   │        │
   │        └─ holdout.py ──► <label>.holdout.json
   │
   └─ submit_aligned_pca.sh   (Slurm driver for all of the above)

cylindrical.py       shared coordinate / horizon-wedge geometry (no plotting deps)
residuals.py         residual-representation transforms and their inverses
check_residuals.py   self-checking exercise of residuals.py; run directly
```

## build_aligned_samples.py

Joins the rows of a merged UVPSpec on **physical time**, not per-baseline row
order. The upstream pipeline averaged integrations into windows on a shared
grid of window edges, but per-baseline flagging shifts each window's centroid
by up to half a window, so each centroid is assigned to a window index
`floor((t - anchor)/window)` with the window length inferred from per-baseline
time spacing (`--window-sec` to force it).

A window becomes a sample row when at least `--quorum` (default 0.95) of the
baseline-pairs measure it; baseline-pairs absent from a retained window simply
contribute no weight. `--quorum 1.0` reproduces a strict all-baseline
intersection. Rows sharing a window within one baseline-pair (centroids
straddling an edge) are discarded.

Baseline-pairs are grouped by redundant length and averaged incoherently with
$1/P_N^2$ weights; delay spectra are folded (positive delays, half-sum of the
$\pm\tau$ bins). The effective noise of each averaged cell is
$P_{N,\mathrm{eff}} = 1/\sqrt{\sum w}$, folded in quadrature.

Output NPZ per spectral window (`<label>.aligned.spwNN.npz`):

| Key | Shape | Meaning |
|---|---|---|
| `matrix` | (Ntimes, Ngroups·Ndly) | aligned, redundantly averaged, folded real power |
| `pn_eff` | same | effective noise per cell; `inf` where no weight |
| `valid` | same, bool | both fold halves carried weight |
| `cube_shape` | (3,) | (Ntimes, Ngroups, Ndly) for reshaping |
| `time_grid`, `lst_grid` | (Ntimes,) | window centres (JD) and LST (radians) |
| `window_counts` | (Ntimes,) | contributing baseline-pairs per window |
| `blp_lens` | (Ngroups,) | redundant-group baseline length [m] |
| `dlys` | (Ndly,) | folded positive delays [s] |
| `kperps`, `kparas` | (Ngroups,) / (Ndly,) | cosmological coordinates [h/Mpc]; `kperps` NaN when the container carried no cosmology |
| `group_reps` | (Ngroups,) | representative baseline-pair code per group |

The sidecar `<label>.provenance.json` records the input path/size/mtime,
group/spectrum/polarization selections, window length, quorum and per-window
contributor statistics, drop counts, weighting mode, `hera_pspec` version,
exact command line, and runtime.

## run_pca_aligned.py

Per-spectral-window PCA (mean-subtracted economy SVD) on the aligned matrices.

Modes and options:

- **Contrast mode** (`--subtract-dir/--subtract-label`): PCA of the row-wise
  difference of two branches matched on the time grid.
- `--missing {zero,drop-features}`: cells with no contributing baseline enter
  as exact zeros (historical behaviour) or are excluded as features and the
  components scattered back onto the full grid afterwards.
- `--mask {none,min-delay,above-wedge}`: restrict the PCA to part of the
  cylindrical plane. `min-delay` drops delay bins below `--min-delay-ns`;
  `above-wedge` keeps cells above the horizon plus `--wedge-buffer-ns`.
  Stored components remain full-grid (zero outside the mask), so
  `cube_shape` and the coordinate arrays stay valid.
- `--whiten pn-median`: divide each feature by its median effective noise
  before the SVD. Redundant-group noise spans orders of magnitude across
  baseline lengths, so an unwhitened PCA assigns leading components to the
  noisiest groups rather than to shared structure. Components are stored in
  whitened space (orthonormal there); the applied scale is stored as
  `feature_scale` for mapping back to power units.

Output NPZ per spectral window (`<label>.pca.spwNN.npz`): `mean`,
`components` (top 40, rows are modes), `singular_values`,
`explained_variance_ratio`, `scores` (samples × top modes),
`n_for_threshold` (components needed for `variance_threshold`, default 0.99),
`feature_mask`, `feature_scale` (whitened runs), `valid`, plus the coordinate
arrays copied through from the sample file. The `<label>.pca_summary.json`
lists, per spectral window, the sample/feature counts, missing policy, mask,
whitening, invalid-cell fraction, threshold component count, and top-5
explained-variance ratios.

## plot_aligned_modes.py

Renders the PCA output in physical $(k_\perp, k_\parallel)$ coordinates:
mean map (symmetric log), leading signed modes on a diverging scale, scree,
scores against LST, and (when a validity mask is present) the
fraction-of-rows-valid map. The horizon wedge and a delay-buffer line are
overlaid on every panel.

When `kperps` is absent it is derived: the ratio `kpara/delay` is a constant
per spectral window,

$$k_\parallel = \frac{2\pi\,\tau\,\nu_{21} H(z)}{c\,(1+z)^2},$$

so inverting that constant gives the window's redshift and then
$k_\perp = 2\pi |b|\,\nu/(c\,D_M(z))$. Derived redshifts are written to
`<label>.coords_report.json` for cross-checking against stored coordinates.

## cylindrical.py

Shared geometry, no plotting dependencies: redshift recovery from the
`kpara/delay` constant, $k_\perp$ from baseline length, the horizon-wedge
slope $H(z) D_M(z)/(c(1+z))$ (dimensionless in h/Mpc coordinates), and
`wedge_mask(...)` with an optional constant-$k_\parallel$ buffer set by a
delay floor. Cosmology: `astropy` Planck15.

## residuals.py / check_residuals.py

Candidate residual representations over a matched
(systematic, ideal) pair of power spectra, each with an exact inverse:

| Name | Form |
|---|---|
| `linear` | $P_s - P_i$ |
| `signed_asinh` | $\operatorname{asinh}(P_s/s) - \operatorname{asinh}(P_i/s)$, scale $s$ defaults to the per-cell noise |
| `log_ratio` | $\log(P_s + P_0) - \log(P_i + P_0)$; cells non-positive after the floor are NaN |
| `noise_weighted` | $(P_s - P_i)/P_N$ |

Cross-power spectra are real but not positive, so transforms mark cells they
cannot define with NaN rather than substituting values. The inverses exist so
reconstruction error can be compared across representations in one common
space (linear power). `signed_asinh` is a reweighting, not a
shape-preserving map: to first order the residual is
$\delta P/\sqrt{s^2 + P^2}$.

`check_residuals.py` runs a self-checking suite over synthetic matched pairs
(steep delay falloff, negative noise-dominated cells, known injected mode):

```bash
python check_residuals.py
```

## holdout.py

Held-out reconstruction comparison of the representations. Folds are
**contiguous LST blocks** — the retained windows form one sky track and
neighbouring rows are strongly correlated, so a random split would leak.
Scoring: fit PCA on training rows in residual space, project the held-out
rows, invert back to linear power, then compute an inverse-$P_N$-weighted
mean squared error over cells above the horizon wedge (+ buffer), restricted
to cells the basis actually reconstructed. Without `--ideal-dir` the
training-row mean stands in for the ideal reference; the output records which
reference was used.

Output `<label>.holdout.json`: metric and split descriptions, then per
spectral window and per representation the fold count, feature counts, the
mean error-vs-components curve, and the best component count.

## submit_aligned_pca.sh

Slurm driver. Submits, per branch, a short high-memory build job
(the merged container's arrays are ~31 GB in memory) and one low-memory
analysis job that runs every PCA variant (unmasked, `min-delay` at each
threshold in `MIN_DELAY_LIST`, `above-wedge`, and the two whitened variants)
plus the held-out comparison and the branch contrast. BLAS thread counts are
pinned to the Slurm allocation.

Environment knobs (defaults in parentheses): `QUORUM` (0.95),
`MIN_DELAY_LIST` ("200 300"), `WEDGE_BUFFER_NS` (500), `SKIP_BUILD` (0; set
to 1 to reuse existing samples and submit only the analysis), `BUILD_MEM`
(64G), `ANALYSIS_MEM` (8G), `ACCTG_FREQ` (5; accounting sample period so
short jobs still report a measured peak memory).
