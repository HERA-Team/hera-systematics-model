# Analysis entrypoints and artifact formats

The installable package supplies paired residual analysis. Install from the
repository root with `python -m pip install -e '.[kernel,plot]'`. Core array
analysis does not import the HERA visibility or power-spectrum libraries;
conversion of HERA files requires the optional I/O environment.

## Versioned residual workflow

```bash
hera-systematics pair --corrupted corrupted.npz --ideal ideal.npz --output paired.npz
hera-systematics evaluate --samples paired.npz --config analysis.json --output evaluation.npz
hera-systematics fit --samples paired.npz --config analysis.json --output fit.npz
hera-systematics diagnostics --samples paired.npz --fit fit.npz \
    --evaluation evaluation.npz --output diagnostics.npz
hera-systematics stability --samples paired.npz --fit fit.npz \
    --replicates 500 --block-length 12 --output stability.npz
hera-systematics plot diagnostics.npz --output-dir figures
hera-systematics verify evaluation.npz
```

`pair` accepts two `spectrum-records` artifacts. It checks spectral-window,
polarization, units, cosmology, baseline geometry, physical window identity and
delay coordinates. Both branches use the same contributing baselines and
normalized corrupted-noise weights. A valid zero is measured power. Invalid
samples supply no averaging weight. The residual is corrupted power minus ideal
power, with uncertainty derived from the corrupted branch.

Each NPZ has a JSON sidecar containing the schema version, kind, numerical-file
hash, array descriptions and structured metadata. Existing destinations are
rejected. Numerical unavailable values are accompanied by validity/support
arrays; JSON numerical unavailability uses `null` and a reason. The four public
artifact kinds are `spectrum-records`, `paired-samples`, `fitted-model` and
`evaluation`; diagnostic measurements use `diagnostics`.

| Artifact | Numerical contents | Identity and reconstruction metadata |
|---|---|---|
| Spectrum records | Unfolded power, noise, validity and one physical baseline/window identity per row | SPW, polarization, units, cosmology, reference grid, source files |
| Paired samples | Corrupted and ideal planes, noise, validity, delay and cylindrical coordinates, baseline membership and normalized averaging weights | Window IDs, physical time, source identities and folding/noise assumptions |
| Fitted model | Means, support, components or kernel state, training IDs, scales and decoder state | Method, representation, rank, learned parameters and convergence |
| Evaluation | Exact partitions, predictions, target support, coverage, losses and inferred coefficients | Candidate configurations, selected models, training IDs, runtime and input hashes |

`evaluate` uses four outer physical-time folds and three inner folds. A target
band and its guard features are excluded from coefficient inference. Selection
uses withheld-cell loss in linear power units; projection reconstruction is a
separate diagnostic. `fit` performs full-data descriptive selection and does not
supply an out-of-fold performance estimate. Incomplete evaluations are retained
with reasons and return a nonzero exit status.

An empty `analysis.json` selects the full comparison. Supported configuration
keys are `guard` (8, 12 or 16 native windows), `max_rank` (0–20),
`include_kernel`, `representations`, `methods`, `group`, `delay`, and
`group_exclusion`. `group` selects one baseline-length group; `delay` selects one
delay bin. They are mutually exclusive. Group exclusion is available only for
full-plane SPWs 6 and 7, using `leading_linear_loading` or
`noise_weighted_energy`; each training partition supplies its own exclusions.

Representations are `linear`, `noise_weighted`, `signed_asinh`, and `log_ratio`.
The methods are `complete` and `masked`, plus the optional kernel comparison.
Log offsets and all fitted preprocessing use training rows. Missing targets
cannot improve a score by disappearing from the required scoring support.
Noise-scaled and nonlinear representations require their complete inverse
transforms; their components are not a universal physical-power basis.

Diagnostics include signed means and loadings, validity/contributor/noise maps,
score distributions, physical-time autocorrelation and coverage-aware regional
losses. Fourier phase surrogates hold amplitudes fixed and assume circular
stationarity; they are not exactly Gaussian draws. Bootstrap blocks stay inside
continuous observed time segments. Component signs, assignments, full subspace
angles and near-degenerate cluster angles are retained. A score distribution
from one correlated LST arc does not calibrate a nuisance prior.

## Compatibility entrypoints

These script names invoke the corresponding package command and accept its
current arguments:

| Script | Package command |
|---|---|
| `holdout.py` | `hera-systematics evaluate` |
| `mode_diagnostics.py` | `hera-systematics diagnostics` |
| `plot_mode_diagnostics.py` | `hera-systematics plot` |
| `residuals.py` | Import of the package residual-transform API |
| `check_residuals.py` | Residual-transform numerical self-checks |

A historical `--samples-dir` or implicit ideal-reference fallback is not a valid
input to the nested residual evaluator. Use paired-sample artifacts with a
verified ideal branch.

## Historical aligned artifacts

`build_aligned_samples.py` reads a merged single-branch UVPSpec product and writes
`<label>.aligned.spwNN.npz`. Its independently constructed branch averages do
not establish the common contributors and weights required for a paired
residual. These artifacts remain readable for descriptive analysis.

| Key | Shape | Meaning |
|---|---|---|
| `matrix`, `pn_eff`, `valid` | `(ntime, ngroup * ndelay)` | Power, effective noise and explicit validity |
| `cube_shape` | `(3,)` | Original time/group/delay dimensions |
| `time_grid`, `lst_grid` | `(ntime,)` | Physical JD and LST in radians |
| `window_counts` | `(ntime,)` | Baseline-pair count per averaging window |
| `blp_lens`, `group_reps` | `(ngroup,)` | Lengths in metres and representative group identifiers |
| `dlys`, `kparas` | `(ndelay,)` | Folded delay in seconds and parallel coordinate in h/Mpc |
| `kperps` | `(ngroup,)` | Transverse coordinate in h/Mpc, sometimes absent numerically in old products |

`run_pca_aligned.py` performs descriptive single-branch SVD on features valid in
every input row. It requires explicit validity and finite positive noise,
preserves measured zeros, rejects branch subtraction and uses a new output
directory. `--whiten pn-median` divides each feature by its median noise;
`--whiten pn-cell` divides by each cell's noise. The saved `feature_scale` or
`pn_eff` supplies the inverse. All components, scores and singular values are
saved. A variance-threshold count is descriptive and does not identify physical
systematics dimension.

`plot_aligned_modes.py` reads historical PCA files, labels the scaling used for
the mean, and plots scores in physical time order through LST zero. Its optional
coordinate recovery uses the Planck15 cosmology and records the derived
redshift. A smooth LST trend does not establish the cause of a component.

`trim_aligned_windows.py` makes explicit row-exclusion copies, preserving
physical times and original row identifiers. It rejects mismatched time grids,
missing requested SPWs, repeated trimming and an existing output directory.
`--drop-empty` uses validity, finite power and positive noise, preserving valid
all-zero rows. `--max-noise-inflation` is a descriptive sensitivity operation;
it does not implement training-only exclusion for predictive validation.

## Compute tasks and verification

`hera-systematics production init` snapshots the actual configuration and input
identities into a new run. `production define` reserves an immutable task JSON.
`production submit` applies shared CPU, RAM, task-count and retained-storage
limits before Slurm submission. `production verify` requires successful job and
step exit states plus verified output hashes and readable structures.

`submit_aligned_pca.sh` is a wrapper for `production submit`; run it with `--help`
for the required run, task, Python and package-source paths. `HSM_PYTHON` chooses
the Python used by the submission command. Heavy work belongs in the submitted
task. The wrapper creates no independent branch averages or implicit contrasts.

Structural/numerical parity is available through
`python -m hera_systematics_model.parity`. Baseline-pair and polarization-code
options compare explicit subsets of merged libraries. Coordinates remain exact;
an optional `--power-noise-atol` records an absolute numerical allowance in units
of reference noise, alongside the strict relative-tolerance result. Scientific
acceptance also requires the complete expected inventory and source ancestry.
