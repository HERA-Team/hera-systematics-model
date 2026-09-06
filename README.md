# hera-systematics-model

Coordinate-explicit matching and predictive analysis of HERA cylindrical
power spectra $P(k_\perp, k_\parallel)$. The residual is corrupted power minus
matched ideal power. It includes differences between processing paths as well
as the effects of simulated corruptions. A learned mode does not identify a
particular physical cause.

What is where:

- `manifests/` - list of the H6C IDR2 simulation products on NRAO disk (json + csv)
- `scripts/inventory/` - the scripts that made that list
- `scripts/analysis/` - compatibility entrypoints and historical artifact readers
- `src/hera_systematics_model/` - paired artifacts, residual transforms, models and validation
- `tests/` - synthetic verification without the HERA data stack
- `results/` - small summary files from finished runs
- `Progress.md` - log of completed work

Big files (pspec containers, sample matrices, per-mode npz files) stay on
NRAO storage. Only code, manifests and small summaries are kept here.

Python 3.10 or newer is required. Install the array-analysis package with
`python -m pip install -e .`. Optional dependency groups are `io` for HERA
products, `kernel` for kernel fitting, `plot` for figures, and `dev` for tests.
For example, `python -m pip install -e '.[kernel,dev]'` enables synthetic
verification with `python -m pytest -q`. CI checks Python 3.10 with NumPy 1.26
and Python 3.12 with NumPy 2.0. Production I/O environments must be captured
and verified separately against their retained products.

The unified command operates on versioned NPZ artifacts with JSON sidecars:

```bash
hera-systematics pair --corrupted corrupted.npz --ideal ideal.npz --output paired.npz
hera-systematics evaluate --samples paired.npz --config analysis.json --output evaluation.npz
hera-systematics fit --samples paired.npz --config analysis.json --output fit.npz
hera-systematics diagnostics --samples paired.npz --fit fit.npz --evaluation evaluation.npz --output diagnostics.npz
hera-systematics plot diagnostics.npz --output-dir figures
hera-systematics verify evaluation.npz
```

An empty configuration object selects all four residual representations,
complete-feature PCA, masked factorization and the kernel comparison. Optional
keys are `guard` (8, 12 or 16 native windows), `max_rank` (0 through 20),
`include_kernel`, `representations`, `methods`, `group_exclusion`, and either `group` or `delay`
for a localized slice. The defaults use guard 12 and maximum rank 20. Run
expensive analyses through the compute scheduler.

Evaluation uses four outer physical-time folds and three inner folds, with
withheld feature regions for coefficient inference. Final descriptive fits
use separate inner selection and carry training statistics only. Model
collections retain explicit transforms, components or kernel state, decoders,
training identities and hash-linked references. New products are immutable;
an existing destination is an error. Verification of an incomplete evaluation
returns a nonzero exit status even when the artifact itself is readable.

Each command records configuration, consumed source hashes and resolved package
versions. Numerical arrays can contain masked unavailable values; JSON cannot
contain nonfinite numbers. Nonlinear and per-cell-scaled representations do
not define a universal basis in physical power units.

Production tasks use immutable run and task definitions. The submission
command accounts for all active workflow tasks, checks resource and retained
storage limits, and reserves each task once:

```bash
hera-systematics production --help
hera-systematics production submit --run RUN --task TASK --python PYTHON --package-source SRC
hera-systematics production verify --run RUN --task TASK --output acceptance.json
```

Historical single-branch PCA files remain readable for descriptive plotting:

```bash
python scripts/analysis/plot_aligned_modes.py --pca-dir <dir> --label sum --outdir figures
```

See `scripts/analysis/README.md` for what each script does and what is in
the output files.
