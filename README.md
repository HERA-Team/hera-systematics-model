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

Visibility I/O commands accept explicit JSON lists of paths. Run large
inventories, baseline mapping and construction inside accounted Slurm tasks:

```bash
hera-systematics inventory --files references.json --output inventory.json
hera-systematics ideal map --references references.json --source source.uvh5 --output baselines.json
hera-systematics ideal chunk --reference reference.uvh5 --sources source-files.json --mapping baselines.json --output ideal.uvh5
hera-systematics ideal verify --reference reference.uvh5 --product ideal.uvh5 --output verification.json
hera-systematics ideal batch --chunks chunks.json --reference-inventory inventory.json --mapping baselines.json --output-dir ideal-chunks
hera-systematics cornerturn --files ideal-chunks.json --baselines antenna-pairs.json --output-dir baseline-files
```

The inventory records metadata and array layouts without asserting data
validity. Ideal construction uses cubic interpolation without extrapolation;
a source curve with any invalid knot is unavailable for that chunk. Finite
supported samples, including zeros, receive cleared flags and unit counts.
Support is determined by finite unflagged model knots; source sample counts
are not used because model visibilities need not represent counted observations.
Unsupported cells contain an invalid flag and zero count. Integration times
and physical coordinates are preserved. Reversed baselines use conjugated
visibility data with exchanged cross-polarizations. The verification command
checks every output cell, reference coordinates and the recorded file hashes.
Entirely unsupported chunks fail acceptance. Cornerturning accepts antenna
pairs such as `[[0, 1], [6, 34]]`, preserves exact data, flags, counts and time
gaps, and writes each output row once. It requires a new output directory and
produces an input manifest, per-file sidecars and a verification record.
An ideal batch uses a deterministic chunk inventory with explicit source files
and measured reference metadata. Add `--baselines antenna-pairs.json` to exercise
a selected subset. Repeated inputs are hashed before first consumption and
again before batch acceptance; intermediate lookups check file identity and
change timestamps. Each chunk retains its own verification product. These
temporary hash caches are never reused across tasks or executions.

Scheduler submissions accept repeated `--afterok JOB_ID` arguments for recorded
predecessors. Such jobs can be queued while their predecessors run, because
Slurm permits them to start only after every predecessor succeeds. Recorded
predecessor relationships bound every possible CPU, memory and two-job overlap;
independent queued jobs are conservatively treated as simultaneous. Storage
reservations include every queued task, including predecessors.

Spectral conversion requires native-sample membership exported during the
actual averaging operation. Matching row counts or rounded centroids is
insufficient. The spectral commands expose verified merging and per-window
record conversion:

```bash
hera-systematics spectra merge --inventory spectra.json --output merged.h5 --memberships-output merged-windows.npz
hera-systematics spectra groups --spectrum corrupted-merged.h5 --output length-groups.json
hera-systematics spectra records --spectrum corrupted-merged.h5 --native-grid native-grid.json --grouping length-groups.json --memberships corrupted-windows.npz --role corrupted --spw 0 --output corrupted-spw0.npz
```

The merge inventory has `schema_version: 1` and an `inputs` list. Each entry
contains `spectrum` and `memberships` file paths plus an integer
`baseline_pair_code`. Every declared baseline must occur exactly once. All
14 spectral windows, polarization order, delay/frequency coordinates, units,
cosmology and normalization must agree. Inputs are sorted by physical baseline;
their native row order is preserved. The writer uses bounded data blocks,
reopens the completed file, and compares every copied value exactly, including
nonfinite data support. Partial outputs are retained and cannot be overwritten.

The `.merge.npz` and `.merge.json` sidecars preserve source histories and
baseline-specific header values, such as fringe-rate corrections, with exact
source-file identities. Only header values common to all inputs become shared
merged-header attributes. Numerical source attributes remain in NPZ arrays,
including unavailable numerical values. The `.inputs.json` report records
successful input hashes before and after consumption. `hera-systematics verify`
on the merge NPZ also verifies its bound HDF5 file and input report.

The native-grid JSON contains `schema_version: 1`, `policy` (`shared` or
`retained`), monotonic `native_time_jd`, `anchor_jd`, `window_seconds`,
`native_samples_per_window`, and source identities in `sources`. Shared mode
requires complete windows and the exact same native time array in each input.
The captured notebook runner accepts `--native-grid`; it exports the consumed
native row identifiers, actual interleave centroids and resulting spectrum
centroids to `window-memberships.npz`. A merged export must cover each source
exactly and is rebound to the merged spectrum's verified file identity.

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
`include_kernel`, `representations`, `methods`, `group_exclusion`, `region`, and either `group` or `delay`
for a localized slice. The defaults use guard 12 and maximum rank 20. Run
expensive analyses through the compute scheduler.

The `region` defaults to `full`; `horizon` retains delays above baseline length
divided by the speed of light, and `horizon_buffer` adds 500 ns to that boundary.
These fixed masks precede any training-only group exclusion. Evaluation retains
the original eligible-cell denominator and records geometric exclusions
separately. Regions without feature support return an explicit failure record.

For guard sensitivity, run `evaluate --selection-from primary.npz` with a
configuration using guard 8 or 16. The primary evaluation must be complete and
use guard 12. Sample identities and all other configuration fields must match.
Outer choices and target partitions remain fixed; preprocessing and model fits
use the new training support. Both primary artifact hashes are recorded.

`cross-spw --left-samples left.npz --right-samples right.npz --left-fit left-fit.npz
--right-fit right-fit.npz --output comparison.npz` compares saved descriptive
fits. Exact native delay-center matching and conservative common-bin comparisons
have separate arrays and coverage counts. Signed-component and squared-loading
similarities are distinct. Bin comparisons retain overlap weights and assume
constant profiles within native bins; they do not equate spectral window functions.
Rank-zero and kernel fits report unavailable linear-component comparisons while
retaining conditional physical-mode energy measurements above 0.3 h Mpc^-1.

`spectra batch --inventory batch.json --output-dir batch-products --workers 4`
runs inside an existing Slurm allocation. Each concurrent command reserves two
CPUs and 16 GiB; four workers therefore require eight CPUs and 64 GiB. The outer
production runner still enforces aggregate job and storage limits. Inventory
schema version 1 contains `code_commit`, `notebook`, `native_grid`, `configuration`
and `baselines`. Each baseline entry supplies its integer `baseline_pair_code`,
absolute visibility `file` and adjacent `auto` path. The configuration explicitly
names `EFIELD_HEALPIX_BEAM_FILE` and `FR_SPECTRA_FILE` alongside notebook parameters.

Baseline entries run in sorted physical-code order. Each completed command is
verified before replacement work launches; a failed command or product stops new
launches while active commands finish. An optional baseline `reuse` directory
must contain an unchanged acceptance receipt with identical code, resolved runtime,
parameters, native grid and consumed-file identities. Reuse reads existing files
without overwriting them. All consumed inputs and accepted outputs are hashed
again before the batch verification record is written.

`replay --samples paired.npz --evaluation evaluation.npz --output replay.json`
checks saved predictor-only coefficients, decoded target predictions, coverage,
and selected/zero/mean physical-window losses against the bound paired samples.
It does not refit bases or repeat selection. Prediction checks use relative
tolerance 1e-10 plus 1e-10 times the recorded corrupted noise; coefficient and
loss tolerances are included in the output. An incomplete evaluation remains
incomplete even when its available predictions replay successfully.

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

Bootstrap stability reads the saved fit configuration, including a fixed cylindrical slice and geometric region. Data-dependent group exclusions are recomputed using each bootstrap draw, with repeated rows retaining their sampling multiplicity. Artifacts record reference and replicate feature support and exclusion measurements. The source sample identities must match the descriptive fit.

Per-group and per-delay diagnostic views retain the selected physical coordinates and contributor identities. Their plots use coordinate profiles for a single group or delay; these are cylindrical slices without spherical averaging. Fit and evaluation configurations must agree except for the validation guard, and the source file identities must match both artifacts.

`hera-systematics summary --fit fit.npz --evaluation evaluation.npz --output summary.json` exports per-fold choices, coverage, equal-fold predictive losses and zero/mean comparisons. Standard errors use physical-time fold scores, not feature partitions. Incomplete fold sets have unavailable aggregate losses; a zero baseline has an unavailable loss ratio. Descriptive training loss is labeled separately. The export contains numerical evidence and does not assign a scientific conclusion.
