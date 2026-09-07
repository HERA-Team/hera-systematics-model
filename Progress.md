# 26th July 2026

## Completed

### H6C-IDR2 product inventory

- Cataloged the available simulation, corrupted-chain, single-baseline power-spectrum, merged, and time-averaged products, including their locations, sizes, shapes, and available provenance.
- Reconciled all 884 single-baseline inputs: 789 power spectra were completed, 94 fully flagged polarization cases contained no usable weighted data, and one autocorrelation was excluded from the cross-correlation analysis. No existing input remains unaccounted for.

### Official analysis inputs

- Selected the official merged `sum` product at `/lustre/aoc/projects/hera/Validation/H6C_IDR2/lstbin-outputs/redavg-smoothcal-inpaint-500ns-lstcal/inpaint/single_baseline_files/baselines_merged.pspec.h5` as the foreground-plus-EoR branch containing the existing corrupted processing chain.
- Selected the official merged `eor-only` product at `/lustre/aoc/projects/hera/Validation/H6C_IDR2/lstbin-outputs/eor-only/single_baseline_files/baselines_merged.pspec.h5` as the EoR reference branch. These products replace the earlier custom notebook merge and provide consistent spectral-window, baseline-pair, and polarization organization.

### Project storage

- Established `/lustre/aoc/projects/hera/kmandar/systematics-model/` as the NRAO output area and configured access to all. 
- Kept large matrices and generated products on NRAO storage while retaining compact manifests, provenance, summaries, and analysis code in my local repository.

### Analysis implementation

- Aligned baseline-pair records using physical 270.59-second JD windows instead of assuming that equal row numbers represent the same observation time.
- Kept time windows with data from at least 750 of 789 baseline pairs and recorded missing data, contributor counts, and noise estimates for each cell.
- Implemented PCA/SVD analysis, residual-transform comparisons, diagnostic plots, held-out tests, and Slurm batch jobs.

# 7th September 2026

## Completed

- Added an installable array-analysis package, a unified command interface and synthetic CI on Python 3.10 and 3.12.
- Added versioned sample, model and evaluation artifacts with physical identifiers, explicit validity, file hashes and reconstructible prediction state.
- Implemented nested physical-time and withheld-feature validation, complete-feature PCA, observed-entry factorization, kernel prediction with a saved decoder, common-coverage scoring and training-only model selection.
- Added distribution, baseline, localized-slice, subspace-stability and diagnostic-plot implementations. Synthetic checks exercise missing data, leakage, genuine zeros, signed power, solver failures and serialization.
- Added immutable Slurm task definitions, aggregate resource and retained-storage checks, failure propagation and verified-product acceptance.
- Reproduced baseline 6–34 against the retained merged corrupted library across all 14 spectral windows. Structural coordinates agreed exactly; raw and time-averaged comparisons passed relative tolerance 1e-8 with an additional power tolerance of 1e-6 times the recorded thermal-noise scale. Verification job 11812880 completed with exit status 0:0.
- Added deterministic source-baseline mapping, explicit ideal flags and unit counts, conjugate-polarization handling, physical visibility inventories and output-policy verification.
