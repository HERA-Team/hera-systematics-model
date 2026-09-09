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
- Verified eight ideal visibility chunks on both sides of LST zero, including every supported cross-baseline and the autocorrelation. The products contain 43,450,368 supported cells; 13,940,583 inherited flags were cleared. Verification job 11816832 completed with exit status 0:0.
- Added streamed cornerturning, explicit native averaging exports, physical spectral merge checks and bounded-memory copies with reopened payload verification. Synthetic regression checks cover missing inputs, mismatched metadata, reordered baselines, corrupted copies and exact source-to-membership binding.
- Hash-verified 8,053 consumed reference and archived-sky files totaling 1,558,825,892,274 bytes. Inventory job 11814631 completed with exit status 0:0; the compact product hashes and independent acceptance record were checked.
- Verified shared-grid spectral processing on baseline 6–34 across all 14 spectral windows. Job 11822216 exported 146 windows with 28 native samples each, covered all 4,088 native rows, and passed exact merged-payload and spectrum-reader checks.
- Verified explicit unprojected UVW correction on baseline 81–0. Job 11824401 changed 2,044 metadata rows by up to 81.33693836783411 metres; raw and time-averaged spectra were bitwise identical before and after correction and against the retained library across all 14 spectral windows.
- Added bounded spectral subprocess batches, exact per-baseline reuse receipts, saved-evaluation numerical replay, frozen-choice guard sensitivity, and source-bound per-fold evidence summaries.
- Added configuration-preserving bootstrap refits, both cylindrical diagnostic slice views, and standalone bootstrap and cross-window figures. Synthetic block-length checks completed 500 draws each at lengths 8, 12 and 16.
- Verified four concurrent corrupted-baseline runs for 0–1, 0–2, 0–3 and 0–4 across all 14 spectral windows. Job 11829640 completed in 16:54 with exit status 0:0; its merged product contains 584 physical sample rows, with 146 native windows per baseline. Independent acceptance job 11829672 completed with exit status 0:0 and verified the products and resource measurements.
- Verified canonical single-row labels in fully time-averaged companions. All eight raw and time-averaged numerical payload comparisons were bitwise identical to their recorded comparison products. Original and resulting labels are retained in per-baseline metadata reports. The four-baseline task retained 554,608,982 bytes, including notebooks, spectra, merge and verification products.
- Verified spectrum-record extraction and serialization for baseline 6–34 across all 14 source SPWs. Job 11830546 completed with exit status 0:0. Independent checks matched saved power and noise bitwise to each original HDF5 window and matched physical frequencies, external SPW identifiers and all 146 time windows exactly, including reader-local SPW renumbering.

# 9th September 2026

## Completed

- Accepted the complete ideal visibility library after independent verification of 884 physical baselines, 4,088 native time rows, finite-sample validity, flags, unit counts, physical coordinates and payload hashes. Producer job 11871664 and acceptance job 11871888 completed with exit status 0:0.
- Added geometry-bound fringe-rate cache aliases for 26 ideal baselines absent from the original cache. The maximum mapped ENU-vector difference is 1.0669556142951236e-9 metres against a 1e-6-metre tolerance. Full-span baseline 0–326 smoke job 11942776 verified all 14 spectral windows and 146 averaging windows.
- Added exact spectral-batch input forensics and reuse acceptance without overwriting retained products. Ideal reuse job 11974627 and corrupted reuse job 11975159 completed with exit status 0:0 after double-hash verification of their registered inputs.
- Added a versioned command that applies the declared per-window scientific conclusion rule to saved four-fold evidence. It requires common-plane selected, zero and mean scores in all four physical-time folds, selects the baseline with the lower four-fold mean, and requires its positive paired mean improvement to exceed one fold standard error.
