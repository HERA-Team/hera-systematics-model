#!/usr/bin/env bash
# Submit the LST-aligned sample-building, PCA and validation jobs on the NRAO
# nmpost cluster. Run ON herapost-master from the directory holding these
# scripts.
#
#   ./submit_aligned_pca.sh                 # quorum 0.95 (default)
#   QUORUM=1.0 ./submit_aligned_pca.sh      # reproduce the strict intersection
#
# Jobs:
#   A) sum branch      : build -> PCA (3 masks) -> held-out comparison
#   B) eor-only branch : build -> PCA (3 masks) -> held-out comparison
#   C) contrast        : PCA of (sum - eor-only), after A and B
#
# The PCA runs three times per branch over the same samples. Unmasked is the
# comparison baseline; the two masked variants exclude the brightest part of
# the plane, where an unrestricted PCA in linear power spends its components,
# and so report what the basis looks like over the region inference uses.
# Outputs are kept in separate directories because the per-spw filenames do not
# encode the mask.
set -euo pipefail

PY=/lustre/aoc/projects/hera/kmandar/miniconda3/envs/validation_env/bin/python
BASE=/lustre/aoc/projects/hera/kmandar/systematics-model
SCRIPTS=$BASE/scripts
OUT=$BASE/aligned-samples
PCA=$BASE/pca
HOLDOUT=$BASE/holdout
LOGS=$BASE/slurm-logs
W=/lustre/aoc/projects/hera/Validation/H6C_IDR2
SUM_PSPEC=$W/lstbin-outputs/redavg-smoothcal-inpaint-500ns-lstcal/inpaint/single_baseline_files/baselines_merged.pspec.h5
EOR_PSPEC=$W/lstbin-outputs/eor-only/single_baseline_files/baselines_merged.pspec.h5

QUORUM=${QUORUM:-0.95}
MIN_DELAY_NS=${MIN_DELAY_NS:-300}
WEDGE_BUFFER_NS=${WEDGE_BUFFER_NS:-500}
MASKS="none min-delay above-wedge"

mkdir -p "$OUT/sum" "$OUT/eor-only" "$PCA" "$HOLDOUT" "$LOGS"

submit_branch () {
  local label=$1 pspec=$2
  local steps="set -e
    $PY $SCRIPTS/build_aligned_samples.py $pspec \
      --outdir $OUT/$label --label $label --quorum $QUORUM"
  for mask in $MASKS; do
    steps="$steps
    $PY $SCRIPTS/run_pca_aligned.py \
      --samples-dir $OUT/$label --label $label \
      --mask $mask --min-delay-ns $MIN_DELAY_NS \
      --wedge-buffer-ns $WEDGE_BUFFER_NS \
      --outdir $PCA/$mask"
  done
  # held-out comparison of the residual representations. Without a matched
  # ideal branch this uses the training mean as the reference, which is a
  # stand-in: the numbers rank the representations against each other, they do
  # not validate a systematics model.
  steps="$steps
    $PY $SCRIPTS/holdout.py \
      --samples-dir $OUT/$label --label $label \
      --wedge-buffer-ns $WEDGE_BUFFER_NS \
      --outdir $HOLDOUT"
  sbatch --parsable \
    --job-name="align-${label}" \
    --partition=batch \
    --mem=120G --cpus-per-task=4 --time=12:00:00 \
    --output="$LOGS/align-${label}-%j.out" \
    --wrap="$steps"
}

JOB_SUM=$(submit_branch sum "$SUM_PSPEC")
JOB_EOR=$(submit_branch eor-only "$EOR_PSPEC")

JOB_CONTRAST=$(sbatch --parsable \
  --job-name=align-contrast \
  --partition=batch \
  --dependency=afterok:${JOB_SUM}:${JOB_EOR} \
  --mem=32G --cpus-per-task=2 --time=02:00:00 \
  --output="$LOGS/align-contrast-%j.out" \
  --wrap="set -e
    $PY $SCRIPTS/run_pca_aligned.py \
      --samples-dir $OUT/sum --label sum \
      --subtract-dir $OUT/eor-only --subtract-label eor-only \
      --time-tol-sec 130 \
      --outdir $PCA/none")

echo "submitted: sum=$JOB_SUM eor-only=$JOB_EOR contrast=$JOB_CONTRAST"
echo "  quorum=$QUORUM masks='$MASKS' min-delay=${MIN_DELAY_NS}ns"
echo "  check: sacct -j $JOB_SUM,$JOB_EOR,$JOB_CONTRAST --format=JobID,JobName,State,ExitCode"
echo "$(date -Is) quorum=$QUORUM sum=$JOB_SUM eor-only=$JOB_EOR contrast=$JOB_CONTRAST" \
  >> "$BASE/job-history.txt"
