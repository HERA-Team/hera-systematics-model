#!/usr/bin/env bash
# Submit the LST-aligned sample-building + PCA jobs on the NRAO nmpost cluster.
# Run ON herapost-master from the directory holding these scripts.
#
#   ./submit_aligned_pca.sh
#
# Jobs:
#   A) sum branch      : build_aligned_samples.py -> run_pca_aligned.py
#   B) eor-only branch : build_aligned_samples.py -> run_pca_aligned.py
#   C) contrast        : run_pca_aligned.py (sum - eor-only), after A and B
set -euo pipefail

PY=/lustre/aoc/projects/hera/kmandar/miniconda3/envs/validation_env/bin/python
BASE=/lustre/aoc/projects/hera/kmandar/systematics-model
SCRIPTS=$BASE/scripts
OUT=$BASE/aligned-samples
PCA=$BASE/pca
LOGS=$BASE/slurm-logs
W=/lustre/aoc/projects/hera/Validation/H6C_IDR2
SUM_PSPEC=$W/lstbin-outputs/redavg-smoothcal-inpaint-500ns-lstcal/inpaint/single_baseline_files/baselines_merged.pspec.h5
EOR_PSPEC=$W/lstbin-outputs/eor-only/single_baseline_files/baselines_merged.pspec.h5

mkdir -p "$OUT/sum" "$OUT/eor-only" "$PCA" "$LOGS"

submit_branch () {
  local label=$1 pspec=$2
  sbatch --parsable \
    --job-name="align-${label}" \
    --partition=batch \
    --mem=120G --cpus-per-task=4 --time=12:00:00 \
    --output="$LOGS/align-${label}-%j.out" \
    --wrap="set -e
      $PY $SCRIPTS/build_aligned_samples.py $pspec \
        --outdir $OUT/$label --label $label
      $PY $SCRIPTS/run_pca_aligned.py \
        --samples-dir $OUT/$label --label $label --outdir $PCA"
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
      --outdir $PCA")

echo "submitted: sum=$JOB_SUM eor-only=$JOB_EOR contrast=$JOB_CONTRAST"
echo "$(date -Is) sum=$JOB_SUM eor-only=$JOB_EOR contrast=$JOB_CONTRAST" >> "$BASE/job-history.txt"
