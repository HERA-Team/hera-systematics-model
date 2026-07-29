#!/usr/bin/env bash
# Submit the LST-aligned sample-building, PCA and validation jobs on the NRAO
# nmpost cluster. Run ON herapost-master from the directory holding these
# scripts.
#
#   ./submit_aligned_pca.sh                 # quorum 0.95 (default)
#   QUORUM=1.0 ./submit_aligned_pca.sh      # reproduce the strict intersection
#
# Jobs:
#   A) build-sum       : build_aligned_samples.py on the sum branch
#   B) build-eor-only  : build_aligned_samples.py on the eor-only branch
#   C) analysis        : PCA over every mask for both branches, the held-out
#                        representation comparison, and the labelled contrast;
#                        runs after A and B
#
# Why the work is split this way. Building the samples has to hold the whole
# merged UVPSpec in memory -- the arrays in each merged file total about 31 GB
# -- but it finishes in well under a minute. Everything downstream operates on
# per-window matrices of a few MB, so the PCA and validation steps need almost
# no memory. Keeping them in one job would reserve tens of GB for work that
# does not use it, which is both wasteful on a shared partition and slower to
# schedule.
#
# The PCA runs once per mask over the same samples. Unmasked is the comparison
# baseline; the masked variants exclude the brightest part of the plane, where
# an unrestricted PCA in linear power spends its components, and so report what
# the basis looks like over the region inference uses. Outputs go to separate
# directories because the per-spw filenames do not encode the mask.
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
# SKIP_BUILD=1 reuses existing aligned samples and submits only the analysis
# job, for iterating on PCA/validation settings without rebuilding.
SKIP_BUILD=${SKIP_BUILD:-0}
# One PCA per threshold. Delay bins begin at 94 ns and step by about the same,
# so 200 removes only the brightest bin while 300 removes the brightest three.
# Running both shows whether the basis changes gradually as bins are removed or
# all at once with the first.
MIN_DELAY_LIST=${MIN_DELAY_LIST:-"200 300"}
WEDGE_BUFFER_NS=${WEDGE_BUFFER_NS:-500}

# Sized from the measured array footprint (~31 GB) plus room for transient
# copies during the read. The earlier 120 G was inherited, not measured.
BUILD_MEM=${BUILD_MEM:-64G}
BUILD_CPUS=${BUILD_CPUS:-2}
BUILD_TIME=${BUILD_TIME:-00:20:00}
# Observed build time is well under a minute; the walltime is generous so a
# slow filesystem cannot kill the job, while staying short enough to backfill.
ANALYSIS_MEM=${ANALYSIS_MEM:-8G}
ANALYSIS_CPUS=${ANALYSIS_CPUS:-4}
ANALYSIS_TIME=${ANALYSIS_TIME:-00:30:00}
ACCTG_FREQ=${ACCTG_FREQ:-5}

# numpy and its BLAS default to every core on the node, not to the cores Slurm
# allocated, so without this a few-core job can spawn threads across a shared
# node. Single-quoted: these must reach the job script unexpanded.
THREADS='export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS'

mkdir -p "$OUT/sum" "$OUT/eor-only" "$PCA" "$HOLDOUT" "$LOGS"

submit_build () {
  local label=$1 pspec=$2
  # The cluster samples accounting every 30 s and this step finishes inside
  # that window, so MaxRSS comes back empty and the memory request cannot be
  # checked against reality. Sampling every few seconds makes the peak
  # measurable, so BUILD_MEM can be set from evidence after a run.
  sbatch --parsable \
    --job-name="build-${label}" \
    --partition=batch \
    --acctg-freq="task=$ACCTG_FREQ" \
    --mem="$BUILD_MEM" --cpus-per-task="$BUILD_CPUS" --time="$BUILD_TIME" \
    --output="$LOGS/build-${label}-%j.out" \
    --wrap="set -e
$THREADS
$PY $SCRIPTS/build_aligned_samples.py $pspec \
  --outdir $OUT/$label --label $label --quorum $QUORUM"
}

if [ "$SKIP_BUILD" = 1 ]; then
  JOB_SUM=skipped
  JOB_EOR=skipped
  DEP_ARGS=()
else
  JOB_SUM=$(submit_build sum "$SUM_PSPEC")
  JOB_EOR=$(submit_build eor-only "$EOR_PSPEC")
  DEP_ARGS=("--dependency=afterok:${JOB_SUM}:${JOB_EOR}")
fi

# ---- analysis: every mask for both branches, then validation and contrast ----
steps="set -e
$THREADS"
for label in sum eor-only; do
  variants="none:"
  for d in $MIN_DELAY_LIST; do variants="$variants min-delay:$d"; done
  variants="$variants above-wedge:"
  for v in $variants; do
    mask=${v%%:*}
    thr=${v##*:}
    extra=""
    sub=$mask
    if [ "$mask" = "min-delay" ]; then
      sub="min-delay-$thr"
      extra="--min-delay-ns $thr"
    fi
    steps="$steps
$PY $SCRIPTS/run_pca_aligned.py \
  --samples-dir $OUT/$label --label $label \
  --mask $mask $extra --wedge-buffer-ns $WEDGE_BUFFER_NS \
  --outdir $PCA/$sub"
  done
  # Noise-whitened variants. Redundant-group noise spans orders of magnitude
  # across baseline lengths, so the unwhitened runs assign leading components
  # to the noisiest groups; the whitened runs measure structure relative to
  # noise instead.
  for wv in "none:whitened" "above-wedge:whitened-above-wedge"; do
    mask=${wv%%:*}
    sub=${wv##*:}
    steps="$steps
$PY $SCRIPTS/run_pca_aligned.py \
  --samples-dir $OUT/$label --label $label \
  --mask $mask --whiten pn-median --wedge-buffer-ns $WEDGE_BUFFER_NS \
  --outdir $PCA/$sub"
  done
  # Held-out comparison of the residual representations. Without a matched
  # ideal branch this uses the training mean as the reference, which is a
  # stand-in: the numbers rank the representations against each other, they do
  # not validate a systematics model.
  steps="$steps
$PY $SCRIPTS/holdout.py \
  --samples-dir $OUT/$label --label $label \
  --wedge-buffer-ns $WEDGE_BUFFER_NS \
  --outdir $HOLDOUT"
done
steps="$steps
$PY $SCRIPTS/run_pca_aligned.py \
  --samples-dir $OUT/sum --label sum \
  --subtract-dir $OUT/eor-only --subtract-label eor-only \
  --time-tol-sec 130 \
  --outdir $PCA/none"

JOB_ANALYSIS=$(sbatch --parsable \
  --job-name=align-analysis \
  --partition=batch \
  ${DEP_ARGS[@]+"${DEP_ARGS[@]}"} \
  --mem="$ANALYSIS_MEM" --cpus-per-task="$ANALYSIS_CPUS" --time="$ANALYSIS_TIME" \
  --output="$LOGS/align-analysis-%j.out" \
  --wrap="$steps")

echo "submitted: build-sum=$JOB_SUM build-eor-only=$JOB_EOR analysis=$JOB_ANALYSIS"
echo "  quorum=$QUORUM  min-delay thresholds='$MIN_DELAY_LIST' ns, plus none and above-wedge"
echo "  build: $BUILD_MEM / $BUILD_CPUS cpu / $BUILD_TIME   analysis: $ANALYSIS_MEM / $ANALYSIS_CPUS cpu / $ANALYSIS_TIME"
echo "  check: sacct -j $JOB_SUM,$JOB_EOR,$JOB_ANALYSIS --format=JobID,JobName%16,State,ExitCode,Elapsed,MaxRSS"
echo "$(date -Is) quorum=$QUORUM build-sum=$JOB_SUM build-eor-only=$JOB_EOR analysis=$JOB_ANALYSIS" \
  >> "$BASE/job-history.txt"
