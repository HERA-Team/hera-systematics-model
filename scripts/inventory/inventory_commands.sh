#!/usr/bin/env bash
# Read-only reconnaissance commands used for the 2026-07-25 H6C-IDR2 inventory
# (manifests/h6c_idr2_inventory.json). Run from a machine with the NRAO guest
# gateway SSH master active; every command is non-mutating.
#
# Rerunning these and comparing against the manifest is the reproducibility
# check. Heavy recursive walks are intentionally avoided (Lustre etiquette):
# counts are -maxdepth 1 and there is exactly one du -sh per product family.
set -euo pipefail

JUMP="ssh -J kmandar@guest-login.aoc.nrao.edu kmandar@herapost-master"
B=/lustre/aoc/projects/hera/Validation/DataReleases/H6C-IDR2
W=/lustre/aoc/projects/hera/Validation/H6C_IDR2
I=$B/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint
PY=/lustre/aoc/projects/hera/kmandar/miniconda3/envs/validation_env/bin/python

# --- layout -----------------------------------------------------------------
$JUMP "ls -la $B/"                          # 4 symlinks -> /home/herastore02-2
$JUMP "for d in chunked-ideal-sims lstbin-outputs sim_data sky_models; do
         echo \"== \$d ==\"; ls -la $B/\$d/; done"

# --- file counts (maxdepth 1) and naming conventions ------------------------
$JUMP "for d in chunked-ideal-sims/eor-grf-1024 chunked-ideal-sims/gsm_nside1024 \
             chunked-ideal-sims/ptsrc1024 \
             chunked-ideal-sims/ptsrc1024-original-wrong-specindex \
             sim_data/diffuse sim_data/eor sim_data/foregrounds sim_data/ptsrc \
             sim_data/sum sky_models/Canonical/eor-grf-1024 \
             sky_models/Canonical/gsm_nside1024 sky_models/Canonical/ptsrc1024 \
             sky_models/Canonical/ptsrc1024-original-wrong-specidx sky_models/raw; do
         n=\$(find $B/\$d -maxdepth 1 -type f | wc -l)
         echo \"## \$d files=\$n\"; ls $B/\$d | head -3; done"

# --- single-baseline census --------------------------------------------------
$JUMP "cd $I/single_baseline_files
       echo total: \$(find . -maxdepth 1 -type f | wc -l)
       echo pspec: \$(ls -1 *.sum.pspec.h5 | wc -l)
       echo uvh5:  \$(ls -1 *.sum.uvh5 | wc -l)
       autos=0; cross=0; crossnopspec=0
       for f in zen.LST.baseline.*.sum.uvh5; do
         bl=\${f#zen.LST.baseline.}; bl=\${bl%.sum.uvh5}
         a=\${bl%_*}; b=\${bl#*_}
         if [ \"\$a\" = \"\$b\" ]; then autos=\$((autos+1)); else
           cross=\$((cross+1))
           [ -f \"zen.LST.baseline.\${bl}.sum.pspec.h5\" ] || crossnopspec=\$((crossnopspec+1))
         fi
       done
       echo autos=\$autos cross=\$cross cross_without_pspec=\$crossnopspec"

# --- lstbin output categories ------------------------------------------------
$JUMP "ls -1 $I/*.uvh5 | sed 's/.*zen\.//' | sed 's/\.[0-9].*//' | sort | uniq -c"

# --- sizes (one du per family) ----------------------------------------------
$JUMP "for d in chunked-ideal-sims sim_data sky_models lstbin-outputs; do
         du -sh $B/\$d/; done"

# --- per-file metadata (representative sample) ------------------------------
$JUMP "$PY -" < "$(dirname "$0")/extract_metadata.py" > 06_metadata.jsonl

# --- provenance --------------------------------------------------------------
$JUMP "cat $B/lstbin-outputs/v3.1-simultaneous-inpaint/lstavg-config.toml"
$JUMP "head -60 $B/lstbin-outputs/v3.1-simultaneous-inpaint/lstbin-config.toml"
$JUMP "cat $W/configs/2459861.yaml"        # systematics injection recipe
$JUMP "ls -la $W/; ls -1 $W/2459861/ | sed 's/zen\.[0-9]*\.[0-9]*\.//' | sort |
       uniq -c | sort -rn | head -12"

# --- capacity ----------------------------------------------------------------
$JUMP "df -h /lustre/aoc/projects/hera | tail -2; df -h /home/herastore02-2 | tail -1
       lfs quota -h -g hera /lustre/aoc | head -6"

# --- rebuild the manifest locally -------------------------------------------
# python3 build_manifest.py manifests/h6c_idr2_inventory.json
