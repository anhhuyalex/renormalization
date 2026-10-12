#!/usr/bin/env bash
# Submit Figure 2 cache regeneration as a 4-task Slurm Array across separate cluster nodes,
# followed by an automatic dependent job that renders Figure 2 once all 4 complete.

set -euo pipefail

cd /mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization
mkdir -p ./slurms ./figs_camera_ready

echo "Submitting 4-task Slurm array (1 node per cache stage)..."
ARRAY_JOB=$(sbatch --parsable --array=1-4 analysis5_cache.sh)
echo "Submitted array job ID: $ARRAY_JOB (Tasks 1-4: top_ab, top_c, bottom_ab, bottom_c)"

echo "Submitting dependent rendering job to run after all 4 array tasks complete..."
RENDER_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} --job-name=figure2-render \
  --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --output=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization/slurms/figure2-render-%j.out \
  analysis5_generate.sh --figures 2)
echo "Submitted dependent render job ID: $RENDER_JOB (will run after $ARRAY_JOB finishes)"
