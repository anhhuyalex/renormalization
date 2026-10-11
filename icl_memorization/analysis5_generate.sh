#!/usr/bin/env bash
#SBATCH --job-name=analysis5-generate
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization
#SBATCH --output=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization/slurms/figure-generate-%j.out

set -euo pipefail

project_dir="/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization"
cd "$project_dir"

# TeX tools, libraries, and Python environment
FMRI_DIR="/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri"
export PATH="${FMRI_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${FMRI_DIR}/lib:${LD_LIBRARY_PATH:-}"

# Fast local cache to avoid NFS locking and directory collision
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${UID:-$(id -u)}}"
mkdir -p "$MPLCONFIGDIR" ./figs_camera_ready ./slurms

PYTHON="${FMRI_DIR}/bin/python"

echo "=========================================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Hostname:         $(hostname)"
echo "Allocated CPUs:   ${SLURM_CPUS_PER_TASK:-not-set}"
echo "Date:             $(date)"
echo "Python:           $PYTHON"
echo "MPLCONFIGDIR:     $MPLCONFIGDIR"
echo "Arguments:        ${*:-<none (all figures)>}"
echo "=========================================================="

"$PYTHON" analysis5.py --parallel "$@"

echo "=========================================================="
echo "All figures generated successfully at $(date)"
echo "Output files written to: ${project_dir}/figs_camera_ready"
echo "=========================================================="
