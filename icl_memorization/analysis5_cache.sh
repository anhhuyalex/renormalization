#!/usr/bin/env bash
#SBATCH --job-name=figure2-cache
#SBATCH --cpus-per-task=20
#SBATCH --mem=258G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization
#SBATCH --output=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization/slurms/figure2-cache-%A_%a.out

set -euo pipefail

project_dir="/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization"
cd "$project_dir"

# TeX tools, libraries, font maps, and Python environment
FMRI_DIR="/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri"
FMRI_SHARE="${FMRI_DIR}/share"
export PATH="${FMRI_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${FMRI_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TEXMFVAR="${FMRI_SHARE}/texmf-var"
export TEXMFCONFIG="${FMRI_SHARE}/texmf-config"
export TEXMFDIST="${FMRI_SHARE}/texmf-dist"
export TEXMFLOCAL="${FMRI_SHARE}/texmf-local"
export TEXMFCNF="${FMRI_SHARE}:${TEXMFCNF:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/icl_memorization_mpl_${UID:-$(id -u)}}"
mkdir -p "$MPLCONFIGDIR" ./figs_camera_ready ./slurms ./cache

PYTHON="${FMRI_DIR}/bin/python"
CPUS="${SLURM_CPUS_PER_TASK:-20}"

echo "=========================================================="
echo "Job ID:           ${SLURM_JOB_ID:-local} (Array Task: ${SLURM_ARRAY_TASK_ID:-none})"
echo "Hostname:         $(hostname)"
echo "Allocated CPUs:   $CPUS"
echo "Date:             $(date)"
echo "Python:           $PYTHON"
echo "MPLCONFIGDIR:     $MPLCONFIGDIR"
echo "=========================================================="

# Check if running as a Slurm Array task (Tasks 1 to 4)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    case "$SLURM_ARRAY_TASK_ID" in
        1)
            echo "[Array Task 1] Regenerating Figure 2 Top-Row (a, b) caches (GPT-2)..."
            "$PYTHON" analysis5.py --cache fig2_top_ab --workers "$CPUS"
            ;;
        2)
            echo "[Array Task 2] Regenerating Figure 2 Top-Row Panel (c) cache (GPT-2 data switch)..."
            "$PYTHON" analysis5.py --cache fig2_top_c --workers "$CPUS"
            ;;
        3)
            echo "[Array Task 3] Regenerating Figure 2 Bottom-Row (a, b) caches (1-layer)..."
            "$PYTHON" analysis5.py --cache fig2_bottom_ab --workers "$CPUS"
            ;;
        4)
            echo "[Array Task 4] Regenerating Figure 2 Bottom-Row Panel (c) cache (1-layer data switch)..."
            "$PYTHON" analysis5.py --cache fig2_bottom_c --workers "$CPUS"
            ;;
        *)
            echo "Unknown SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID (expected 1-4)" >&2
            exit 1
            ;;
    esac
    echo "=========================================================="
    echo "Array task $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
    echo "=========================================================="
    exit 0
fi

# If target is explicitly passed via CLI argument
TARGET="${1:-parallel}"

if [ "$TARGET" = "parallel" ] || [ "$TARGET" = "--parallel" ]; then
    echo "Running Figure 2 cache regeneration in parallel using $CPUS CPUs..."
    "$PYTHON" analysis5.py --cache fig2 --cache_parallel --workers "$CPUS"
    
    echo "Rendering final publication-quality Figure 2..."
    "$PYTHON" -c '
import analysis5
import matplotlib.pyplot as plt

fig = analysis5.create_figure2(force_regenerate_cache=False)
for ext in ["png", "pdf", "eps"]:
    out = f"./figs_camera_ready/Figure2.{ext}"
    if ext == "png":
        fig.savefig(out, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}", flush=True)
plt.close(fig)
'
elif [ "$TARGET" = "sequential" ]; then
    echo "Running Figure 2 cache regeneration sequentially using $CPUS CPUs per task..."
    "$PYTHON" analysis5.py --cache fig2 --workers "$CPUS"
    "$PYTHON" analysis5.py --figure 2 --workers "$CPUS"
else
    echo "Regenerating specific cache target: $TARGET using $CPUS CPUs..."
    "$PYTHON" analysis5.py --cache "$TARGET" --workers "$CPUS"
fi

echo "=========================================================="
echo "Cache processing finished at $(date)"
echo "=========================================================="
