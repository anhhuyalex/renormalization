#!/usr/bin/env bash
#SBATCH --job-name=figure2-cache
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization
#SBATCH --output=/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization/slurms/figure2-cache-%j.out

set -euo pipefail

project_dir="/mnt/cup/labs/norman/qanguyen/renormalization/icl_memorization"
cd "$project_dir"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/icl_memorization_mpl}"
mkdir -p "$MPLCONFIGDIR" ./figs_camera_ready

echo "host: $(hostname)"
echo "allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-set}"

/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri/bin/python -c '
import analysis5
import matplotlib.pyplot as plt

fig = analysis5.create_figure2(force_regenerate_cache=True)
fig.savefig("./figs_camera_ready/Figure2.png", dpi=300, bbox_inches="tight")
plt.close(fig)
'
