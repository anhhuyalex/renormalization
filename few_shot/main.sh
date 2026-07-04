#!/bin/bash
#SBATCH --job-name=regression-fewshot
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=30G
#SBATCH --time=72:00:00
#SBATCH -o slurms/%j-regression.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

set -euo pipefail

source ~/.bashrc

hostname="$(hostname)"
if [[ "$hostname" == *della* ]]; then
  python="/scratch/gpfs/KNORMAN/qanguyen/learning_to_learn/l2l/bin/python"
  venv_activate="/scratch/gpfs/KNORMAN/qanguyen/learning_to_learn/l2l/bin/activate"
  savedir="/scratch/gpfs/KNORMAN/qanguyen/gautam"
else
  python="/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri/bin/python"
  venv_activate=""
  savedir="/scratch/qanguyen/gautam"
fi

if [[ ! -x "$python" ]]; then
  echo "Expected python not found at: $python" >&2
  exit 1
fi

if [[ -n "$venv_activate" && -f "$venv_activate" ]]; then
  # shellcheck disable=SC1090
  source "$venv_activate"
fi

print_python_diagnostics() {
  echo "===== python diagnostics ====="
  echo "date: $(date)"
  echo "hostname: $(hostname)"
  echo "PATH: $PATH"
  echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
  echo "python variable: $python"
  echo "\$python --version: $($python --version 2>&1 || true)"
  echo "\$python executable: $($python -c 'import sys; print(sys.executable)' 2>&1 || true)"
  echo "\$python torch check: $($python -c 'import torch; print(torch.__version__)' 2>&1 || true)"
  echo "=============================="
}

print_python_diagnostics

num_heads_vals=(1 2 4 8)
num_layers_vals=(1 2 4 8)
idx=${SLURM_ARRAY_TASK_ID:-0}
num_heads=${num_heads_vals[$(( idx % ${#num_heads_vals[@]} ))]}
num_layers=${num_layers_vals[$(( idx / ${#num_heads_vals[@]} ))]}

fileprefix="regression_zipf_alpha1"

$python -u main.py \
  --savedir "$savedir" \
  --fileprefix "$fileprefix" \
  --seed "$idx" \
  --batch-size 256 \
  --optimizer Adam \
  --lr 1e-3 \
  --wd 0.0 \
  --arch gpt \
  --gpt_bias True \
  --num_hidden_features 16 \
  --num_heads "$num_heads" \
  --num_layers "$num_layers" \
  --len_context 100 \
  --input_dim 16 \
  --noise_std 0.1 \
  --K 1000 \
  --task_distribution zipf \
  --zipf_alpha 1.0 \
  --num_iters 50000 \
  --eval_every 1000 \
  --num_eval_seen_tasks 256 \
  --num_eval_heldout_tasks 256 \
  --num_rank_buckets 5 \
  --fewshot_num_batches 50 \
  --fewshot_mix_probability 0.5 \
  --fewshot_mse_threshold 0.05 \
  --fewshot_eval_num_tasks 64 \
  --fewshot_lr 5e-4 \
  --fewshot_batch_size 256 \
  --fewshot_source heldout
