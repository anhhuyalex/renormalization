#!/bin/bash
#SBATCH --job-name=regression-fewshot
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=30G
#SBATCH --time=72:00:00
#SBATCH -o slurms/%j-regression.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-35

source ~/.bashrc

hostname="$(hostname)"
if [[ "$hostname" == *della* ]]; then
  python="/scratch/gpfs/KNORMAN/qanguyen/learning_to_learn/l2l/bin/python"
  venv_activate="/scratch/gpfs/KNORMAN/qanguyen/learning_to_learn/l2l/bin/activate"
  savedir="/scratch/gpfs/KNORMAN/qanguyen/gautam/fewshot"
else
  python="/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri/bin/python"
  venv_activate=""
  savedir="/scratch/qanguyen/gautam/fewshot"
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

num_heads_vals=(1 2 4 8 12 16)
num_hidden_features_vals=(7 14 30 60 90 120)
idx=${SLURM_ARRAY_TASK_ID:-0}
num_heads=${num_heads_vals[$(( idx % ${#num_heads_vals[@]} ))]}
num_hidden_features=${num_hidden_features_vals[$(( idx / ${#num_heads_vals[@]} ))]}
# Set experiment_name to "gpt" or "OneLayer"
K=100000
experiment_name="gpt_K_${K}"
fileprefix="regression_forget_${experiment_name}_zipf_K_${K}"

$python -u main.py \
  --savedir "$savedir" \
  --experiment_name "$experiment_name" \
  --fileprefix "$fileprefix" \
  --seed "$idx" \
  --batch-size 256 \
  --optimizer Adam \
  --lr 1e-3 \
  --wd 0.0 \
  --gpt_bias True \
  --num_hidden_features "$num_hidden_features" \
  --num_heads "$num_heads" \
  --num_layers 12 \
  --len_context 32 \
  --input_dim 16 \
  --noise_std 0.5 \
  --K "$K" \
  --task_distribution zipf \
  --zipf_alpha 1.0 \
  --num_iters 500000 \
  --eval_every 10000 \
  --num_eval_seen_tasks 100000 \
  --eval_task_stride 10 \
  --num_eval_heldout_tasks 256 \
  --num_rank_buckets 5 \
  --fewshot_num_batches 1000 \
  --steps_per_epoch 10000 \
  --num_finetune_tasks 50 \
  --fewshot_lr 5e-4 \
  --fewshot_batch_size 256 \
  --rank_forget_sequence 100 \
  --forget_sequence_eval 50
