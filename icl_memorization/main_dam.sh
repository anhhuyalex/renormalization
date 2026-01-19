#!/bin/bash
#SBATCH --job-name=memo
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=72:00:00
#SBATCH -o slurms/%j-heads.out
#SBATCH --gres=gpu:1
# SBATCH --array=0-80%20
# SBATCH --partition=mig
# source activate renormalization
source ~/.bashrc
# source ../../learning_to_learn/l2l/bin/activate
# cd /jukebox/norman/qanguyen/patdiff_seq
conda activate /mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri

savedir="/scratch/qanguyen/gautam"

# Parameter Grid
etas=(1 10 100)
lrs=(1e-3 1e-2 1e-1)
init_stds=(1e-2 1e-1 1e0)
Hs=(1 5 10 30 100)

# Decode SLURM_ARRAY_TASK_ID
# Total combos = 3 * 3 * 3 * 5 = 135
# Order calculation: H changes fastest
idx=$SLURM_ARRAY_TASK_ID

num_Hs=${#Hs[@]}
i_H=$((idx % num_Hs))
idx=$((idx / num_Hs))

num_init_stds=${#init_stds[@]}
i_std=$((idx % num_init_stds))
idx=$((idx / num_init_stds))

num_lrs=${#lrs[@]}
i_lr=$((idx % num_lrs))
idx=$((idx / num_lrs))

num_etas=${#etas[@]}
i_eta=$((idx % num_etas))

# Retrieve values
eta=${etas[$i_eta]}
lr=${lrs[$i_lr]}
init_std=${init_stds[$i_std]}
H=${Hs[$i_H]}

echo "Running with eta=$eta, lr=$lr, init_std=$init_std, H=$H"

# Construct specific prefix

# Run script
H=$1
if [ -z "$H" ]; then
  echo "Error: missing H. Usage: $0 <H>, where H in {100, 30, 10, 5, 1}"
  exit 1
fi
if [ "$H" -eq 100 ]; then
  eta=10.0
  lr=1e-2
  init_std=1e-1
  H=100
  
elif [ "$H" -eq 30 ]; then
  eta=10.0
  lr=1e-2
  init_std=1e-2
  H=30
elif [ "$H" -eq 10 ]; then
  eta=10.0
  lr=1e-2
  init_std=1e-2
  H=10
elif [ "$H" -eq 5 ]; then
  eta=10.0
  lr=1e-2
  init_std=1e-1
  H=5
elif [ "$H" -eq 1 ]; then
  eta=10.0
  lr=1e-3
  init_std=1e-2
  H=1
else
  echo "Error: unsupported H=$H. Expected H in {100, 30, 10, 5, 1}"
  exit 1
fi

prefix="dam_jan19_largeK_no_adapt_H_${H}"
num_steps=10000
K=10000
batch_size=1000
M=500000
N=100
python -u main_dam.py --savedir $savedir --prefix $prefix \
    --eta $eta --lr $lr --INIT_STD $init_std --H $H --NUM_STEPS $num_steps \
    --BATCH_SIZE $batch_size --K $K --M $M --N $N