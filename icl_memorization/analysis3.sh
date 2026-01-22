#!/bin/bash
#SBATCH --job-name=analysis3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1:00:00
#SBATCH --output analysis3-%J.log
#SBATCH -o slurms/%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --array=20-35
# 1-9%10

# Environment setup
source ~/.bashrc
source ../../learning_to_learn/l2l/bin/activate

# In the bash script, get all the folders in the './cache/memo_may26_zipf_onelayerattention_lr_1e-3_vary_num_hidden_features_num_heads/*'
# Then, for each folder, run the analysis3.py script

# Get all the folders in the './cache/memo_may26_zipf_onelayerattention_lr_1e-3_vary_num_hidden_features_num_heads/*'
# Use sort to ensure deterministic ordering
folder=$(ls -d ./cache/memo_may26_zipf_onelayerattention_lr_1e-3_vary_num_hidden_features_num_heads/* | sort | head -n $((SLURM_ARRAY_TASK_ID+1)) | tail -n 1)
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Selected folder: $folder"

# Check if folder exists and is not empty
if [ -z "$folder" ] || [ ! -d "$folder" ]; then
    echo "Error: Folder not found or invalid for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# For each folder, run the analysis3.py script
python -u analysis3.py --folder $folder