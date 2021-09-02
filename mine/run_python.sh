#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
source activate pytorch_resnet
python_script=$1
job_num=$2
job_type=$3
lr=$4
# srun --pty -p gpu -c 4 -t 2:00:00 --gres=gpu:1 --mem-per-cpu=10G --cpus-per-task=4 bash
DEBUG=1 python -u $python_script $job_num $job_type --hidden_size 1000 --num_hidden_layers 6 --lr $lr
