#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output polynomial-%J.log


# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
source activate renormalization
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash

# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm --model_name attn --pixel_shuffled
# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm_freezeconv --freeze_epoch 0 --model_name vgg11
# python -u exp.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/renorm_freezeconv --model_name cnn --freeze_conv
#python -u exp.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/renorm_quench --fix_permutation --model_name vgg11
python -u polynomial.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/poly --num_inputs 80 --order 1 --model_name mlp 
python -u polynomial.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/poly --num_inputs 80 --order 1 --model_name mlp --random_coefs

python -u polynomial.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/poly --num_inputs 80 --order 5 --model_name mlp 
python -u polynomial.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/poly --num_inputs 80 --order 5 --model_name mlp --random_coefs

python -u polynomial.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/poly --num_inputs 80 --order 10 --model_name mlp 
python -u polynomial.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/poly --num_inputs 80 --order 10 --model_name mlp --random_coefs
