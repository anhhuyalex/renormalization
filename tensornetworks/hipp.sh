#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output polynomial-norm-%J.log


# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
source activate renormalization
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# sbatch hipp.sh 2 && sbatch hipp.sh 4 && sbatch hipp.sh 8 && sbatch hipp.sh 12 && sbatch hipp.sh 20 && sbatch hipp.sh 30 && sbatch hipp.sh 40 
# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm --model_name attn --pixel_shuffled
# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm_freezeconv --freeze_epoch 0 --model_name vgg11
# python -u exp.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/renorm_freezeconv --model_name cnn --freeze_conv

# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_norm --num_inputs 60 --order 3 --num_examples 50000 --model_name mlp_small_batchnorm --random_coefs True --random_inputs True --noise 0.0 --weight_decay 0.0
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_norm --num_inputs 80 --order 3 --num_examples 50000 --model_name mlp_small_batchnorm --random_coefs True --random_inputs True --noise 0.0 --weight_decay 0.0

python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_fix_x --num_inputs $1 --order 3 --num_inputs_kept 8 --num_examples 50000 --model_name mlp_small_silence --random_coefs True --input_strategy random --noise 0.0 --weight_decay 0.0 

# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly1 --num_inputs $2 --order 3 --num_examples 50000 --model_name attention_small --random_coefs True --random_inputs True --noise 0.0 --weight_decay 0.0

# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_norm --num_inputs 50 --order 2 --num_examples 50000 --model_name mlp_large_batchnorm --random_coefs True --random_inputs True --noise 0.0 --weight_decay 0.0

