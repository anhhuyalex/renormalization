#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --partition=della-gpu
#SBATCH --output imagenet-%J.log
#SBATCH --gres=a100:1



# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
source activate renormalization
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# sbatch hipp.sh 2 && sbatch hipp.sh 4 && sbatch hipp.sh 8 && sbatch hipp.sh 12 && sbatch hipp.sh 20 && sbatch hipp.sh 30 && sbatch hipp.sh 40 
# for num_inputs in {2..60..2}; do sbatch hipp.sh $num_inputs; done
# for num_inputs in {2..60..2}; do for first_layer_l1_regularize in 0.0 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01; do sbatch hipp.sh $num_inputs $first_layer_l1_regularize; done; done

# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm --model_name attn --pixel_shuffled
# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm_freezeconv --freeze_epoch 0 --model_name vgg11
# python -u exp.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/renorm_freezeconv --model_name cnn --freeze_conv

# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 5 --num_examples 50000 --model_name mlp_small --random_coefs True --input_strategy random  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize $2 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 6 --num_examples 50000 --model_name mlp_small --random_coefs True --input_strategy random  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize $2 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 4 --num_examples 50000 --model_name mlp_small_repeat --random_coefs True --input_strategy repeat  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize $2 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 7 --num_examples 50000 --model_name mlp_small_repeat --random_coefs True --input_strategy repeat  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
python -u imagenet.py --model_name resnet18 --num_train_epochs 90 --resume resnet18_rep_1673537483.310051.pkl
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 3 --num_examples 50000 --model_name mlp_small --input_strategy random  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize 0 --tags vary_sample_strategy 
# python -u polynomial_freeze.py --save_dir /scratch/gpfs/qanguyen/poly_freeze --num_inputs $1 --num_inputs_kept 10 --order 3 --num_examples 50000 --model_name mlp_small_silence --random_coefs True --input_strategy random --output_strategy evaluate_at_0 --noise 0.0 --weight_decay 0.0 --lr 1e-3 --is_online False --num_pretrain_epochs 500 --num_train_epochs 1500
