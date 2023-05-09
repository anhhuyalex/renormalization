#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --time=12:00:00
#SBATCH --partition=della-gpu
#SBATCH --output imagenet-%J.log
#SBATCH --gres=gpu:1



# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
source activate renormalization
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# sbatch hipp.sh 2 && sbatch hipp.sh 4 && sbatch hipp.sh 8 && sbatch hipp.sh 12 && sbatch hipp.sh 20 && sbatch hipp.sh 30 && sbatch hipp.sh 40 
# for num_inputs in {2..60..2}; do sbatch hipp.sh $num_inputs; done
# for num_inputs in {2..60..2}; do for first_layer_l1_regularize in 0.0 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01; do sbatch hipp.sh $num_inputs $first_layer_l1_regularize; done; done

# for data_rescale in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1;  do sbatch hipp.sh $data_rescale; done 

# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 5 --num_examples 50000 --model_name mlp_small --random_coefs True --input_strategy random  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize $2 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 6 --num_examples 50000 --model_name mlp_small --random_coefs True --input_strategy random  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize $2 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 4 --num_examples 50000 --model_name mlp_small_repeat --random_coefs True --input_strategy repeat  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --first_layer_l1_regularize $2 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 7 --num_examples 50000 --model_name mlp_small_repeat --random_coefs True --input_strategy repeat  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --weight_decay 0.0 --lr 5e-3 --tags vary_l1_regularizer_vary_num_inputs_vary_order 
# python -u imagenet.py --model_name resnet18 --num_train_epochs 90 --resume resnet18_rep_1673614260.528039.pkl
# python -u imagenet_devel.py --model_name resnet18 --num_train_epochs 90 --data_rescale $1 #--resume resnet18_rep_1673625654.48548.pkl
# python imagenet_devel.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization --data_rescale $1 --epochs 36 --scheduler_step_size 12 --zero_out all_except_center
# python imagenet_devel.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 36 --scheduler_step_size 12  --image_transform_loader TileImagenet --tiling_imagenet  $1 --tiling_orientation_ablation True
# for tiling in "1,1" "1,2" "1,3" "2,1" "2,2" "2,3" "3,1" "3,2" "3,3"; do sbatch hipp.sh $tiling; done
# for tiling in "3,4" "3,5" "3,6" "4,3" "4,4" "4,4" "5,3" "5,4" "5,5"; do sbatch hipp.sh $tiling; done
# for tiling in 1 2 3 4 5 ; do sbatch hipp.sh $tiling; done
# python imagenet.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 24 --scheduler_step_size 8  --image_transform_loader TileImagenet --tiling_imagenet  $1 --tiling_orientation_ablation no_ablation --gaussian_blur True --max_sigma 4.0 --fileprefix AUGSTILINGmar23
# python imagenet.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 24 --scheduler_step_size 8  --image_transform_loader TileImagenet --tiling_imagenet  $1 --tiling_orientation_ablation orientation --gaussian_blur False --max_sigma 0.0 --fileprefix AUGSTILINGmar23 
i_vals=(1 2 3 4 5 6 7 8 9)
# i_vals=(0.00001 0.00005 0.0001 0.0005 0.001 0.002 0.005 0.01)
j_vals=(5 10 50 100 500 1000 5000 10000 50000 100000)
i=${i_vals[$SLURM_ARRAY_TASK_ID / ${#j_vals[@]}]}
j=${j_vals[$SLURM_ARRAY_TASK_ID % ${#j_vals[@]}]}
echo "i = $i, j = $j"
# i=$1
# python -u tractable_polynomial.py --save_dir /scratch/gpfs/qanguyen/poly_roots --num_inputs $1 --order 7 --num_examples 50000 --model_name ridge --random_coefs True --input_strategy random  --is_online False --output_strategy evaluate_at_0 --sample_strategy roots --noise 0.0 --tags ridge 

# python imagenet.py -a coarsegrain_attention --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 24 --scheduler_step_size 8 --lr 0.03  --image_transform_loader CoarseGrainImagenet --coarsegrain_blocksize  $i --fileprefix coarsegraining_attention_APR25_lr0.03
python randomfeatures_imagenet.py -a randomfeatures --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 5 --scheduler_step_size 8 --lr 0.01 --image_transform_loader SubsampleImagenet --coarsegrain_blocksize  $i --num_hidden_features $j  --fileprefix randomfeatures_MAY6
# for i in 9 8 7 6 5 4 3 2 1; do sbatch hipp.sh $i; done
# python imagenet_ensemble.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 24 --scheduler_step_size 8  --image_transform_loader TileImagenet --num_models_ensemble $1
# python imagenet_vmap_ensemble.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --epochs 24 --scheduler_step_size 8  --image_transform_loader TileImagenet --num_independent_samples num_models_ensemble --num_models_ensemble $1 --fileprefix Ensemble24EpochsMar19TwoStageAug  --gaussian_blur False --max_sigma 0.0

# for growth_factor in 1.0 1.3 1.6 1.9 2.2 2.5 2.8 3.1;  do sbatch hipp.sh $growth_factor; done 
# python imagenet_devel.py -a resnet18 --dist-url 'tcp://127.0.0.1' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization --data_rescale 0.3 --epochs 36 --scheduler_step_size 12 --zero_out grow_from_center --growth_factor $1
