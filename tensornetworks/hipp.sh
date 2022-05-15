#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output renorm-alexnet-%J.log


# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
source activate renormalization
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# python -u oneshot_cls.py --config_path "definitions.aha_config_hopfield.aha_hopfield_cfg" --model_name Hopfield_TargetLabel_Attention
# python -u oneshot_cls.py --config_path "definitions.aha_config_hopfield.aha_hopfield_label_cfg" --model_name Hopfield_LabelHopfield_Attention_onehidlayer
# for i in {20..100..20}; do sbatch hipp.sh $i; done
# python -u oneshot_cls.py --config_path definitions.aha_config_theremintest.aha_theremintest_cfg --model_name CLS --numtestitems $1 --ca3_num_units 2400 --end_training_threshold 0.34
# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm --model_name attn --pixel_shuffled
python -u exp.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/renorm --model_name alexnet