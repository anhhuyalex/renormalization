#!/bin/bash
#SBATCH --job-name=memo
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=30G
#SBATCH --time=36:00:00
#SBATCH -o slurms/%j-with-projection-layer-with-residual-mlp.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-99
#SBATCH --partition=mig
# source activate renormalization
source ~/.bashrc
source ../../learning_to_learn/l2l/bin/activate
# cd /jukebox/norman/qanguyen/patdiff_seq
# conda activate /mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri

# wandb login --relogin --host=https://stability.wandb.io
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# srun --pty -p mig -c 1 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# jupytext --to py main.ipynb

# for i in {0..20}; do sbatch --array=0-95 main.sh 0.01; done
# i_vals=(0.0 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99) # len 30
# D_visible_frac_vals=(1 3 6 9 12 15 18 21 24 27 30 32) # 12
# j_vals=(10 50 100 500 1000 10000) # 6
len_context_vals=(100) # 6
# i_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
# j_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
# SLURM_ARRAY_TASK_ID=0
len_context=100 #${len_context_vals[$SLURM_ARRAY_TASK_ID / ${#D_visible_frac_vals[@]}]}


echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID, i = $i, j = $j"
gpt_bias=True
lr=1e-3
optimizer="Adam"
num_iters=5000000
K=1000
sequence_sampling_distribution="uniform"
num_mlp_layers=3
# SLURM_ARRAY_TASK_ID=0
# memo_may3_zipf_num_heads_8_num_layers_12_zero_all_attn_weights
# memo_may3_zipf_num_heads_24_num_layers_36_lr_1e-4_zero_all_attn_weights
# memo_may3_zipf_num_heads_8_num_layers_12_zero_all_attn_except_cproj_weights
# memo_may3_zipf_num_heads_24_num_layers_36_lr_1e-4_zero_all_attn_except_cproj_weights
# wandb_group_name="memo_aug15_zipf_onelayerattention_lr_1e-3"
wandb_group_name="memo_nov10_zipf_gpt2_vary_num_hidden_features_num_heads_resample"
# wandb_group_name="memo_jul4_zipf_onelayerattention_lr_1e-3_swapmlp_eval"
# wandb_group_name="memo_aug2_zipf_onelayerattention_lr_1e-3_swapmlp_eval"
jupytext --to py main.ipynb && for run in {0..0}; do WANDB_MODE=offline python -u main.py --data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --num_iters ${num_iters} --arch gpt --gpt_bias ${gpt_bias}  --num_mlp_layers ${num_mlp_layers} --len_context ${len_context} --K ${K} --sequence_sampling_distribution ${sequence_sampling_distribution} --no-wandb_log --wandb_project l2l --wandb_group_name  ${wandb_group_name}  ; done 
# jupytext --to py main_fork.ipynb  && for run in {0..0}; do WANDB_MODE=offline python -u main_fork.py --data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --num_iters ${num_iters} --arch gpt --gpt_bias ${gpt_bias} --num_mlp_layers ${num_mlp_layers} --len_context ${len_context} --K ${K} --sequence_sampling_distribution ${sequence_sampling_distribution} --no-wandb_log --wandb_project l2l --wandb_group_name  ${wandb_group_name}  ; done 
# jupytext --to py main_freeze.ipynb  && for run in {0..0}; do WANDB_MODE=offline python -u main_freeze.py --data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --num_iters ${num_iters} --arch gpt --gpt_bias ${gpt_bias} --num_mlp_layers ${num_mlp_layers} --len_context ${len_context} --K ${K} --sequence_sampling_distribution ${sequence_sampling_distribution} --no-wandb_log --wandb_project l2l --wandb_group_name  ${wandb_group_name}  ; done 
# jupytext --to py main_freeze.ipynb  && for run in {0..0}; do WANDB_MODE=offline python -u main_freeze.py --data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --num_iters ${num_iters} --arch gpt --gpt_bias ${gpt_bias} --num_mlp_layers ${num_mlp_layers} --len_context ${len_context} --K ${K} --sequence_sampling_distribution ${sequence_sampling_distribution} --no-wandb_log --wandb_project l2l --wandb_group_name  ${wandb_group_name}  ; done 


 