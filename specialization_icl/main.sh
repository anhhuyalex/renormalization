#!/bin/bash
#SBATCH --job-name=specialization_icl
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output l2l-%J.log
#SBATCH -o slurms/%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
# source activate renormalization
source ../../learning_to_learn/l2l/bin/activate


# wandb login --relogin --host=https://stability.wandb.io
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# srun --pty -p mig -c 1 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
jupytext --to py main.ipynb
# jupyter nbconvert --to python main.ipynb

# for i in {0..20}; do sbatch --array=0-95 main.sh 0.01; done
# i_vals=(0.0 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99) # len 30
D_visible_frac_vals=(8 7 6 5 4 3 2 1) # len 10
# j_vals=(10 50 100 500 1000 10000) # 6
len_context_vals=(16 8 4) # 3
# i_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
# j_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
D_visible_frac=${D_visible_frac_vals[$SLURM_ARRAY_TASK_ID % ${#D_visible_frac_vals[@]}]}
len_context=${len_context_vals[$SLURM_ARRAY_TASK_ID / ${#D_visible_frac_vals[@]}]}


echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID, i = $i, j = $j"
gpt_bias=True
lr=1e-4
optimizer="Adam"
epochs=500

# for run in {0..0}; do WANDB_MODE=offline python -u main.py --data ./cache --fileprefix transformer1layer  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 1e-3 --wd 1e-10  --epochs 200 --arch causal_transformer_embed --num_hidden_features 256 --num_layers 1 --len_context $j --K 100000 --D_sum 50 --D_visible_frac $i --sigma_xi 0.0 --coarse_graining abstop --wandb_log --wandb_project renormalization --wandb_group_name linreg_oct14_specgen ; done 
for run in {0..0}; do WANDB_MODE=offline ../../learning_to_learn/l2l/bin/python -u main.py --data ./cache --fileprefix no_layernorm_input_opt_${optimizer}_lr_${lr}_gpt_bias_${gpt_bias}_epochs_${epochs}  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --epochs ${epochs} --arch gpt --gpt_bias ${gpt_bias} --num_hidden_features 128 --num_layers 8 --len_context $len_context --K 1048576 --D_sum 8 --D_visible_frac $D_visible_frac --sigma_xi 0.0 --coarse_graining abstop --wandb_log --wandb_project renormalization --wandb_group_name linreg_nov4_specgen_bias_${gpt_bias}  ; done 
 