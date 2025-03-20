#!/bin/bash
#SBATCH --job-name=specialization_icl
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=36:00:00
#SBATCH --output l2l-%J.log
#SBATCH -o slurms/abstop%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
# source activate /mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri
# conda activate renormalization
source ../../learning_to_learn/l2l/bin/activate
# source ~/.bashrc
# conda activate /mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri

# wandb login --relogin --host=https://stability.wandb.io
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# srun --pty -p mig -c 1 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# jupytext --to py main.ipynb

# for i in {0..20}; do sbatch --array=0-95 main.sh 0.01; done
# i_vals=(0.0 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99) # len 30
# D_visible_frac_vals=(1 3 6 9 12 15 18 21 24 27 30 32) # 12
# D_visible_frac_vals=(32) # 12
# j_vals=(10 50 100 500 1000 10000) # 6
# i_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
# j_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
# D_visible_frac=${D_visible_frac_vals[$SLURM_ARRAY_TASK_ID % ${#D_visible_frac_vals[@]}]}
# len_context=${len_context_vals[$SLURM_ARRAY_TASK_ID / ${#D_visible_frac_vals[@]}]}
 
len_context=200 # 6
gpt_bias=True
lr=1e-5
optimizer="Adam"
epochs=10000
D_sum=32
K=1048576
data_scale=1.0
resume="./cache/linreg_nov19_specgen_bias_Dsum__scheduler_None_K_1024_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_1024_D_64_L_100_hidden_128_coarse_abstop_1732079333.0764203.pkl" 
coarse_graining="shrink_norm"
fileprefix="feb_15" # "jan17_2pm"
wandb_group_name="linreg_mar12_aniso_datascale_${data_scale}_lr_${lr}_epochs_${epochs}_nopermute_forcesnr_1"
scheduler="None"
# coarse_graining="aniso_highvariance_shift"
# input_covariance="anisotropic"
# echo $resume
echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID, coarse_graining $coarse_graining, resume $resume"
# for run in {0..0}; do WANDB_MODE=offline python -u main.py --data ./cache --fileprefix transformer1layer  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 1e-3 --wd 1e-10  --epochs 200 --arch causal_transformer_embed --num_hidden_features 256 --num_layers 1 --len_context $j --K 100000 --D_sum 50 --D_visible_frac $i --sigma_xi 0.0 --coarse_graining abstop --wandb_log --wandb_project renormalization --wandb_group_name linreg_oct14_specgen ; done 
# for run in {0..0}; do WANDB_MODE=offline ../../learning_to_learn/l2l/bin/python -u main.py --data ./cache --fileprefix no_layernorm_input_opt_${optimizer}_lr_${lr}_gpt_bias_${gpt_bias}_epochs_${epochs}_visible_${D_visible_frac}  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 256 --optimizer ${optimizer} --scheduler LinearWarmUpCosineDecay --lr ${lr} --wd 0.0  --epochs ${epochs} --arch gpt --gpt_bias ${gpt_bias} --num_hidden_features 128 --num_layers 8 --len_context $len_context --K 1048576 --D_sum 32 --D_visible_frac $D_visible_frac --sigma_xi 0.5 --coarse_graining abstop --no-wandb_log --wandb_project renormalization --wandb_group_name linreg_nov16_specgen_bias_Dsum_32_scheduler_LinearWarmUpCosineDecay  ; done 
# for run in {0..0}; do WANDB_MODE=offline ../../learning_to_learn/l2l/bin/python -u main.py --data ./cache --fileprefix no_layernorm_input_opt_${optimizer}_lr_${lr}_gpt_bias_${gpt_bias}_epochs_${epochs}_visible_${D_visible_frac}  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 256 --optimizer ${optimizer} --scheduler None --lr ${lr} --wd 0.0  --epochs ${epochs} --arch gpt --gpt_bias ${gpt_bias} --num_hidden_features 128 --num_layers 8 --len_context $len_context --K ${K} --D_sum ${D_sum} --D_visible_frac $D_visible_frac --sigma_xi 0.5 --coarse_graining abstop --no-wandb_log --wandb_project renormalization --wandb_group_name linreg_nov19_specgen_bias_Dsum_${Dsum}_scheduler_None_K_${K}  ; done 
# jupytext --to py main.ipynb && for run in {0..0}; do WANDB_MODE=offline python -u main.py --data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --scheduler ${scheduler} --lr ${lr} --wd 0.0  --epochs ${epochs} --arch gpt --gpt_bias ${gpt_bias} --num_hidden_features 128 --num_layers 8 --len_context ${len_context} --K ${K} --D_sum ${D_sum} --sigma_xi 0.5 --data_scale ${data_scale} --coarse_graining abstop --no-wandb_log --wandb_project renormalization --is_iso False --wandb_group_name ${wandb_group_name}  ; done 
# resume="./cache/linreg_nov16_specgen_bias_Dsum_32_scheduler_Triangle_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_1048576_D_32_L_200_hidden_128_coarse_abstop_1731797026.0355098.pkl"
# resume="./cache/linreg_nov16_specgen_bias_Dsum_32_scheduler_LinearWarmUpCosineDecay_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_1048576_D_32_L_200_hidden_128_coarse_abstop_1731798789.0959225.pkl"
# resume="./cache/linreg_nov19_specgen_bias_Dsum__scheduler_None_K_32_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_32_D_64_L_100_hidden_128_coarse_abstop_1732079383.3274248.pkl"

# resume="./cache/linreg_nov19_specgen_bias_Dsum__scheduler_None_K_1048576_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_1048576_D_64_L_100_hidden_128_coarse_abstop_1732079442.9435685.pkl"
# resume="./cache/linreg_nov19_specgen_bias_Dsum__scheduler_None_K_32768_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_32768_D_64_L_100_hidden_128_coarse_abstop_1732079299.9278684.pkl"
# resume="./cache/linreg_nov19_specgen_bias_Dsum__scheduler_None_K_32_no_layernorm_input_opt_Adam_lr_1e-4_gpt_bias_True_epochs_500_visible_32_K_32_D_64_L_100_hidden_128_coarse_abstop_1732079383.3274248.pkl"
jupytext --to py analysis2.ipynb && WANDB_MODE=offline python -u analysis2.py --data ./cache --fileprefix jan29_2pm  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 512 --optimizer Adam --lr 0.01 --wd 0.0  --epochs 1 --arch gpt --gpt_bias True --num_hidden_features 128 --num_layers 8 --len_context ${len_context} --K 1048576 --D_sum 64 --D_visible_frac 1 --sigma_xi 0.5   --no-wandb_log --wandb_project renormalization --wandb_group_name linreg_nov13_specgen_bias_Dsum_32 
# SLURM_ARRAY_TASK_ID=5
# which python
# jupytext --to py analysis_scale_alignment_heatmap.ipynb && WANDB_MODE=offline python -u analysis_scale_alignment_heatmap.py --data ./cache --fileprefix ${fileprefix}  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 512 --optimizer Adam --lr 0.01 --wd 0.0  --epochs 1 --arch gpt --gpt_bias True --num_hidden_features 128 --num_layers 8 --len_context ${len_context} --K 1048576 --D_sum 32 --D_visible_frac 1 --sigma_xi 0.5   --no-wandb_log --wandb_project renormalization --wandb_group_name linreg_nov13_specgen_bias_Dsum_32 
# jupytext --to py analysis2.ipynb && coarse_graining="aniso_highvariance_shift" input_covariance="True" WANDB_MODE=offline python -u analysis2.py --data ./cache --fileprefix test  --SLURM_ARRAY_TASK_ID 0 --batch-size 256 --optimizer Adam --lr 0.01 --wd 0.0  --epochs 1 --arch gpt --gpt_bias True --num_hidden_features 128 --num_layers 8 --len_context ${len_context} --K 1048576 --D_sum 32 --D_visible_frac 1 --sigma_xi 0.5 --coarse_graining ${coarse_graining} --no-wandb_log --wandb_project renormalization --wandb_group_name linreg_nov13_specgen_bias_Dsum_32 --input_covariance=${input_covariance} --resume ${resume}