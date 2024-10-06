#!/bin/bash
#SBATCH --job-name=specialization_icl
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output l2l-%J.log
#SBATCH -o slurms/%j.out
#SBATCH --partition=mig
#SBATCH --gres=gpu:1

# source activate renormalization
source ../../learning_to_learn/l2l/bin/activate


# wandb login --relogin --host=https://stability.wandb.io
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# srun --pty -p mig -c 1 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=20G bash
# jupytext --to py main.ipynb
jupyter nbconvert --to python main.ipynb

# for i in {0..20}; do sbatch --array=0-95 main.sh 0.01; done
i_vals=(10 100 1000 10000 20000 60000) # 6
j_vals=(10 100 1000 10000 20000 60000) # 6
# i_vals=(10 100 500 1000 2000 5000 7500 10000) # 6
# j_vals=(10 100 500 1000 2000 5000 7500 10000) # 6

i=${i_vals[$SLURM_ARRAY_TASK_ID / ${#j_vals[@]}]}
j=${j_vals[$SLURM_ARRAY_TASK_ID % ${#j_vals[@]}]}

echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID, i = $i, j = $j"
 
for i in {0..0}; do WANDB_MODE=offline python -u main.py --data ./cache --fileprefix transformer1layer  --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 0.01 --wd 1e-10  --epochs 200 --arch causal_transformer_embed --num_hidden_features 256 --len_context 100 --coarse_graining abstop --wandb_log --wandb_project renormalization --wandb_group_name linreg_oct5_specgen ; done 
 