# Few-Shot Regression Experiments

This repo trains regression models on a finite pool of tasks sampled from a rank-frequency distribution, then measures:

- capacity on seen tasks
- interference on seen tasks over training
- generalization to held-out tasks
- few-shot adaptation via mixed fine-tuning batches and an MSE threshold

The main entrypoint is [main.py](/jukebox/norman/qanguyen/renormalization/few_shot/main.py:1). The cluster launch example is [main.sh](/jukebox/norman/qanguyen/renormalization/few_shot/main.sh:1).

## Quick Start

Choose the Python environment used on your machine:

```bash
source ~/.bashrc
python=/scratch/gpfs/KNORMAN/qanguyen/learning_to_learn/l2l/bin/python
```

If you are not on Della, use the alternate path already encoded in `main.sh`.

### Run a GPT experiment

```bash
$python -u main.py \
  --savedir ./outputs \
  --fileprefix gpt_quickstart \
  --arch gpt \
  --seed 0 \
  --batch-size 128 \
  --optimizer Adam \
  --lr 1e-3 \
  --len_context 100 \
  --input_dim 16 \
  --noise_std 0.1 \
  --K 1000 \
  --task_distribution zipf \
  --zipf_alpha 1.0 \
  --num_iters 50000 \
  --eval_every 1000 \
  --num_eval_seen_tasks 256 \
  --num_eval_heldout_tasks 256 \
  --num_rank_buckets 5 \
  --num_hidden_features 16 \
  --num_heads 4 \
  --num_layers 4 \
  --fewshot_num_batches 50 \
  --fewshot_mix_probability 0.5 \
  --fewshot_mse_threshold 0.05 \
  --fewshot_eval_num_tasks 64 \
  --fewshot_lr 5e-4 \
  --fewshot_batch_size 128 \
  --fewshot_source heldout
```

### Run a OneLayerAttention experiment

```bash
$python -u main.py \
  --savedir ./outputs \
  --fileprefix onelayer_quickstart \
  --arch OneLayerAttention \
  --seed 0 \
  --batch-size 128 \
  --optimizer Adam \
  --lr 1e-3 \
  --len_context 100 \
  --input_dim 16 \
  --noise_std 0.1 \
  --K 1000 \
  --task_distribution zipf \
  --zipf_alpha 1.0 \
  --num_iters 50000 \
  --eval_every 1000 \
  --num_eval_seen_tasks 256 \
  --num_eval_heldout_tasks 256 \
  --num_rank_buckets 5 \
  --num_hidden_features 16 \
  --num_heads 4 \
  --num_layers 4 \
  --fewshot_num_batches 50 \
  --fewshot_mix_probability 0.5 \
  --fewshot_mse_threshold 0.05 \
  --fewshot_eval_num_tasks 64 \
  --fewshot_lr 5e-4 \
  --fewshot_batch_size 128 \
  --fewshot_source heldout
```

The output pickle is written to:

```text
<savedir>/cache/<fileprefix>/<fileprefix>_... .pkl
```

It stores:

- parsed args
- parameter count
- pretraining logs
- capacity metrics
- held-out generalization metrics
- interference histories
- few-shot adaptation trace

## Important Flags

- `--arch`: `gpt` or `OneLayerAttention`
- `--num_hidden_features`: per-head width
- `--num_heads`: number of attention heads
- `--num_layers`: transformer depth for `gpt`, MLP depth after attention for `OneLayerAttention`
- `--K`: number of tasks in the finite training pool
- `--task_distribution`: `uniform` or `zipf`
- `--zipf_alpha`: Zipf skew, default `1.0`
- `--fewshot_mix_probability`: fraction of adaptation batches drawn from the fine-tuning stream in expectation
- `--fewshot_mse_threshold`: threshold used to declare few-shot success

## Sweep Model Size

To study the effect of model size, vary `--num_hidden_features`, `--num_heads`, and `--num_layers`. The exact parameter count is logged in the output record for every run.

### Local shell sweep

This is the simplest way to launch multiple runs with different model sizes:

```bash
source ~/.bashrc
python=/scratch/gpfs/KNORMAN/qanguyen/learning_to_learn/l2l/bin/python

for arch in gpt OneLayerAttention; do
  for num_heads in 1 2 4 8; do
    for num_layers in 1 2 4 8; do
      for hidden in 8 16 32; do
        $python -u main.py \
          --savedir ./outputs \
          --fileprefix ${arch}_h${num_heads}_L${num_layers}_d${hidden} \
          --arch ${arch} \
          --seed 0 \
          --batch-size 128 \
          --optimizer Adam \
          --lr 1e-3 \
          --len_context 100 \
          --input_dim 16 \
          --noise_std 0.1 \
          --K 1000 \
          --task_distribution zipf \
          --zipf_alpha 1.0 \
          --num_iters 50000 \
          --eval_every 1000 \
          --num_eval_seen_tasks 256 \
          --num_eval_heldout_tasks 256 \
          --num_rank_buckets 5 \
          --num_hidden_features ${hidden} \
          --num_heads ${num_heads} \
          --num_layers ${num_layers} \
          --fewshot_num_batches 50 \
          --fewshot_mix_probability 0.5 \
          --fewshot_mse_threshold 0.05 \
          --fewshot_eval_num_tasks 64 \
          --fewshot_lr 5e-4 \
          --fewshot_batch_size 128 \
          --fewshot_source heldout
      done
    done
  done
done
```

### SLURM array sweep

`main.sh` already demonstrates a basic array-job pattern for a GPT sweep over `num_heads` and `num_layers`.

Submit it with:

```bash
sbatch main.sh
```

To sweep `OneLayerAttention`, copy `main.sh` or change this line:

```bash
--arch gpt
```

to:

```bash
--arch OneLayerAttention
```

You can also extend the array indexing logic to include `num_hidden_features`. For example:

```bash
num_heads_vals=(1 2 4 8)
num_layers_vals=(1 2 4 8)
hidden_vals=(8 16 32)

idx=${SLURM_ARRAY_TASK_ID:-0}
num_heads=${num_heads_vals[$(( idx % ${#num_heads_vals[@]} ))]}
num_layers=${num_layers_vals[$(( (idx / ${#num_heads_vals[@]}) % ${#num_layers_vals[@]} ))]}
hidden=${hidden_vals[$(( idx / (${#num_heads_vals[@]} * ${#num_layers_vals[@]}) ))]}
```

Then pass:

```bash
--num_hidden_features "$hidden"
```

and set the SLURM array size to cover the full Cartesian product.

## Notes

- `gpt` and `OneLayerAttention` share the same regression task pipeline and evaluation code.
- The few-shot metric is the number of fine-tuning presentations needed before fine-tuning-set MSE falls below `--fewshot_mse_threshold`.
- If the threshold is never crossed within `--fewshot_num_batches`, the run records that failure along with the final MSE.
