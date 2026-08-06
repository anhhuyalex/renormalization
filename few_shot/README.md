# Few-Shot Regression Experiments

This repo trains regression models on a finite pool of tasks sampled from a rank-frequency distribution, then measures:

- capacity on seen tasks
- generalization to held-out tasks
- few-shot adaptation via mixed fine-tuning batches and an MSE threshold

The main entrypoint is [main.py](/jukebox/norman/qanguyen/renormalization/few_shot/main.py:1). The cluster launch example is [main.sh](/jukebox/norman/qanguyen/renormalization/few_shot/main.sh:1).

## Training notes

**Few-Shot Adaptation (Fine-Tuning)**
At the end of every epoch (determined by `--steps_per_epoch`), the model undergoes a few-shot fine-tuning (adaptation) experiment to evaluate how quickly it adapts to a specific set of fixed tasks:
- A new optimizer is instantiated for a copy of the model using a dedicated learning rate (`--fewshot_lr`).
- Fine-tuning runs for `--fewshot_num_batches` steps.
- In each step, the model trains on a mixed batch. Each sequence in the batch is sampled from the `--num_finetune_tasks` fixed tasks with probability `--fewshot_mix_probability`, and from the original pretraining pool otherwise.
- The primary metric recorded is the `loss_trace`, a time series matrix tracking the MSE of each fine-tuning task whenever it is presented during the adaptation batches.

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
  --steps_per_epoch 1000 \
  --num_finetune_tasks 50 \
  --fewshot_lr 5e-4 \
  --fewshot_batch_size 128
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
  --steps_per_epoch 1000 \
  --num_finetune_tasks 50 \
  --fewshot_lr 5e-4 \
  --fewshot_batch_size 128
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
- few-shot adaptation trace

## Important Flags

- `--arch`: `gpt` or `OneLayerAttention`
- `--num_hidden_features`: per-head width
- `--num_heads`: number of attention heads
- `--num_layers`: transformer depth for `gpt`, MLP depth after attention for `OneLayerAttention`
- `--K`: number of tasks in the finite training pool
- `--task_distribution`: `uniform` or `zipf`
- `--zipf_alpha`: Zipf skew, default `1.0`
- `--steps_per_epoch`: number of pretraining gradient steps per epoch
- `--num_finetune_tasks`: number of fixed fine-tuning tasks generated at the start of training
- `--fewshot_mix_probability`: probability that an individual sequence in an adaptation batch comes from the fine-tuning set

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
          --steps_per_epoch 1000 \
          --num_finetune_tasks 50 \
          --fewshot_lr 5e-4 \
          --fewshot_batch_size 128
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
- The few-shot metric is a `loss_trace` matrix that records the exact MSE for each fine-tuning task whenever it is included in an adaptation batch.
