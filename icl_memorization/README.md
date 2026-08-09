# Hierarchical Memories
Consider the following setup for Hierarchical Memories.
I'd like memories to be organized into a cluster of memories, where memories within a cluster are similar to each other.
e.g. If cluster size = 2, then we have multiple pairs of memories, where one sequence is similar to the other one.
We still keep the variable $K$ to mean the total number of memories.
If cluster size $c = 2$, then we have K/2 clusters 
Have a parameter in args. that controls the similarity of the two memories in a cluster 
Add a data class that implements this hierarchical memory setup.
Add a wandb_group_name experiment to implement this experiment

### Implementation Details:
The Hierarchical Memories feature has been implemented as follows:
- **Command Line Arguments**: Added `--cluster_size` and `--cluster_similarity` to `main.py`.
  - `cluster_size`: Defines how many sequences belong to a single cluster. Total clusters = $K / c$.
  - `cluster_similarity`: Represents the probability that any bit in a sequence matches the corresponding bit in its cluster's prototype sequence (1.0 = identical, 0.5 = random noise).
- **Data Class**: A new `HierarchicalSequence` dataset class replaces the standard `Sequence` when `--cluster_size > 1`. It generates `K/c` prototype sequences, and assigns members to each cluster by flipping bits according to the `cluster_similarity`.
- **Experiments**: A new wandb configuration block under `--wandb_group_name "memo_hierarchical_memories"` demonstrates varying `cluster_similarity` uniformly across `SLURM_ARRAY_TASK_ID` for $c=2$.
# Main_freeze experiments 
Main_freeze.ipynb
Validation: 
We save the state_dict of the model before switching the sequences.
```
c_attn_weight = model.c_attn.weight.detach().clone()
c_attn_bias = model.c_attn.bias.detach().clone() 
```
We generate new sequences with: `train_dataset.generate_sequences()`

We test the model on all sequences in the validation set, both the pre-switch and post-switch sequences
We only test the model on every 10th sequence in the train set

# Analysis of the experiments

A directory e.g. ./cache/memo_aug2_zipf_onelayerattention_lr_1e-3_swapmlp_eval can contain subdirectories storing the results of the experiments.
The results of experiments are stored in a pickle file with the same name as the directory but ending with ".pkl".
There is another pickle file in the same directory with the same name but ending with "_sequences.pkl" that contains the sequences used in the experiments.
You can load the results of the experiments with:
```
with open(f"{dir_name}/{dir_name}.pkl", "rb") as f:
    data = utils.CPU_Unpickler(f).load()
```

You can load the sequences with:
```
with open(f"{dir_name}/{dir_name}_sequences.pkl", "rb") as f:
    sequences = utils.CPU_Unpickler(f).load()
```
Now `sequences` is a dictionary with the following keys: 
* `sequences`: a list of sequence arrays. The length of the list is the number of switches. Each sequence array is of shape (num_sequences, length_of_sequence).
* `switch_start_iter`: the iteration at which the switch happened 


# Phenomenological Model of Memorization 
 

## Implementation notes
- Initialization of DAM: $\mathcal{M}_0 = \vec 0$
- If no memory has been added, then we return 0.5 for both classes.
- If no bit has been revealed, i.e. $n=0$:
    - If memory has been added, then we return the probability of the first bit being +1.
-  The $M$ sequences $\chi_1, \chi_2, \ldots, \chi_M$ are stored in the DAM as $H$ dimensional vectors $\varphi_1, \varphi_2, \ldots, \varphi_M$, where $\varphi_\mu=B \chi_\mu$ for $\mu=1,2, \ldots, M$.
    - I assumed $\xi$ is a binary sequence, so $B$ is a $H \times N$ matrix and $\chi_\mu$ is a $N \times 1$-dimensional binary vector.
    - So $\varphi_\mu$ is a $H \times 1$-dimensional real vector.
=======
# Transformer Memorization Analysis

This project analyzes how transformer models memorize sequences with different frequencies in a Zipfian distribution. It provides tools to visualize the relationship between sequence frequency (number of presentations) and model performance (loss and accuracy).

## Overview

When training language models, some tokens or sequences appear much more frequently than others, following a Zipfian distribution. This analysis explores how a model's ability to memorize sequences correlates with their frequency in the training data.

This analysis pipeline:

1. Processes experiment data from pickled training logs
2. Extracts performance metrics for each sequence based on its frequency
3. Generates visualizations showing the relationship between:
   - Number of presentations vs. Log Loss
   - Number of presentations vs. Accuracy

## Features

- **Data Caching**: Processed data is cached to avoid redundant computations
- **Configurable Analysis**: Easily adjust visualization parameters
- **Comparative Analysis**: Compare performance across different model architectures
- **Command Line Interface**: Run analyses with different configurations via CLI arguments

## Usage

### Basic Usage

To run the basic analysis on the small model with default settings:

```bash
python analysis.py
```

### Command Line Options

```bash
python analysis.py --experiment all --rank-selection low --comparative
```

Available options:

- `--experiment`: Which experiment configuration to analyze (choices: small, medium, large_rope, large_lr, all)
- `--rank-selection`: Which rank selection to use for visualizations (choices: low, mid, high, selected)
- `--force-reprocess`: Force reprocessing of data even if cache exists
- `--no-show-plots`: Don't display plots during execution
- `--no-save-plots`: Don't save plots to files
- `--comparative`: Generate comparative visualizations across model architectures

### Configuration

Edit `config.py` to modify:

- Experiment paths
- Plot configuration (figure size, limits, etc.)
- Rank selections for visualization
- Default analysis settings

## Project Structure

- `analysis.py`: Main analysis script
- `config.py`: Configuration settings for the analysis
- `cache/`: Directory for experiment data and processed results
  - `zipf_figs/`: Output directory for generated figures
  - `processed_data/`: Cache directory for processed data

## Visualizations

The pipeline generates several types of visualizations:

1. **Loss vs. Presentations (line and scatter plots)**: Shows how log loss decreases as the model sees sequences more frequently
2. **Accuracy vs. Presentations**: Shows how accuracy increases with more presentations
3. **Comparative Visualizations** (when enabled):
   - Model comparisons
   - Architecture comparisons (number of heads, layers, embedding types)

## Performance Tips

- Use the caching system to avoid reprocessing data
- For large experiments, use the `--no-show-plots` option and view the saved figures later
- Select appropriate rank subsets to reduce visual complexity in plots

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn

## References

For more information on the Zipfian distribution and how it relates to language model memorization, see:
- [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law)
- Research papers on transformer memorization capabilities 

# Phenomenological Model of Memorization 
Suppose that we have $K$ sequences where each sequence $\boldsymbol{\xi} = \xi_1, \xi_2, \ldots, \xi_{N}$ has $N_i$ bits. Here we note that $\xi_i$ is the content embedding of the $i$th bit in the sequence. We want to model the probability that the transformer correctly predicts the $j+1$th bit of sequence $\boldsymbol{\xi}$, given the prefix $\boldsymbol{\xi}_{0..j}$. 

We suppose that the model generates a representation $\boldsymbol{v}$ of the sequence $\boldsymbol{\xi}$. This representation is equivalent to a vector $v_1, v_2, \ldots, v_N$ where each vector $v_i\in \mathbb{R}^d$ is a vector of $d$ features. 

Since the model is autoregressive, this representation $v_i$ for the $n$th bit is only a function of the prefix $\boldsymbol{\xi}_{0..i-1}$. Therefore, we can write $$v_i = \sum_{i=1}^{n-1} A^{(n)}_i \xi_i^{(i)}$$ where $v_i^{(i)}$ is the representation of the $i$th bit of the $i$th sequence.

The probability that the model assigns to the $n+1$-th bit being positive is given by $$P(\boldsymbol{\xi}_{n+1}=+1 | \boldsymbol{\xi}_{0..j}) = \sum_{m,n}p_m\sigma(v_j \cdot B)$$ where $\sigma$ is the sigmoid function.
The loss function is then given by $$L = -\frac{1}{MN}\sum_{m=1}^{M}\sum_{n=0}^{N-1} \mathcal {L}_{BCE}( {m,n} ; A^{(n)}, B)$$ where $\mathcal {L}_{BCE}$ is the binary cross-entropy loss.
Here $A^{(n)}$ is the matrix of the attention weights for the $n$th bit and $B$ is the matrix corresponding to the query vector.