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

## Main Grow experiments

### 1. Research Motivation & Hypothesis

The **Model Growth and Weight Transfer Experiment** investigates whether pretraining a smaller transformer model on an in-context sequence memorization task can provide an effective inductive bias and initialization when scaled into a larger architecture.

Key scientific questions addressed by this experiment:
1. **Model Expansion vs. Training from Scratch**: Does warm-starting a larger model with submatrices from a trained smaller model accelerate convergence and improve final memorization capacity compared to random initialization?
2. **Growth Geometry (Width vs. Heads)**: Is performance more sensitive to expanding the head dimension $d$ (width per head), increasing the number of attention heads $H$, or scaling both simultaneously?
3. **Component Ablation (Attention vs. MLP Transfer)**: Does the learned representation reside predominantly in the **attention routing/query-key-value geometry** or in the **MLP associative key-value memory**? By selectively transferring either attention weights, MLP weights, or both, we isolate the functional contributions of each transformer sub-network to meta-memorization.

---

### 2. Two-Phase Training Protocol

The experiment follows a strict two-phase protocol parameterized by iterations ($1{,}000$ iterations per epoch, $100{,}000$ total iterations):

```
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Base Model Training (Epochs 0 to 49 / Iterations 0 to 50,000)                             |
|  - Architecture: OneLayerAttention (H_base = 8, d_base = 32, C_base = 256, mlp_dim = 256)           |
|  - Dataset: Zipfian sequence library (K = 500,000, context length N = 100, vocab size V = 2)       |
|  - Optimizer: Adam (lr = 1e-3, weight_decay = 0.0)                                                 |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                      [ Epoch 50 Transition ]
                                                  |
  1. Base Checkpoint: Save model state dict to {dirprefix}/{fileprefix}_base_model_epoch_50.pkl
  2. Sequence Resampling: Resample all K = 500,000 sequences via train_dataset.generate_sequences()
     - Appearance counters reset (num_appearances = 0)
     - Test evaluation set synchronized
     - Sequence switch record saved to {dirprefix}/{fileprefix}_all_sequences.pkl
  3. Model Growth & Weight Transfer: Instantiate target architecture (H_target, d_target)
     - Transfer submatrices according to transfer_mode ('both', 'attn_only', 'mlp_only')
  4. Optimizer Reset: Reinitialize Adam optimizer moments for new parameter shapes
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 2: Retraining on Resampled Sequences (Epochs 50 to 99 / Iterations 50,000 to 100,000)       |
|  - Evaluates how well the transferred subnetwork adapts to / memorizes fresh Zipfian sequences      |
|  - Tracks per-sequence presentation loss, accuracy, and forgetting dynamics                         |
+----------------------------------------------------------------------------------------------------+
```

---

### 3. Submatrix Weight Transfer Mechanics

Let the base model have $H_s$ heads and $d_s$ hidden features per head ($C_s = H_s d_s$), and the target model have $H_t \ge H_s$ heads and $d_t \ge d_s$ hidden features per head ($C_t = H_t d_t$). The MLP has $M_s = C_s$ and $M_t = C_t$ hidden dimensions.

The utility `grow_onelayer_attention_weights` in `grow_utils.py` embeds the base parameters into the corresponding subspaces of the target model:

1. **Token and Positional Embeddings** ($W_{te} \in \mathbb{R}^{V 	imes C_t}$, $W_{pe} \in \mathbb{R}^{N 	imes C_t}$):
   For each head $h \in \{0, \dots, H_s - 1\}$:
   $$W_{target}[:, h \cdot d_t : h \cdot d_t + d_s] \leftarrow W_{base}[:, h \cdot d_s : (h+1) d_s]$$

2. **Layer Normalizations** ($\gamma, eta \in \mathbb{R}^{C_t}$ for $LN_f$ and $LN_{	ext{pre}}$):
   $$\gamma_{target}[h \cdot d_t : h \cdot d_t + d_s] \leftarrow \gamma_{base}[h \cdot d_s : (h+1) d_s]$$

3. **Multi-Head Attention Projection** ($W_{qkv} \in \mathbb{R}^{3 C_t 	imes C_t}$):
   For each projection block $qkv \in \{Q, K, V\}$ and head pair $(h, h') \in \{0, \dots, H_s - 1\}^2$:
   $$W_{target}[qkv \cdot C_t + h \cdot d_t : qkv \cdot C_t + h \cdot d_t + d_s,\; h' \cdot d_t : h' \cdot d_t + d_s] \leftarrow W_{base}[qkv \cdot C_s + h \cdot d_s : qkv \cdot C_s + (h+1) d_s,\; h' \cdot d_s : (h'+1) d_s]$$

4. **First MLP Layer** ($W_{mlp, 0} \in \mathbb{R}^{M_t 	imes C_t}$, $b_{mlp, 0} \in \mathbb{R}^{M_t}$):
   $$W_{target}[:M_s,\; h' \cdot d_t : h' \cdot d_t + d_s] \leftarrow W_{base}[:M_s,\; h' \cdot d_s : (h'+1) d_s]$$
   $$b_{target}[:M_s] \leftarrow b_{base}[:M_s]$$

5. **Intermediate MLP Layers** ($W_{mlp, i} \in \mathbb{R}^{M_t 	imes M_t}$, $b_{mlp, i} \in \mathbb{R}^{M_t}$ for $i \ge 1$):
   $$W_{target}[:M_s, :M_s] \leftarrow W_{base}[:M_s, :M_s], \quad b_{target}[:M_s] \leftarrow b_{base}[:M_s]$$

6. **LM Prediction Head** ($W_{head} \in \mathbb{R}^{V 	imes M_t}$, $b_{head} \in \mathbb{R}^{V}$):
   $$W_{target}[:, :M_s] \leftarrow W_{base}[:, :M_s], \quad b_{target}[:] \leftarrow b_{base}[:]$$

Any target parameters outside these sliced ranges retain their fresh random initialization per GPT-2 initialization standards.

---

### 4. Component Ablation Modes (`--transfer_mode`)

To isolate where transferable knowledge is stored, each architecture is tested under three weight transfer modes:

- **`both`**: Transfers both attention projections/embeddings and MLP layers into the target model.
- **`attn_only`**: Transfers embeddings, layer norms, and attention projections ($W_{qkv}$); all MLP layers and the LM head remain freshly initialized with random weights.
- **`mlp_only`**: Transfers all MLP layers and the LM head; embeddings, layer norms, and attention projections remain freshly initialized with random weights.

---

### 5. Experimental Design Matrix ($4 	imes 3 = 12$ Conditions)

The experiment crosses 4 target architectures with 3 transfer modes to produce 12 conditions:

| Setting ID | Architecture Condition | Target Heads ($H$) | Target Width ($d$) | Total Hidden ($C$) | MLP Dim ($M$) | Transfer Mode | Description |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `base` | 8 | 32 | 256 | 256 | `attn_only` | Keep attention; reset MLP |
| **1** | `base` | 8 | 32 | 256 | 256 | `mlp_only` | Keep MLP; reset attention |
| **2** | `base` | 8 | 32 | 256 | 256 | `both` | Full continuation baseline |
| **3** | `width_growth` | 8 | 64 | 512 | 512 | `attn_only` | Width growth; transfer attention only |
| **4** | `width_growth` | 8 | 64 | 512 | 512 | `mlp_only` | Width growth; transfer MLP only |
| **5** | `width_growth` | 8 | 64 | 512 | 512 | `both` | Width growth; transfer both |
| **6** | `head_growth` | 16 | 32 | 512 | 512 | `attn_only` | Head growth; transfer attention only |
| **7** | `head_growth` | 16 | 32 | 512 | 512 | `mlp_only` | Head growth; transfer MLP only |
| **8** | `head_growth` | 16 | 32 | 512 | 512 | `both` | Head growth; transfer both |
| **9** | `both_growth` | 16 | 64 | 1024 | 1024 | `attn_only` | Width & head growth; transfer attention only |
| **10** | `both_growth` | 16 | 64 | 1024 | 1024 | `mlp_only` | Width & head growth; transfer MLP only |
| **11** | `both_growth` | 16 | 64 | 1024 | 1024 | `both` | Width & head growth; transfer both |

---

### 6. SLURM Array Allocation & Paired Random Seeds

The experiment contains two complementary job array blocks:

#### Block 1: Model Growth & Weight Transfer (`#SBATCH --array=0-35`)
36 tasks evaluating progressive model growth from the small base model (H=8, d=32):
- **Setting Index**: `setting_idx = SLURM_ARRAY_TASK_ID % 12`
- **Seed Offset**: `seed_offset = SLURM_ARRAY_TASK_ID // 12`
  - **Tasks 0–11**: Seed `42` (`seed_offset = 0`)
  - **Tasks 12–23**: Seed `1042` (`seed_offset = 1`)
  - **Tasks 24–35**: Seed `2042` (`seed_offset = 2`)

#### Block 2: Base Initialization Controls at Larger Sizes (`#SBATCH --array=36-62`)
27 tasks evaluating large models trained directly from scratch with standard base initialization from Epoch 0 (resampling data at Epoch 50, controlling for model size without progressive growth):
- **Arch Conditions**:
  - `base_width_growth`: H=8, d=64 (1.63M params, base initialization from Epoch 0)
  - `base_head_growth`: H=16, d=32 (1.63M params, base initialization from Epoch 0)
  - `base_both_growth`: H=16, d=64 (6.41M params, base initialization from Epoch 0)
- **Transfer Modes at Epoch 50 Data Resampling**:
  - `both`: All trained weights retained across the data switch
  - `attn_only`: Attention weights retained, MLP reinitialized
  - `mlp_only`: MLP weights retained, attention reinitialized
- **Task Mapping**:
  - **Tasks 36–44**: Seed `42` (`seed_offset = 0`)
  - **Tasks 45–53**: Seed `1042` (`seed_offset = 1`)
  - **Tasks 54–62**: Seed `2042` (`seed_offset = 2`)

> [!NOTE]
> Because Tasks 0–11 share Seed `42`, all 12 settings begin from **identical initial weights and identical Step 1 sequence samples**, providing a rigorous paired comparison when models branch off at Epoch 50.

---

### 7. Code Organization & Execution

- **Training Script**: [`main_grow.py`](main_grow.py)
- **Batch Submission**: [`main_grow.sh`](main_grow.sh)
- **Growing Utility**: [`grow_utils.py`](grow_utils.py)
- **Unit Test Suite**: [`tests/test_grow_small.py`](tests/test_grow_small.py)

#### Running the Experiment
Submit the array job to SLURM:
```bash
sbatch main_grow.sh
```

#### Running Unit Tests
Execute the unit test suite on small networks:
```bash
python tests/test_grow_small.py
```

#### Output Artifacts
Each run saves artifacts to `/scratch/qanguyen/gautam/cache/memo_sep27_zipf_onelayerattention_grow/<fileprefix>/`:
- `<fileprefix>_base_model_epoch_50.pkl`: Base model weights checkpointed immediately prior to growth at Epoch 50.
- `<fileprefix>_all_sequences.pkl`: Pickled dictionary containing the full sequence lists before and after the Epoch 50 resampling.
- `<fileprefix>_model_final.pkl`: Final retrained model state dictionary at Iteration 100,000.
- `<fileprefix>.pkl`: Test evaluation metrics, loss per appearance, and training logs across iterations.
