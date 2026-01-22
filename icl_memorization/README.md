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