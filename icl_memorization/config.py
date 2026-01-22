"""
Configuration settings for the Zipfian memorization analysis pipeline.

This file contains all configuration parameters used in the analysis script.
"""

import os
from pathlib import Path

# Directory paths
BASE_DIR = Path("./cache")
FIGURE_OUTPUT_DIR = BASE_DIR / "zipf_figs"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# Ensure directories exist
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Experiment paths
EXPERIMENT_PATHS = {
    # Small model
    "small": [
        './cache/memo_apr6_zipf_num_heads_8_num_layers_12/memo_apr6_zipf_num_heads_8_num_layers_12_transformer_K_100000_L_100_hidden_8_nheads_8_nlayers_12_1744050810.647857.pkl'
    ],
    # Medium model
    "medium": [
        './cache/memo_apr6_zipf_num_heads_16_num_layers_24/memo_apr6_zipf_num_heads_16_num_layers_24_transformer_K_100000_L_100_hidden_8_nheads_16_nlayers_24_1744134825.3819082.pkl'
    ],
    # Large model with RoPE embedding
    # "large_rope": [
    #     './cache/memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding/memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1740632491.7617574.pkl'
    # ],
    # Large model with different learning rate
    "large_lr": [
        './cache/memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4/memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1744050810.8180943.pkl'
    ],
    # All models combined
    "all": [
        './cache/memo_apr6_zipf_num_heads_8_num_layers_12/memo_apr6_zipf_num_heads_8_num_layers_12_transformer_K_100000_L_100_hidden_8_nheads_8_nlayers_12_1744050810.647857.pkl',
        './cache/memo_apr6_zipf_num_heads_16_num_layers_24/memo_apr6_zipf_num_heads_16_num_layers_24_transformer_K_100000_L_100_hidden_8_nheads_16_nlayers_24_1744134825.3819082.pkl',
        './cache/memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4/memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1744050810.8180943.pkl'
    ]
}

# Plot configuration
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 300,
    "x_lim": (1e1, 1e5),
    "loss_y_lim": (.9e-2, 1),
    "accuracy_y_lim": (0.5, 1),
    "legend_loc": (1.3, 0.0),
    "legend_ncol": 3,
    "color_palette": "bright"  # Seaborn color palette name
}

# Rank selection for visualization (to avoid cluttering plots)
RANK_SELECTIONS = {
    "low": list(range(10)),               # First 10 ranks (most frequent)
    "mid": list(range(30, 40)),           # Ranks 30-40
    "high": list(range(60, 70)),          # Ranks 60-70
    "selected": list(range(10)) + list(range(30, 40)) + list(range(60, 70))  # Combined selection
}

# Analysis settings
ANALYSIS_CONFIG = {
    "force_reprocess": False,  # Whether to force reprocessing of data even if cache exists
    "show_plots": True,        # Whether to display plots during execution
    "save_plots": True,        # Whether to save plots to files
} 