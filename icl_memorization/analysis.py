#!/usr/bin/env python
"""
Analysis script for transformer memorization experiments with Zipfian distributions.

This script analyzes how transformer models memorize sequences with different 
frequencies in a Zipfian distribution, visualizing the relationship between
sequence frequency and model performance.
"""

import pickle
import traceback
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import hashlib
import time
from pathlib import Path
import argparse
from collections import defaultdict

# Import configuration
from config import (
    FIGURE_OUTPUT_DIR, 
    PROCESSED_DATA_DIR, 
    EXPERIMENT_PATHS, 
    PLOT_CONFIG, 
    RANK_SELECTIONS,
    ANALYSIS_CONFIG
)

wandb_group_name = "memo_feb17_zipf"
# Filter runs by group name
# Initialize lists to store data for the heatmap
n_heads_list = []
n_layers_list = []
accuracy_list = []
train_loss_list = [[], [], []]
test_loss_list = []
item=0
runs = glob(f"./cache/{wandb_group_name}/{wandb_group_name}_*.pkl")

import json
from collections import defaultdict
runs=['./cache/memo_feb26_zipf_num_heads_8_num_layers_12/memo_feb26_zipf_num_heads_8_num_layers_12_transformer_K_100000_L_100_hidden_8_nheads_8_nlayers_12_1741157493.041445.pkl']
# runs=['./cache/memo_feb26_zipf_num_heads_16_num_layers_24/memo_feb26_zipf_num_heads_16_num_layers_24_transformer_K_100000_L_100_hidden_8_nheads_16_nlayers_24_1740632142.5650826.pkl']
# runs=['./cache/memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding/memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1740632491.7617574.pkl']
# runs=['./cache/memo_feb26_zipf_num_heads_24_num_layers_36_lr_1e-4/memo_feb26_zipf_num_heads_24_num_layers_36_lr_1e-4_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1741161215.594898.pkl']

# runs = ['./cache/memo_feb26_zipf_num_heads_8_num_layers_12/memo_feb26_zipf_num_heads_8_num_layers_12_transformer_K_100000_L_100_hidden_8_nheads_8_nlayers_12_1741157493.041445.pkl',
#         './cache/memo_feb26_zipf_num_heads_16_num_layers_24/memo_feb26_zipf_num_heads_16_num_layers_24_transformer_K_100000_L_100_hidden_8_nheads_16_nlayers_24_1740632142.5650826.pkl',
#         './cache/memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding/memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1740632491.7617574.pkl',
#         './cache/memo_feb26_zipf_num_heads_24_num_layers_36_lr_1e-4/memo_feb26_zipf_num_heads_24_num_layers_36_lr_1e-4_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1741161215.594898.pkl']
        
        

# Configuration
FIGURE_OUTPUT_DIR = "./cache/zipf_figs"
PROCESSED_DATA_DIR = "./cache/processed_data"
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def get_cache_filename(run_paths: List[str]) -> str:
    """
    Generate a cache filename based on the input files.
    
    Args:
        run_paths: List of paths to experiment pickle files
        
    Returns:
        Cache filename for processed data
    """
    # Create a hash based on the file paths and their modification times
    hasher = hashlib.md5()
    for path in sorted(run_paths):
        path_obj = Path(path)
        if path_obj.exists():
            mtime = os.path.getmtime(path)
            hasher.update(f"{path}_{mtime}".encode())
    
    return os.path.join(PROCESSED_DATA_DIR, f"processed_{hasher.hexdigest()}.pkl")

def load_experiment_data(run_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load experiment data from pickle files with caching.
    
    Args:
        run_paths: List of paths to experiment pickle files
        
    Returns:
        List of loaded experiment records
    """
    records = []
    for run_path in run_paths:
        print(f"Loading: {run_path}")
        try:
            with open(run_path, "rb") as f:
                record = pickle.load(f)
            print(f"Loaded record with {len(record['logs'])} logs")
            records.append(record)
        except Exception as e:
            print(f"Error loading {run_path}: {e}")
    return records

def process_experiment_data(record: Dict[str, Any], experiment_name: str = "unknown") -> Dict[str, List]:
    """
    Process experiment data to extract performance metrics.
    
    Args:
        record: Dictionary containing experiment logs and configuration
        experiment_name: Name identifier for this experiment
        
    Returns:
        Dictionary with processed metrics for plotting
    """
    acc_vs_presentations = defaultdict(list)
    
    # Extract model configuration
    model_config = {
        "n_heads": record["args"].get("num_heads", "unknown"),
        "n_layers": record["args"].get("num_layers", "unknown"),
        "hidden_size": record["args"].get("num_hidden_features", "unknown"),
        "learning_rate": record["args"].get("lr", "unknown"),
        "embedding_type": "rope" if "rope" in record["args"].get("wandb_group_name", "") else "standard",
        "experiment_name": experiment_name
    }
    
    # Get the last log entry with test metrics (assuming final training state)
    test_df_list = []
    test_df_dict = defaultdict(list)
    rank_ranges = [0,5,10,100,200,500,1000]
    while rank_ranges[-1] < (record["args"]["K"]):
        rank_ranges.append(rank_ranges[-1] + 1000)
        
    for log in (record["logs"]):
        if log.get("test_metrics") and len(log.get("test_metrics", {})) > 0:
            final_log = log
        else:
            continue 
     

        test_df = pd.DataFrame(final_log["test_metrics"])
        
        expected_presentations = final_log.get("num_apppearances", [])
        if not isinstance(expected_presentations, np.ndarray):
            expected_presentations = np.array(expected_presentations)
        expected_presentations = expected_presentations + 1e-10  # Avoid division by zero
        test_df["num_apppearances"] = expected_presentations
        test_df_list.append(test_df)
        
        for i in range(len(rank_ranges) - 1):
            test_df_range = test_df[(test_df["sequence_rank"] >= rank_ranges[i]) & (test_df["sequence_rank"] < rank_ranges[i+1])]
            # take the mean over all rows for each column if column is numeric
            # else take the first non-null value
            # loop through columns 
            for col in ["sequence_rank", "num_apppearances", "logsoftmaxloss", "accuracy"]:
                test_df_dict[col].append(test_df_range[col].mean())
            test_df_dict["sequences_in_range"].append(f"{rank_ranges[i]}-{rank_ranges[i+1]}")
            test_df_dict["num_heads"].append(record["args"]["num_heads"])
            test_df_dict["num_layers"].append(record["args"]["num_layers"])
            test_df_dict["experiment_name"].append(record["args"].get("wandb_group_name", ""))
    test_df = pd.concat(test_df_list) 
    # Also store model configuration for reference
    for key, value in model_config.items():
        test_df[key] = ([value] * len(test_df))
        
    # break up the test_df into a list of dataframes, one for rank ranges 0-5,5-10,10-100,100-200,200-500,500-1000,
    # and then increments of 1000 afterwards
    
    
    
    test_df_dict = pd.DataFrame(test_df_dict)
    # display(test_df_dict)
    return test_df_dict

def process_and_cache_data(run_paths: List[str], 
                          experiment_name: str = "unknown",
                          force_reprocess: bool = False) -> pd.DataFrame:
    """
    Process experiment data and cache results for faster reuse.
    
    Args:
        run_paths: List of paths to experiment files
        experiment_name: Name identifier for this set of experiments
        force_reprocess: Whether to force reprocessing even if cache exists
        
    Returns:
        DataFrame with processed metrics
    """
    # Generate cache filename based on input files
    cache_file = get_cache_filename(run_paths)
    print(f"Cache file: {cache_file}")
    # Try to load from cache first
    if os.path.exists(cache_file) and not force_reprocess:
        print(f"Loading processed data from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache, will reprocess: {e}")
    
    print(f"Processing data from scratch for {experiment_name}...")
    start_time = time.time()
    
    # Load and process data
    records = load_experiment_data(run_paths)
    all_metrics = defaultdict(list)
    
    for i, record in enumerate(records):
        # Use a unique name for each record if multiple in the same experiment
        record_name = f"{experiment_name}_{i}" if len(records) > 1 else experiment_name
        experiment_metrics = process_experiment_data(record, record_name)
        for key, values in experiment_metrics.items():
            all_metrics[key].extend(values)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    
    # Save to cache
    print(f"Saving processed data to cache: {cache_file}", flush = True)
    with open(cache_file, 'wb') as f:
        pickle.dump(metrics_df, f)
    
    processing_time = time.time() - start_time
    print(f"Data processing completed in {processing_time:.2f} seconds")
    
    return metrics_df

def create_visualization(metrics_df: pd.DataFrame, 
                         ranks_to_show: Optional[List[int]] = None,
                         title_suffix: str = "",
                         output_prefix: str = "",
                         hue_column: str = "sequences_in_range",
                         show_plots: bool = True,
                         save_plots: bool = True) -> None:
    """
    Create and save visualizations of model performance vs. presentations.
    
    Args:
        metrics_df: DataFrame containing metrics to plot
        ranks_to_show: List of ranks to include in visualization (subset for clarity)
        title_suffix: Additional text to append to plot titles
        output_prefix: Prefix for output filenames
        hue_column: Column to use for the hue (color) in the plots
        show_plots: Whether to display plots during execution
        save_plots: Whether to save plots to files
    """
    # Create a copy to avoid modifying the original
    df = metrics_df.copy()
    
    # Filter ranks if specified and we're using rank as the hue
    if ranks_to_show is not None and hue_column == "rank":
        df = df[df["rank"].isin(ranks_to_show)]
        df[hue_column] = df[hue_column].astype("category")
    # map hue_column to a color 
    hue_to_color = {hue: color for hue, color in zip(df[hue_column].unique(), sns.color_palette(PLOT_CONFIG["color_palette"], len(df[hue_column].unique())))}
    # Number of unique categories for coloring
    n_categories = len(df[hue_column].unique())
    color_palette = sns.color_palette(PLOT_CONFIG["color_palette"], n_categories)
    display(df)
    # Plot 1: Loss vs Presentations (line)
    plt.figure(figsize=PLOT_CONFIG["figsize"])
    sns.lineplot(
        # df["num_apppearances"],
        # df["logsoftmaxloss"], 
        # color=df[hue_column].map(hue_to_color),
        x = "num_apppearances",
        y = "logsoftmaxloss", 
        hue=df[hue_column], 
        data=df,
        palette=color_palette,
        marker = "D",
        markersize = 1,
        linewidth = 2
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of Presentations")
    plt.ylabel("Log Loss")
    plt.xlim(*PLOT_CONFIG["x_lim"])
    plt.ylim(*PLOT_CONFIG["loss_y_lim"])
    plt.title(f"Log Loss vs. Number of Presentations{title_suffix}")
    # legend in many columns 
    plt.legend(loc=PLOT_CONFIG["legend_loc"], ncol=PLOT_CONFIG["legend_ncol"])
    # Handle legend placement
    # if n_categories <= 100:  # Only show legend if not too many categories
    #     plt.legend(loc=PLOT_CONFIG["legend_loc"], ncol=PLOT_CONFIG["legend_ncol"])
    # else:
    #     plt.legend([],[], frameon=False)  # Hide legend if too many categories
    
    if save_plots:
        plt.savefig(
            f"{FIGURE_OUTPUT_DIR}/{output_prefix}logsoftmaxloss_vs_presentations_line.png", 
            bbox_inches='tight', 
            dpi=PLOT_CONFIG["dpi"]
        )
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    
     
    
    # Plot 2: Accuracy vs Presentations
    plt.figure(figsize=PLOT_CONFIG["figsize"])
    sns.lineplot(
        x="num_apppearances", 
        y="accuracy", 
        hue=hue_column, 
        data=df,
        palette=color_palette,
        marker = "D",
        markersize = 3,
        linewidth = 2
    )
    plt.xscale("log")
    plt.xlabel("Number of Presentations")
    plt.ylabel("Accuracy")
    plt.ylim(*PLOT_CONFIG["accuracy_y_lim"])
    plt.xlim(*PLOT_CONFIG["x_lim"])
    plt.legend(loc=PLOT_CONFIG["legend_loc"], ncol=PLOT_CONFIG["legend_ncol"])
    plt.title(f"Accuracy vs. Number of Presentations{title_suffix}")
    
    # Handle legend placement
    # if n_categories <= 100:
    #     plt.legend(loc=PLOT_CONFIG["legend_loc"], ncol=PLOT_CONFIG["legend_ncol"])
    # else:
    #     plt.legend([],[], frameon=False)
    
    if save_plots:
        plt.savefig(
            f"{FIGURE_OUTPUT_DIR}/{output_prefix}accuracy_vs_presentations.png", 
            bbox_inches='tight', 
            dpi=PLOT_CONFIG["dpi"]
        )
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_comparative_visualizations(all_data: pd.DataFrame, 
                                     show_plots: bool = True,
                                     save_plots: bool = True) -> None:
    """
    Create visualizations comparing model architectures.
    
    Args:
        all_data: DataFrame containing data from multiple models
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
    """
    ranks_subset = [0,5,10,100,200,500,1000]
    while ranks_subset[-1] < (100000):
        ranks_subset.append(ranks_subset[-1] + 1000)
    wandb_group_names_to_legend_names = {
        "memo_feb26_zipf_num_heads_8_num_layers_12": "8 heads, 12 layers",
        "memo_feb26_zipf_num_heads_16_num_layers_24": "16 heads, 24 layers",
        "memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding": "24 heads, 36 layers",
        "memo_feb26_zipf_num_heads_24_num_layers_36_lr_1e-4": "24 heads, 36 layers (lr=1e-4)"
    }
    all_data["experiment_name"] = all_data["experiment_name"].map(wandb_group_names_to_legend_names)
    for i_rank in range(len(ranks_subset) - 1):
        # Filter to only include specified ranks
        lower_rank = ranks_subset[i_rank]
        upper_rank = ranks_subset[i_rank + 1] 
        filtered_data = all_data[(all_data["sequence_rank"] >= lower_rank) & (all_data["sequence_rank"] < upper_rank)]
        # filtered_data["experiment_name"] = filtered_data["num_heads"].astype(str) + " heads, " + filtered_data["num_layers"].astype(str) + " layers"
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot log loss
        sns.lineplot(
            x="num_apppearances",
            y="logsoftmaxloss", 
            hue="experiment_name",
            data=filtered_data,
            ax=axs[0]
        )
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlim(*PLOT_CONFIG["x_lim"])
        axs[0].set_ylim(*PLOT_CONFIG["loss_y_lim"])
        axs[0].set_xlabel("Number of Presentations")
        axs[0].set_ylabel("Log Loss")
        axs[0].set_title(f"Log Loss vs. Number of Presentations\nfor {lower_rank}-{upper_rank} Ranks")
        axs[0].legend([],[], frameon=False)
        # Plot accuracy
        sns.lineplot(
            x="num_apppearances",
            y="accuracy",
            hue="experiment_name", 
            data=filtered_data,
            ax=axs[1]
        )
        axs[1].set_xscale("log")
        axs[1].set_xlim(*PLOT_CONFIG["x_lim"])
        axs[1].set_ylim(0.5, 1)
        axs[1].set_xlabel("Number of Presentations")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_title(f"Accuracy vs. Number of Presentations\nfor {lower_rank}-{upper_rank} Ranks")
        axs[1].legend(loc=PLOT_CONFIG["legend_loc"])
        if save_plots:
            plt.savefig(
                f"{FIGURE_OUTPUT_DIR}/comparative_visualizations/logsoftmaxloss_vs_presentations_line_{lower_rank}-{upper_rank}.png", 
                bbox_inches='tight', 
                dpi=PLOT_CONFIG["dpi"]
            )
        if show_plots:
            plt.show()
        else:
            plt.close()
     

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze transformer memorization with Zipfian distributions."
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="small",
        choices=list(EXPERIMENT_PATHS.keys()),
        help="Which experiment configuration to analyze"
    )
    parser.add_argument(
        "--rank-selection", 
        type=str, 
        default="selected",
        choices=list(RANK_SELECTIONS.keys()),
        help="Which rank selection to use for visualizations"
    )
    parser.add_argument(
        "--force-reprocess", 
        action="store_true",
        help="Force reprocessing of data even if cache exists"
    )
    parser.add_argument(
        "--no-show-plots", 
        action="store_true",
        help="Don't display plots during execution"
    )
    parser.add_argument(
        "--no-save-plots", 
        action="store_true",
        help="Don't save plots to files"
    )
    parser.add_argument(
        "--comparative", 
        action="store_true",
        help="Generate comparative visualizations across model architectures"
    )
    
    return parser.parse_args()

def main(args: argparse.Namespace = None) -> None:
    # Example args
    # args = argparse.Namespace(experiment="small", rank_selection="selected", force_reprocess=False, no_show_plots=False, no_save_plots=False, comparative=True)   
    """Main execution function"""
    if args is None:
        args = parse_arguments()
    
    # Get experiment paths from config
    experiment_key = args.experiment
    run_paths = EXPERIMENT_PATHS[experiment_key]
    print(f"Running experiment: {experiment_key}")
    # Get rank selection from config
    ranks_to_show = RANK_SELECTIONS[args.rank_selection]
    
    # Set processing and visualization options
    force_reprocess = args.force_reprocess or ANALYSIS_CONFIG["force_reprocess"]
    show_plots = not args.no_show_plots and ANALYSIS_CONFIG["show_plots"]
    save_plots = not args.no_save_plots and ANALYSIS_CONFIG["save_plots"]
    
    # Process data with caching
    metrics_df = process_and_cache_data(
        run_paths, 
        experiment_name=experiment_key,
        force_reprocess=force_reprocess
    )
    display(metrics_df)
    
    
    
    # Generate comparative visualizations if requested
    if args.comparative:
        create_comparative_visualizations(
            metrics_df,
            show_plots=show_plots,
            save_plots=save_plots
        )
    else:
        # Create basic visualizations
        create_visualization(
            metrics_df, 
            ranks_to_show=ranks_to_show,
            title_suffix=f", {args.rank_selection.title()} Ranks",
            output_prefix=f"{experiment_key}_",
            show_plots=show_plots,
            save_plots=save_plots
        )
if __name__ == "__main__":
    main()


 