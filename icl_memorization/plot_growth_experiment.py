"""
Plotting and Analysis Script for Model Growth & Weight Transfer Experiment
==========================================================================
Analyzes the OneLayerAttention network growth and base control experiments:
- Pre-growth (Phase 1): num_hidden_features=32, num_heads=8 (Epochs 0-49 / Iterations 0-50,000)
- Growth & Resample (Phase 2): Epochs 50-100 / Iterations 50,000-100,000
  * Growth conditions (transferred init from 8h/32d):
    - width_growth (64d, 8h)
    - head_growth (32d, 16h)
    - both_growth (64d, 16h)
  * Base control conditions (trained from scratch with base init):
    - base (32d, 8h)
    - base_width_growth (64d, 8h)
    - base_head_growth (32d, 16h)
    - base_both_growth (64d, 16h)
  * Transfer modes: 'both', 'attn_only', 'mlp_only'

NOTE: No conditions or transfer modes are averaged together!
Replication variance is computed strictly across random seeds of the same condition.

Usage CLI:
  python plot_growth_experiment.py --fig 1       # Learning Trajectories (by arch, by mode, all conditions)
  python plot_growth_experiment.py --fig 2       # Capacity Scaling
  python plot_growth_experiment.py --fig 3       # Transfer Ablation (All conditions bar + strip plot)
  python plot_growth_experiment.py --fig 4       # Rank-Accuracy Profiles
  python plot_growth_experiment.py --fig all     # All Figures & Summary Table
"""

import os
import glob
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import utils

# Apply style
utils.set_plot_style()
plt.rcParams['text.usetex'] = False  # Headless cluster compatibility

OUTPUT_DIR = "./figs/Figure_growth"
utils.ensure_dir(OUTPUT_DIR)

CACHE_DIR = "/scratch/qanguyen/gautam/cache/memo_sep27_zipf_onelayerattention_grow"
DF_CACHE_FILE = os.path.join(OUTPUT_DIR, "growth_summary_df.pkl")

PARAM_MAP = {
    'base': 422402,
    'width_growth': 1633794,
    'head_growth': 1633794,
    'both_growth': 6413314,
    'base_width_growth': 1633794,
    'base_head_growth': 1633794,
    'base_both_growth': 6413314,
}

CONDITION_LABELS = {
    'base': 'Base (32d, 8h)',
    'width_growth': 'Width Growth [Transferred] (64d, 8h)',
    'head_growth': 'Head Growth [Transferred] (32d, 16h)',
    'both_growth': 'Both Growth [Transferred] (64d, 16h)',
    'base_width_growth': 'Width [Base Init] (64d, 8h)',
    'base_head_growth': 'Head [Base Init] (32d, 16h)',
    'base_both_growth': 'Both [Base Init] (64d, 16h)',
}

TRANSFER_LABELS = {
    'both': 'Both (Attn + MLP)',
    'attn_only': 'Attention Only',
    'mlp_only': 'MLP Only'
}

MODE_LINESTYLES = {
    'both': '-',
    'attn_only': '--',
    'mlp_only': ':'
}

ARCH_COLORS = {
    'base': '#1f77b4',
    'width_growth': '#d62728',
    'head_growth': '#2ca02c',
    'both_growth': '#ff7f0e',
    'base_width_growth': '#8c564b',
    'base_head_growth': '#17becf',
    'base_both_growth': '#9467bd'
}


def load_experiment_data(force_reload=False):
    """Load and process run logs into a tidy DataFrame without collapsing conditions."""
    if not force_reload and os.path.exists(DF_CACHE_FILE):
        with open(DF_CACHE_FILE, "rb") as f:
            return pickle.load(f)

    folders = sorted(glob.glob(f"{CACHE_DIR}/*"))
    print(f"Loading {len(folders)} run folders from {CACHE_DIR}...")

    records = []
    rank_profiles = []
    t0 = time.time()

    for idx, folder in enumerate(folders):
        run_name = os.path.basename(folder)
        fname = f"{folder}/{run_name}.pkl"
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "rb") as f:
                data = utils.CPU_Unpickler(f).load()
            if "logs" not in data or len(data["logs"]) == 0:
                continue

            args = data.get("args", {})
            cond = args.get("grow_condition", "base")
            mode = args.get("transfer_mode", "both")
            seed = args.get("seed", 42)
            task_id = args.get("SLURM_ARRAY_TASK_ID", idx)
            n_params = PARAM_MAP.get(cond, 422402)

            logs = data["logs"]
            max_epoch_run = max(l["epoch"] for l in logs)

            for l in logs:
                epoch = l["epoch"]
                iteration = epoch * 1000
                train_loss = l.get("train_loss", np.nan)
                phase = 1 if epoch < 50 else 2

                tm = l.get("test_metrics", {})
                top_3k_acc = np.nan
                top_1k_acc = np.nan
                max_rank_90 = 0
                max_rank_80 = 0

                if "accuracy" in tm and "sequence_rank" in tm:
                    ranks = tm["sequence_rank"]
                    acc = tm["accuracy"]

                    # Top-K accuracy
                    m3k = (ranks <= 3000)
                    if np.any(m3k):
                        top_3k_acc = float(np.mean(acc[m3k]))

                    m1k = (ranks <= 1000)
                    if np.any(m1k):
                        top_1k_acc = float(np.mean(acc[m1k]))

                    # Capacity (max rank with accuracy >= threshold)
                    met_90 = ranks[acc >= 0.90]
                    if len(met_90) > 0:
                        max_rank_90 = int(np.max(met_90))

                    met_80 = ranks[acc >= 0.80]
                    if len(met_80) > 0:
                        max_rank_80 = int(np.max(met_80))

                    # Save rank profiles for the latest epoch in Phase 2
                    if epoch == max_epoch_run and phase == 2:
                        sample_indices = np.unique(np.geomspace(1, min(len(ranks), 30000), num=120).astype(int)) - 1
                        for s_idx in sample_indices:
                            if s_idx < len(ranks):
                                rank_profiles.append({
                                    "condition": cond,
                                    "cond_label": CONDITION_LABELS.get(cond, cond),
                                    "transfer_mode": mode,
                                    "mode_label": TRANSFER_LABELS.get(mode, mode),
                                    "condition_setting": f"{CONDITION_LABELS.get(cond, cond)} [{TRANSFER_LABELS.get(mode, mode)}]",
                                    "seed": seed,
                                    "sequence_rank": int(ranks[s_idx]),
                                    "accuracy": float(acc[s_idx]),
                                    "epoch": epoch
                                })

                records.append({
                    "run_name": run_name,
                    "task_id": task_id,
                    "condition": cond,
                    "cond_label": CONDITION_LABELS.get(cond, cond),
                    "transfer_mode": mode,
                    "mode_label": TRANSFER_LABELS.get(mode, mode),
                    "condition_setting": f"{CONDITION_LABELS.get(cond, cond)} [{TRANSFER_LABELS.get(mode, mode)}]",
                    "seed": seed,
                    "epoch": epoch,
                    "iteration": iteration,
                    "phase": phase,
                    "train_loss": train_loss,
                    "top_1k_acc": top_1k_acc,
                    "top_3k_acc": top_3k_acc,
                    "max_rank_90": max_rank_90,
                    "max_rank_80": max_rank_80,
                    "num_parameters": n_params
                })

        except Exception as e:
            print(f"Error loading {run_name}: {e}")

    df = pd.DataFrame(records)
    df_profile = pd.DataFrame(rank_profiles)
    print(f"Loaded {len(df)} records across {df['condition_setting'].nunique()} distinct conditions in {time.time() - t0:.2f}s.")

    data_bundle = {"trajectory": df, "rank_profiles": df_profile}
    with open(DF_CACHE_FILE, "wb") as f:
        pickle.dump(data_bundle, f)
    return data_bundle


# =====================================================================
# FIGURE 1A: Learning Trajectories Faceted by Architecture
# =====================================================================
def plot_trajectories_by_architecture(df, savefigdir=OUTPUT_DIR):
    """
    Multi-panel grid: Each architecture has its own panel showing:
    - Both Transfer (Full)
    - Attention Only
    - MLP Only
    (and Base Init if present).
    Averaging is strictly across random seeds of the same condition!
    """
    # Group available architectures
    arch_order = [c for c in ['both_growth', 'head_growth', 'width_growth', 'base', 
                              'base_both_growth', 'base_head_growth', 'base_width_growth'] 
                  if c in df['condition'].unique()]
    
    n_arch = len(arch_order)
    n_cols = min(n_arch, 4)
    n_rows = 2  # Top: Accuracy, Bottom: Loss
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 8.5), sharex=True, sharey='row')
    if n_cols == 1:
        axs = axs.reshape(2, 1)

    mode_palette = {"Both (Attn + MLP)": "#2b5c8f", "Attention Only": "#e05a47", "MLP Only": "#4daf4a"}

    for col_idx, cond in enumerate(arch_order):
        sub_arch = df[df["condition"] == cond]
        arch_title = CONDITION_LABELS.get(cond, cond)

        # Top row: Top-3k Accuracy
        ax_acc = axs[0, col_idx]
        sns.lineplot(
            data=sub_arch,
            x="iteration",
            y="top_3k_acc",
            hue="mode_label",
            style="mode_label",
            dashes=False,
            palette=mode_palette,
            linewidth=2.2,
            ax=ax_acc,
            legend=(col_idx == 0)
        )
        ax_acc.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_acc.set_title(arch_title, fontsize=11, fontweight="bold")
        ax_acc.set_ylabel("Avg Accuracy (Ranks 1-3000)" if col_idx == 0 else "")
        ax_acc.set_ylim(0.48, 1.0)
        ax_acc.grid(True, linestyle="--", alpha=0.4)

        # Bottom row: Training Loss
        ax_loss = axs[1, col_idx]
        sns.lineplot(
            data=sub_arch,
            x="iteration",
            y="train_loss",
            hue="mode_label",
            style="mode_label",
            dashes=False,
            palette=mode_palette,
            linewidth=2.2,
            ax=ax_loss,
            legend=False
        )
        ax_loss.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_loss.set_xlabel("Iteration", fontsize=11)
        ax_loss.set_ylabel("Cross-Entropy Loss" if col_idx == 0 else "")
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    if n_cols > 0 and axs[0, 0].get_legend() is not None:
        axs[0, 0].legend(title="Transfer Manipulation", loc="lower right", fontsize=8)

    plt.suptitle("Learning Trajectories by Architecture (Manipulations Separated, Averaging Across Seeds Only)", 
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig1_trajectories_by_architecture.png")
    pdf_path = os.path.join(savefigdir, "fig1_trajectories_by_architecture.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 1 (By Architecture) saved -> {png_path}")
    return png_path


# =====================================================================
# FIGURE 1B: Learning Trajectories Faceted by Transfer Mode
# =====================================================================
def plot_trajectories_by_transfer_mode(df, savefigdir=OUTPUT_DIR):
    """
    Multi-panel grid: Each transfer mode has its own panel showing:
    - Base
    - Width Growth
    - Head Growth
    - Both Growth
    Averaging is strictly across random seeds of the same condition!
    """
    modes = ['both', 'attn_only', 'mlp_only']
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey='row')

    for col_idx, mode in enumerate(modes):
        sub_mode = df[df["transfer_mode"] == mode]
        mode_title = TRANSFER_LABELS.get(mode, mode)

        # Top row: Accuracy
        ax_acc = axs[0, col_idx]
        sns.lineplot(
            data=sub_mode,
            x="iteration",
            y="top_3k_acc",
            hue="cond_label",
            linewidth=2.2,
            ax=ax_acc,
            legend=(col_idx == 0)
        )
        ax_acc.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_acc.set_title(f"Manipulation: {mode_title}", fontsize=12, fontweight="bold")
        ax_acc.set_ylabel("Avg Accuracy (Ranks 1-3000)" if col_idx == 0 else "")
        ax_acc.set_ylim(0.48, 1.0)
        ax_acc.grid(True, linestyle="--", alpha=0.4)

        # Bottom row: Loss
        ax_loss = axs[1, col_idx]
        sns.lineplot(
            data=sub_mode,
            x="iteration",
            y="train_loss",
            hue="cond_label",
            linewidth=2.2,
            ax=ax_loss,
            legend=False
        )
        ax_loss.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_loss.set_xlabel("Iteration", fontsize=11)
        ax_loss.set_ylabel("Cross-Entropy Loss" if col_idx == 0 else "")
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    if axs[0, 0].get_legend() is not None:
        axs[0, 0].legend(title="Architecture", loc="lower right", fontsize=8)

    plt.suptitle("Architecture Scaling by Transfer Mode (Averaging Across Seeds Only)", 
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig1_trajectories_by_transfer_mode.png")
    pdf_path = os.path.join(savefigdir, "fig1_trajectories_by_transfer_mode.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 1 (By Transfer Mode) saved -> {png_path}")
    return png_path


# =====================================================================
# FIGURE 1C: Comprehensive All-Conditions Trajectory Plot
# =====================================================================
def plot_trajectories_all_conditions(df, savefigdir=OUTPUT_DIR):
    """
    Single comprehensive figure where EVERY experimental condition has its own line:
    - Color = Architecture
    - Linestyle = Transfer mode (Solid for Both, Dashed for Attn Only, Dotted for MLP Only)
    - Averaged strictly across seeds!
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6.5))

    # Accuracy
    sns.lineplot(
        data=df,
        x="iteration",
        y="top_3k_acc",
        hue="cond_label",
        style="mode_label",
        linewidth=2.0,
        ax=axs[0]
    )
    axs[0].axvline(50000, color="crimson", linestyle="--", linewidth=1.8, alpha=0.9)
    axs[0].text(51000, 0.50, "Epoch 50 Resampling", color="crimson", fontweight="bold", fontsize=9)
    axs[0].set_title("Top 3,000 Accuracy (Every Condition Plotted Separately)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Iteration", fontsize=11)
    axs[0].set_ylabel("Avg Accuracy (Ranks 1-3000)", fontsize=11)
    axs[0].set_ylim(0.48, 1.0)
    axs[0].grid(True, linestyle="--", alpha=0.4)
    axs[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axs[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # Loss
    sns.lineplot(
        data=df,
        x="iteration",
        y="train_loss",
        hue="cond_label",
        style="mode_label",
        linewidth=2.0,
        ax=axs[1],
        legend=False
    )
    axs[1].axvline(50000, color="crimson", linestyle="--", linewidth=1.8, alpha=0.9)
    axs[1].set_title("Cross-Entropy Loss (Every Condition Plotted Separately)", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Iteration", fontsize=11)
    axs[1].set_ylabel("Loss", fontsize=11)
    axs[1].grid(True, linestyle="--", alpha=0.4)
    axs[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig1_trajectories_all_conditions.png")
    pdf_path = os.path.join(savefigdir, "fig1_trajectories_all_conditions.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 1 (All Conditions) saved -> {png_path}")
    return png_path


def plot_trajectories(df, savefigdir=OUTPUT_DIR):
    """Wrapper that generates all 3 trajectory visualizations."""
    p1 = plot_trajectories_by_architecture(df, savefigdir)
    p2 = plot_trajectories_by_transfer_mode(df, savefigdir)
    p3 = plot_trajectories_all_conditions(df, savefigdir)
    # Also save the multi-panel as default fig1_learning_trajectories.png for backward compatibility
    import shutil
    shutil.copyfile(p1, os.path.join(savefigdir, "fig1_learning_trajectories.png"))
    shutil.copyfile(p1.replace(".png", ".pdf"), os.path.join(savefigdir, "fig1_learning_trajectories.pdf"))
    return p1


# =====================================================================
# FIGURE 2: Memorization Capacity vs. Model Parameters (All Conditions)
# =====================================================================
def plot_capacity_scaling(df, savefigdir=OUTPUT_DIR):
    """
    Capacity Scaling: Plots EVERY experimental condition individually with distinct markers/colors.
    Plots separate scaling trend lines for Both vs Attn Only vs MLP Only!
    """
    latest_df = df[df["epoch"] == df["epoch"].max()].copy()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    markers = {"both": "o", "attn_only": "^", "mlp_only": "s"}
    colors = {
        'base': '#1f77b4',
        'width_growth': '#d62728',
        'head_growth': '#2ca02c',
        'both_growth': '#ff7f0e',
        'base_width_growth': '#8c564b',
        'base_head_growth': '#17becf',
        'base_both_growth': '#9467bd'
    }

    # Plot individual conditions (mean across seeds, plus individual seed dots)
    cond_groups = latest_df.groupby(["condition", "transfer_mode"])
    for (cond, mode), sub in cond_groups:
        mean_cap = sub["max_rank_90"].mean()
        std_cap = sub["max_rank_90"].std()
        n_params = sub["num_parameters"].iloc[0]
        color = colors.get(cond, "#333333")
        marker = markers.get(mode, "o")

        # Individual seeds as small faint dots
        ax.scatter(
            sub["num_parameters"],
            sub["max_rank_90"],
            marker=marker,
            color=color,
            s=40,
            alpha=0.35,
            edgecolors="none"
        )

        # Condition mean as large solid marker
        ax.scatter(
            [n_params],
            [mean_cap],
            marker=marker,
            color=color,
            s=140,
            alpha=0.9,
            edgecolors="k",
            linewidths=1.0,
            label=f"{CONDITION_LABELS.get(cond, cond)} [{TRANSFER_LABELS.get(mode, mode)}]"
        )

    # Add distinct trend lines for each transfer mode across growth architectures
    for mode, style, col in [("both", "-", "black"), ("attn_only", "--", "crimson"), ("mlp_only", ":", "navy")]:
        sub_mode = latest_df[latest_df["transfer_mode"] == mode].groupby("num_parameters")["max_rank_90"].mean().reset_index()
        if len(sub_mode) >= 2:
            log_x = np.log10(sub_mode["num_parameters"])
            log_y = np.log10(np.maximum(sub_mode["max_rank_90"], 1))
            slope, intercept = np.polyfit(log_x, log_y, 1)
            x_vals = np.linspace(sub_mode["num_parameters"].min(), sub_mode["num_parameters"].max(), 50)
            y_vals = 10**(slope * np.log10(x_vals) + intercept)
            ax.plot(x_vals, y_vals, linestyle=style, color=col, linewidth=1.8, alpha=0.8,
                    label=f"Fit ({TRANSFER_LABELS[mode]}): Slope = {slope:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Model Parameters", fontsize=12)
    ax.set_ylabel("Max Memorized Sequence Rank (Acc >= 0.90)", fontsize=12)
    ax.set_title("Capacity Scaling by Condition (Averaged Over Seeds Only)", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig2_capacity_scaling.png")
    pdf_path = os.path.join(savefigdir, "fig2_capacity_scaling.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 2 saved -> {png_path}")
    return png_path


# =====================================================================
# FIGURE 3: Transfer Ablation (All Conditions Bar + Strip Plot)
# =====================================================================
def plot_transfer_ablation(df, savefigdir=OUTPUT_DIR):
    """
    Barplot comparing EVERY architecture and EVERY transfer mode:
    - Left: Max Rank Capacity (Acc >= 0.90)
    - Right: Final Phase 2 Cross-Entropy Loss
    - Plus stripplot overlay showing every individual seed dot!
    """
    latest_df = df[df["epoch"] == df["epoch"].max()].copy()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Bar 1: Max Rank Capacity
    sns.barplot(
        data=latest_df,
        x="cond_label",
        y="max_rank_90",
        hue="mode_label",
        palette="Set2",
        edgecolor="black",
        linewidth=1,
        capsize=0.08,
        ax=axs[0]
    )
    sns.stripplot(
        data=latest_df,
        x="cond_label",
        y="max_rank_90",
        hue="mode_label",
        dodge=True,
        color="black",
        alpha=0.6,
        size=6,
        ax=axs[0],
        legend=False
    )
    axs[0].set_title("Capacity: Max Learned Sequence Rank (Acc >= 0.90)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Architecture", fontsize=11)
    axs[0].set_ylabel("Sequence Rank", fontsize=11)
    axs[0].tick_params(axis="x", rotation=15)
    axs[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Bar 2: Final Loss
    sns.barplot(
        data=latest_df,
        x="cond_label",
        y="train_loss",
        hue="mode_label",
        palette="Set2",
        edgecolor="black",
        linewidth=1,
        capsize=0.08,
        ax=axs[1]
    )
    sns.stripplot(
        data=latest_df,
        x="cond_label",
        y="train_loss",
        hue="mode_label",
        dodge=True,
        color="black",
        alpha=0.6,
        size=6,
        ax=axs[1],
        legend=False
    )
    axs[1].set_title(f"Final Phase 2 Cross-Entropy Loss (Epoch {latest_df['epoch'].max()})", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Architecture", fontsize=11)
    axs[1].set_ylabel("Loss", fontsize=11)
    axs[1].tick_params(axis="x", rotation=15)
    axs[1].grid(axis="y", linestyle="--", alpha=0.5)

    axs[0].legend(title="Transfer Manipulation", loc="upper left", fontsize=9)
    axs[1].legend(title="Transfer Manipulation", loc="upper right", fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig3_transfer_ablation.png")
    pdf_path = os.path.join(savefigdir, "fig3_transfer_ablation.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 3 saved -> {png_path}")
    return png_path


# =====================================================================
# FIGURE 4: Sequence Rank-Accuracy Curves (Grid by Architecture)
# =====================================================================
def plot_rank_accuracy_profiles(df_profile, savefigdir=OUTPUT_DIR):
    """
    Faceted Grid: Plots Accuracy vs Sequence Rank for each architecture separately,
    showing Both vs Attn Only vs MLP Only curves with ZERO averaging across modes!
    """
    if len(df_profile) == 0:
        print("[Warning] No profile points to plot for Figure 4.")
        return None

    arch_order = [c for c in ['both_growth', 'head_growth', 'width_growth', 'base', 
                              'base_both_growth', 'base_head_growth', 'base_width_growth'] 
                  if c in df_profile['condition'].unique()]
    
    n_cols = len(arch_order)
    fig, axs = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5.0), sharey=True)
    if n_cols == 1:
        axs = [axs]

    mode_palette = {"Both (Attn + MLP)": "#2b5c8f", "Attention Only": "#e05a47", "MLP Only": "#4daf4a"}

    for idx, cond in enumerate(arch_order):
        sub_arch = df_profile[df_profile["condition"] == cond]
        ax = axs[idx]

        sns.lineplot(
            data=sub_arch,
            x="sequence_rank",
            y="accuracy",
            hue="mode_label",
            style="mode_label",
            dashes=False,
            palette=mode_palette,
            linewidth=2.2,
            ax=ax,
            legend=(idx == 0)
        )
        ax.axhline(0.90, color="gray", linestyle=":", label="90% Threshold" if idx == 0 else "")
        ax.set_xscale("log")
        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=11, fontweight="bold")
        ax.set_xlabel("Sequence Rank (Zipf Tail)", fontsize=10)
        ax.set_ylabel("Accuracy" if idx == 0 else "")
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if axs[0].get_legend() is not None:
        axs[0].legend(title="Transfer Manipulation", loc="lower left", fontsize=8)

    plt.suptitle("Phase 2 Rank-Accuracy Memorization Curves (By Architecture, Separated by Manipulation)", 
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig4_rank_accuracy_profiles.png")
    pdf_path = os.path.join(savefigdir, "fig4_rank_accuracy_profiles.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 4 saved -> {png_path}")
    return png_path


def print_summary_table(df):
    """Print markdown-ready empirical summary of latest epoch metrics for every single condition."""
    latest_df = df[df["epoch"] == df["epoch"].max()]
    summary = latest_df.groupby(["condition", "transfer_mode"]).agg(
        num_params=("num_parameters", "first"),
        mean_loss=("train_loss", "mean"),
        std_loss=("train_loss", "std"),
        mean_top3k_acc=("top_3k_acc", "mean"),
        std_top3k_acc=("top_3k_acc", "std"),
        mean_max_rank_90=("max_rank_90", "mean"),
        std_max_rank_90=("max_rank_90", "std"),
        n_seeds=("seed", "count")
    ).reset_index()

    print("\n" + "=" * 95)
    print(f"EMPIRICAL SUMMARY AT EPOCH {latest_df['epoch'].max()} (ALL CONDITIONS SEPARATED, SEED STATS ONLY)")
    print("=" * 95)
    print(f"{'Condition':<20} | {'Transfer Mode':<12} | {'Params':<10} | {'Train Loss':<14} | {'Top-3k Acc':<14} | {'Max Rank (90%)':<16} | {'Seeds'}")
    print("-" * 95)
    for _, row in summary.iterrows():
        loss_str = f"{row['mean_loss']:.4f}±{row['std_loss']:.4f}" if pd.notnull(row['std_loss']) else f"{row['mean_loss']:.4f}"
        acc_str = f"{row['mean_top3k_acc']:.3f}±{row['std_top3k_acc']:.3f}" if pd.notnull(row['std_top3k_acc']) else f"{row['mean_top3k_acc']:.3f}"
        rank_str = f"{row['mean_max_rank_90']:.0f}±{row['std_max_rank_90']:.0f}" if pd.notnull(row['std_max_rank_90']) else f"{row['mean_max_rank_90']:.0f}"
        print(f"{row['condition']:<20} | {row['transfer_mode']:<12} | {row['num_params']:<10,d} | {loss_str:<14} | {acc_str:<14} | {rank_str:<16} | {row['n_seeds']}")
    print("=" * 95 + "\n")


def parse_args(argv=None):
    """argv=None parses sys.argv (CLI use); pass a list (e.g. []) to call from a notebook."""
    parser = argparse.ArgumentParser(description="Generate publication figures for model growth experiment")
    parser.add_argument(
        "--fig",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all", "trajectory", "capacity", "ablation", "profile"],
        help="Figure number or name to produce (1=trajectory, 2=capacity, 3=ablation, 4=profile, all=all)"
    )
    parser.add_argument("--force-reload", action="store_true", help="Force re-parsing cache folders instead of using cache")
    parser.add_argument("--table", action="store_true", help="Print summary metrics table")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = load_experiment_data(force_reload=args.force_reload)
    df = bundle["trajectory"]
    df_profile = bundle["rank_profiles"]

    fig_choice = str(args.fig).lower()

    if fig_choice in ["1", "trajectory", "all"]:
        plot_trajectories(df)
    if fig_choice in ["2", "capacity", "all"]:
        plot_capacity_scaling(df)
    if fig_choice in ["3", "ablation", "all"]:
        plot_transfer_ablation(df)
    if fig_choice in ["4", "profile", "all"]:
        plot_rank_accuracy_profiles(df_profile)

    if args.table or fig_choice == "all":
        print_summary_table(df)


if __name__ == "__main__":
    main()
