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

PLOTTING PRINCIPLES:
1. Zero cross-condition averaging: Every manipulation (Architecture, Transfer Mode, Init Type)
   is plotted individually.
2. Error bands and bars represent variance STRICTLY across random seeds of the identical condition.
3. Individual seed replications are displayed (scatter dots / stripplots) for full transparency.
4. Direct comparison: Transferred initialization is directly compared with base initialization
   controls of the identical size.

Usage CLI:
  python plot_growth_experiment.py --fig 1       # Learning Trajectories
  python plot_growth_experiment.py --fig 2       # Capacity Scaling (All conditions & tracks)
  python plot_growth_experiment.py --fig 3       # Transfer & Initialization Ablation
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

MODE_PALETTE = {
    'Both (Attn + MLP)': '#2b5c8f',
    'Attention Only': '#e05a47',
    'MLP Only': '#2ca02c'
}

COND_COLORS = {
    'base': '#1f77b4',
    'width_growth': '#d62728',
    'head_growth': '#2ca02c',
    'both_growth': '#ff7f0e',
    'base_width_growth': '#8c564b',
    'base_head_growth': '#17becf',
    'base_both_growth': '#9467bd'
}


def load_experiment_data(force_reload=False):
    """Load and process run logs into a tidy DataFrame with incremental update support."""
    if not force_reload and os.path.exists(DF_CACHE_FILE):
        with open(DF_CACHE_FILE, "rb") as f:
            bundle = pickle.load(f)
        df = bundle["trajectory"]
        # Check if cache is fully up to date (all conditions have max epoch >= 95)
        if df.groupby("condition")["epoch"].max().min() >= 95 and df["run_name"].nunique() >= 63:
            return bundle
        print("Cached data contains incomplete runs; performing incremental update...")
        old_df = df
        old_profile = bundle.get("rank_profiles", pd.DataFrame())
    else:
        old_df = pd.DataFrame()
        old_profile = pd.DataFrame()

    folders = sorted(glob.glob(f"{CACHE_DIR}/*"))
    print(f"Checking {len(folders)} run folders from {CACHE_DIR}...")

    # Runs that already have >= 95 epochs
    completed_runs = set()
    if len(old_df) > 0 and not force_reload:
        completed_runs = set(old_df.groupby("run_name")["epoch"].max()[lambda x: x >= 95].index)
        clean_df = old_df[old_df["run_name"].isin(completed_runs)].copy()
        clean_profile = old_profile[old_profile["run_name"].isin(completed_runs)].copy() if "run_name" in old_profile.columns else old_profile
    else:
        clean_df = pd.DataFrame()
        clean_profile = pd.DataFrame()

    new_records = []
    new_profiles = []
    t0 = time.time()

    for idx, folder in enumerate(folders):
        run_name = os.path.basename(folder)
        if run_name in completed_runs:
            continue
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
                                new_profiles.append({
                                    "run_name": run_name,
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

                new_records.append({
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

    df = pd.concat([clean_df, pd.DataFrame(new_records)], ignore_index=True)
    df_profile = pd.concat([clean_profile, pd.DataFrame(new_profiles)], ignore_index=True)
    print(f"Loaded {len(df)} records across {df['condition'].nunique()} conditions in {time.time() - t0:.2f}s.")

    data_bundle = {"trajectory": df, "rank_profiles": df_profile}
    with open(DF_CACHE_FILE, "wb") as f:
        pickle.dump(data_bundle, f)
    return data_bundle


# =====================================================================
# FIGURE 1: LEARNING TRAJECTORIES
# =====================================================================

def plot_trajectories_transferred_vs_base(df, savefigdir=OUTPUT_DIR):
    """
    Figure 1A: Direct Side-by-Side Comparison of Transferred vs Base Init.
    Columns:
      1. Width Expansion (64d, 8h)  -> Width Growth (Transferred) vs Width (Base Init)
      2. Head Expansion (32d, 16h)  -> Head Growth (Transferred) vs Head (Base Init)
      3. Joint Expansion (64d, 16h) -> Both Growth (Transferred) vs Both (Base Init)
    Rows:
      Top: Top-3,000 Accuracy
      Bottom: Cross-Entropy Loss
    Zero averaging across transfer modes or initialization types!
    """
    comps = [
        ("Width Expansion (64d, 8h)", "width_growth", "base_width_growth"),
        ("Head Expansion (32d, 16h)", "head_growth", "base_head_growth"),
        ("Joint Expansion (64d, 16h)", "both_growth", "base_both_growth")
    ]
    fig, axs = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True, sharey='row', squeeze=False)

    for col_idx, (title, grow_cond, base_cond) in enumerate(comps):
        sub = df[df["condition"].isin([grow_cond, base_cond])].copy()
        sub["Init & Mode"] = sub.apply(
            lambda r: f"{'Transferred' if 'base_' not in r['condition'] else 'Base Init'} - {TRANSFER_LABELS.get(r['transfer_mode'], r['transfer_mode'])}",
            axis=1
        )

        palette = {
            "Transferred - Both (Attn + MLP)": "#ff7f0e",
            "Transferred - Attention Only": "#d62728",
            "Transferred - MLP Only": "#e377c2",
            "Base Init - Both (Attn + MLP)": "#1f77b4",
            "Base Init - Attention Only": "#17becf",
            "Base Init - MLP Only": "#7f7f7f",
        }

        # Top row: Accuracy
        ax_acc = axs[0, col_idx]
        sns.lineplot(
            data=sub,
            x="iteration",
            y="top_3k_acc",
            hue="Init & Mode",
            style="Init & Mode",
            palette=palette,
            linewidth=2.0,
            ax=ax_acc,
            legend=(col_idx == 0)
        )
        ax_acc.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_acc.set_title(title, fontsize=12, fontweight="bold")
        ax_acc.set_ylabel("Avg Accuracy (Ranks 1-3000)" if col_idx == 0 else "")
        ax_acc.set_ylim(0.48, 1.0)
        ax_acc.grid(True, linestyle="--", alpha=0.4)

        # Bottom row: Loss
        ax_loss = axs[1, col_idx]
        sns.lineplot(
            data=sub,
            x="iteration",
            y="train_loss",
            hue="Init & Mode",
            style="Init & Mode",
            palette=palette,
            linewidth=2.0,
            ax=ax_loss,
            legend=False
        )
        ax_loss.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_loss.set_xlabel("Iteration", fontsize=11)
        ax_loss.set_ylabel("Cross-Entropy Loss" if col_idx == 0 else "")
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    if axs[0, 0].get_legend() is not None:
        axs[0, 0].legend(title="Initialization & Transfer Mode", loc="lower right", fontsize=8)

    plt.suptitle("Transferred Init vs. Base Init Control: Learning Trajectories (Seeds Averaged Only)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig1_trajectories_transferred_vs_base.png")
    pdf_path = os.path.join(savefigdir, "fig1_trajectories_transferred_vs_base.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 1 (Transferred vs Base) saved -> {png_path}")
    return png_path


def plot_trajectories_by_architecture(df, savefigdir=OUTPUT_DIR):
    """
    Figure 1B: Learning Trajectories Faceted by Architecture (7 columns: Growth + Base Controls).
    Compares Both vs Attn Only vs MLP Only within each architecture.
    """
    arch_order = ['both_growth', 'head_growth', 'width_growth', 'base',
                  'base_both_growth', 'base_head_growth', 'base_width_growth']
    arch_order = [c for c in arch_order if c in df['condition'].unique()]
    n_arch = len(arch_order)

    fig, axs = plt.subplots(2, n_arch, figsize=(4.2 * n_arch, 8.0), sharex=True, sharey='row', squeeze=False)

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
            palette=MODE_PALETTE,
            linewidth=2.2,
            ax=ax_acc,
            legend=(col_idx == 0)
        )
        ax_acc.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_acc.set_title(arch_title, fontsize=10, fontweight="bold")
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
            palette=MODE_PALETTE,
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


def plot_trajectories_by_transfer_mode(df, savefigdir=OUTPUT_DIR):
    """
    Figure 1C: Learning Trajectories Faceted by Transfer Mode (3 columns).
    Shows architectural scaling within each transfer protocol.
    """
    modes = ['both', 'attn_only', 'mlp_only']
    fig, axs = plt.subplots(2, 3, figsize=(18, 8.0), sharex=True, sharey='row', squeeze=False)

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
            linewidth=2.0,
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
            linewidth=2.0,
            ax=ax_loss,
            legend=False
        )
        ax_loss.axvline(50000, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
        ax_loss.set_xlabel("Iteration", fontsize=11)
        ax_loss.set_ylabel("Cross-Entropy Loss" if col_idx == 0 else "")
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    if axs[0, 0].get_legend() is not None:
        axs[0, 0].legend(title="Architecture & Init", loc="lower right", fontsize=8)

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


def plot_trajectories_all_conditions(df, savefigdir=OUTPUT_DIR):
    """
    Figure 1D: All 21 conditions plotted individually on a single comprehensive plot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(19, 7.0))

    sns.lineplot(
        data=df,
        x="iteration",
        y="top_3k_acc",
        hue="cond_label",
        style="mode_label",
        linewidth=1.8,
        ax=axs[0]
    )
    axs[0].axvline(50000, color="crimson", linestyle="--", linewidth=1.8, alpha=0.9)
    axs[0].text(51000, 0.50, "Epoch 50 Resampling", color="crimson", fontweight="bold", fontsize=9)
    axs[0].set_title("Top 3,000 Accuracy (All 21 Conditions Separated)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Iteration", fontsize=11)
    axs[0].set_ylabel("Avg Accuracy (Ranks 1-3000)", fontsize=11)
    axs[0].set_ylim(0.48, 1.0)
    axs[0].grid(True, linestyle="--", alpha=0.4)
    axs[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axs[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5)

    sns.lineplot(
        data=df,
        x="iteration",
        y="train_loss",
        hue="cond_label",
        style="mode_label",
        linewidth=1.8,
        ax=axs[1],
        legend=False
    )
    axs[1].axvline(50000, color="crimson", linestyle="--", linewidth=1.8, alpha=0.9)
    axs[1].set_title("Cross-Entropy Loss (All 21 Conditions Separated)", fontsize=12, fontweight="bold")
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
    """Wrapper that generates all trajectory visualizations."""
    p_comp = plot_trajectories_transferred_vs_base(df, savefigdir)
    p_arch = plot_trajectories_by_architecture(df, savefigdir)
    p_mode = plot_trajectories_by_transfer_mode(df, savefigdir)
    p_all = plot_trajectories_all_conditions(df, savefigdir)
    # Default Figure 1
    import shutil
    shutil.copyfile(p_comp, os.path.join(savefigdir, "fig1_learning_trajectories.png"))
    shutil.copyfile(p_comp.replace(".png", ".pdf"), os.path.join(savefigdir, "fig1_learning_trajectories.pdf"))
    return p_comp


# =====================================================================
# FIGURE 2: CAPACITY SCALING (NO CROSS-CONDITION AVERAGING)
# =====================================================================

def plot_capacity_scaling(df, savefigdir=OUTPUT_DIR):
    """
    Figure 2A: Capacity Scaling vs Parameters.
    - Plots EVERY condition individually with distinct markers/colors.
    - Resolves overlapping parameter counts (width vs head, transferred vs base init) with distinct offsets.
    - Faint dots show each individual seed replication.
    - Distinct trendlines for Width, Head, Joint growth tracks and Base Controls.
    - NEVER averages width and head growth together!
    """
    latest_epoch = df["epoch"].max()
    latest_df = df[df["epoch"] == latest_epoch].copy()

    fig, ax = plt.subplots(figsize=(11, 7.2))

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

    # Distinct horizontal multipliers so identical parameter counts never overlap
    param_offsets = {
        'width_growth': 0.94,
        'base_width_growth': 0.88,
        'head_growth': 1.06,
        'base_head_growth': 1.14,
        'both_growth': 0.96,
        'base_both_growth': 1.04,
        'base': 1.0,
    }

    cond_groups = latest_df.groupby(["condition", "transfer_mode"])
    for (cond, mode), sub in cond_groups:
        mean_cap = sub["max_rank_90"].mean()
        base_n_params = sub["num_parameters"].iloc[0]
        offset = param_offsets.get(cond, 1.0)
        n_params = base_n_params * offset

        color = colors.get(cond, "#333333")
        marker = markers.get(mode, "o")

        # Individual seeds as small faint dots
        ax.scatter(
            sub["num_parameters"] * offset,
            sub["max_rank_90"],
            marker=marker,
            color=color,
            s=45,
            alpha=0.45,
            edgecolors="none"
        )

        # Condition mean as large solid marker
        ax.scatter(
            [n_params],
            [mean_cap],
            marker=marker,
            color=color,
            s=140,
            alpha=0.92,
            edgecolors="k",
            linewidths=1.0,
            label=f"{CONDITION_LABELS.get(cond, cond)} [{TRANSFER_LABELS.get(mode, mode)}]"
        )

    # Plot scaling tracks for 'both' transfer mode
    base_sub = latest_df[(latest_df["condition"] == "base") & (latest_df["transfer_mode"] == "both")]
    if len(base_sub) > 0:
        base_p = base_sub["num_parameters"].iloc[0]
        base_c = base_sub["max_rank_90"].mean()

        tracks = [
            ("Transferred Width (64d, 8h)", "width_growth", "both", "#d62728", "--"),
            ("Transferred Head (32d, 16h)", "head_growth", "both", "#2ca02c", "-."),
            ("Transferred Joint (64d, 16h)", "both_growth", "both", "#ff7f0e", "-"),
            ("Base Init Width (64d, 8h)", "base_width_growth", "both", "#8c564b", ":"),
            ("Base Init Head (32d, 16h)", "base_head_growth", "both", "#17becf", ":"),
            ("Base Init Joint (64d, 16h)", "base_both_growth", "both", "#9467bd", ":"),
        ]
        for tr_name, target_cond, target_mode, tr_col, tr_style in tracks:
            t_sub = latest_df[(latest_df["condition"] == target_cond) & (latest_df["transfer_mode"] == target_mode)]
            if len(t_sub) > 0:
                t_p = t_sub["num_parameters"].iloc[0]
                t_c = t_sub["max_rank_90"].mean()
                slope = (np.log10(max(t_c, 1)) - np.log10(max(base_c, 1))) / (np.log10(t_p) - np.log10(base_p))
                ax.plot([base_p, t_p], [base_c, t_c], linestyle=tr_style, color=tr_col, linewidth=1.8, alpha=0.85,
                        label=f"{tr_name} Fit (Slope={slope:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Model Parameters", fontsize=12)
    ax.set_ylabel("Max Memorized Sequence Rank (Acc >= 0.90)", fontsize=12)
    ax.set_title(f"Capacity Scaling by Manipulation Track (Epoch {latest_epoch}, Seeds Plotted Individually)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.03, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(savefigdir, "fig2_capacity_scaling.png")
    pdf_path = os.path.join(savefigdir, "fig2_capacity_scaling.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 2 saved -> {png_path}")

    # Also generate Figure 2B (Faceted by Transfer Mode)
    plot_capacity_scaling_by_mode(latest_df, savefigdir)
    return png_path


def plot_capacity_scaling_by_mode(latest_df, savefigdir=OUTPUT_DIR):
    """
    Figure 2B: Capacity Scaling Faceted by Transfer Mode (Both, Attn Only, MLP Only).
    Compares Transferred vs Base Init side-by-side within each transfer mode.
    """
    modes = ['both', 'attn_only', 'mlp_only']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)

    colors = {
        'base': '#1f77b4',
        'width_growth': '#d62728',
        'head_growth': '#2ca02c',
        'both_growth': '#ff7f0e',
        'base_width_growth': '#8c564b',
        'base_head_growth': '#17becf',
        'base_both_growth': '#9467bd'
    }
    param_offsets = {
        'width_growth': 0.95,
        'base_width_growth': 0.88,
        'head_growth': 1.05,
        'base_head_growth': 1.14,
        'both_growth': 0.96,
        'base_both_growth': 1.04,
        'base': 1.0
    }

    for col_idx, mode in enumerate(modes):
        ax = axs[col_idx]
        sub_mode = latest_df[latest_df["transfer_mode"] == mode]
        mode_title = TRANSFER_LABELS.get(mode, mode)

        for cond, sub in sub_mode.groupby("condition"):
            color = colors.get(cond, "#333333")
            offset = param_offsets.get(cond, 1.0)
            n_p = sub["num_parameters"].iloc[0] * offset
            mean_c = sub["max_rank_90"].mean()

            # Seed dots
            ax.scatter(sub["num_parameters"] * offset, sub["max_rank_90"], color=color, s=40, alpha=0.45)
            # Condition mean
            ax.scatter([n_p], [mean_c], color=color, s=130, edgecolors="k", linewidths=1.0,
                       label=CONDITION_LABELS.get(cond, cond))

        # Trajectories for Width, Head, Both tracks in this transfer mode
        base_sub = sub_mode[sub_mode["condition"] == "base"]
        if len(base_sub) > 0:
            bp = base_sub["num_parameters"].iloc[0]
            bc = base_sub["max_rank_90"].mean()
            tracks = [
                ("width_growth", "#d62728", "--"),
                ("head_growth", "#2ca02c", "-."),
                ("both_growth", "#ff7f0e", "-"),
                ("base_width_growth", "#8c564b", ":"),
                ("base_head_growth", "#17becf", ":"),
                ("base_both_growth", "#9467bd", ":"),
            ]
            for t_cond, col, style in tracks:
                t_sub = sub_mode[sub_mode["condition"] == t_cond]
                if len(t_sub) > 0:
                    tp = t_sub["num_parameters"].iloc[0]
                    tc = t_sub["max_rank_90"].mean()
                    ax.plot([bp, tp], [bc, tc], linestyle=style, color=col, linewidth=1.8, alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Manipulation: {mode_title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Model Parameters", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Max Memorized Rank (Acc >= 0.90)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        if col_idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Capacity Scaling by Manipulation Track (Faceted by Transfer Mode)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p_png = os.path.join(savefigdir, "fig2_capacity_by_mode.png")
    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 2B (By Mode) saved -> {p_png}")


# =====================================================================
# FIGURE 3: TRANSFER & INITIALIZATION ABLATION
# =====================================================================

def plot_transfer_ablation(df, savefigdir=OUTPUT_DIR):
    """
    Figure 3: Component Transfer & Initialization Ablation.
    Compares EVERY architecture and EVERY transfer mode with individual seed dots overlaid.
    Left: Max Rank Capacity (Acc >= 0.90)
    Right: Final Phase 2 Cross-Entropy Loss
    Zero averaging across conditions!
    """
    latest_epoch = df["epoch"].max()
    latest_df = df[df["epoch"] == latest_epoch].copy()

    cond_order = [
        'base',
        'width_growth', 'base_width_growth',
        'head_growth', 'base_head_growth',
        'both_growth', 'base_both_growth'
    ]
    cond_labels = [CONDITION_LABELS.get(c, c) for c in cond_order if c in latest_df['condition'].unique()]

    fig, axs = plt.subplots(1, 2, figsize=(18, 6.5))

    # Bar 1: Max Rank Capacity
    sns.barplot(
        data=latest_df,
        x="cond_label",
        y="max_rank_90",
        hue="mode_label",
        order=cond_labels,
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
        order=cond_labels,
        dodge=True,
        palette="dark:black",
        alpha=0.65,
        size=6,
        ax=axs[0],
        legend=False
    )
    axs[0].set_title("Capacity: Max Learned Sequence Rank (Acc >= 0.90)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Architecture & Initialization Condition", fontsize=11)
    axs[0].set_ylabel("Sequence Rank", fontsize=11)
    axs[0].tick_params(axis="x", rotation=25)
    axs[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Bar 2: Final Loss
    sns.barplot(
        data=latest_df,
        x="cond_label",
        y="train_loss",
        hue="mode_label",
        order=cond_labels,
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
        order=cond_labels,
        dodge=True,
        palette="dark:black",
        alpha=0.65,
        size=6,
        ax=axs[1],
        legend=False
    )
    axs[1].set_title(f"Final Phase 2 Cross-Entropy Loss (Epoch {latest_epoch})", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Architecture & Initialization Condition", fontsize=11)
    axs[1].set_ylabel("Loss", fontsize=11)
    axs[1].tick_params(axis="x", rotation=25)
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
# FIGURE 4: RANK-ACCURACY PROFILES
# =====================================================================

def plot_rank_accuracy_profiles(df_profile, savefigdir=OUTPUT_DIR):
    """
    Figure 4: Accuracy vs. Sequence Rank along the Zipf tail.
    - 4A: Faceted by Architecture (shows transfer mode curves for each size).
    - 4B: Faceted by Transfer Mode (shows architectural curves for each mode).
    - 4C: Direct Transferred Init vs. Base Init Comparison along sequence ranks.
    """
    if len(df_profile) == 0:
        print("[Warning] No profile points to plot for Figure 4.")
        return None

    latest_epoch = df_profile["epoch"].max()
    sub_profile = df_profile[df_profile["epoch"] == latest_epoch].copy()

    # --- View 1: Faceted by Architecture ---
    arch_order = ['both_growth', 'head_growth', 'width_growth', 'base',
                  'base_both_growth', 'base_head_growth', 'base_width_growth']
    arch_order = [c for c in arch_order if c in sub_profile['condition'].unique()]
    n_cols = len(arch_order)
    fig, axs = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 5.0), sharey=True, squeeze=False)

    for idx, cond in enumerate(arch_order):
        sub_arch = sub_profile[sub_profile["condition"] == cond]
        ax = axs[0, idx]

        sns.lineplot(
            data=sub_arch,
            x="sequence_rank",
            y="accuracy",
            hue="mode_label",
            style="mode_label",
            dashes=False,
            palette=MODE_PALETTE,
            linewidth=2.2,
            ax=ax,
            legend=(idx == 0)
        )
        ax.axhline(0.90, color="gray", linestyle=":", label="90% Threshold" if idx == 0 else "")
        ax.set_xscale("log")
        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10, fontweight="bold")
        ax.set_xlabel("Sequence Rank (Zipf Tail)", fontsize=10)
        ax.set_ylabel("Accuracy" if idx == 0 else "")
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if axs[0, 0].get_legend() is not None:
        axs[0, 0].legend(title="Transfer Manipulation", loc="lower left", fontsize=8)

    plt.suptitle(f"Phase 2 Rank-Accuracy Curves by Architecture (Epoch {latest_epoch}, Averaged Over Seeds Only)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    p1_png = os.path.join(savefigdir, "fig4_rank_accuracy_profiles_by_arch.png")
    fig.savefig(p1_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 4A (By Arch) saved -> {p1_png}")

    # --- View 2: Faceted by Transfer Mode ---
    modes = ['both', 'attn_only', 'mlp_only']
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5.0), sharey=True, squeeze=False)

    for idx, mode in enumerate(modes):
        sub_mode = sub_profile[sub_profile["transfer_mode"] == mode]
        ax = axs2[0, idx]

        sns.lineplot(
            data=sub_mode,
            x="sequence_rank",
            y="accuracy",
            hue="cond_label",
            linewidth=2.0,
            ax=ax,
            legend=(idx == 0)
        )
        ax.axhline(0.90, color="gray", linestyle=":", label="90% Threshold" if idx == 0 else "")
        ax.set_xscale("log")
        ax.set_title(f"Manipulation: {TRANSFER_LABELS.get(mode, mode)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sequence Rank (Zipf Tail)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if axs2[0, 0].get_legend() is not None:
        axs2[0, 0].legend(title="Architecture", loc="lower left", fontsize=8)

    plt.suptitle(f"Phase 2 Rank-Accuracy Curves by Transfer Mode (Epoch {latest_epoch}, Averaged Over Seeds Only)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    p2_png = os.path.join(savefigdir, "fig4_rank_accuracy_profiles_by_mode.png")
    fig2.savefig(p2_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 4B (By Mode) saved -> {p2_png}")

    # --- View 3: Direct Transferred Init vs Base Init Profile Comparison ---
    comps = [
        ("Width Expansion (64d, 8h)", "width_growth", "base_width_growth"),
        ("Head Expansion (32d, 16h)", "head_growth", "base_head_growth"),
        ("Joint Expansion (64d, 16h)", "both_growth", "base_both_growth")
    ]
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5.0), sharey=True, squeeze=False)
    for col_idx, (title, grow_cond, base_cond) in enumerate(comps):
        sub_c = sub_profile[sub_profile["condition"].isin([grow_cond, base_cond])].copy()
        sub_c["Condition"] = sub_c.apply(
            lambda r: f"{'Transferred' if 'base_' not in r['condition'] else 'Base Init'} ({TRANSFER_LABELS.get(r['transfer_mode'], r['transfer_mode'])})",
            axis=1
        )
        ax = axs3[0, col_idx]
        sns.lineplot(
            data=sub_c,
            x="sequence_rank",
            y="accuracy",
            hue="Condition",
            linewidth=2.0,
            ax=ax,
            legend=(col_idx == 0)
        )
        ax.axhline(0.90, color="gray", linestyle=":", label="90% Threshold" if col_idx == 0 else "")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sequence Rank (Zipf Tail)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if axs3[0, 0].get_legend() is not None:
        axs3[0, 0].legend(title="Condition", loc="lower left", fontsize=8)

    plt.suptitle("Transferred Init vs. Base Init Control: Rank-Accuracy Curves", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    p3_png = os.path.join(savefigdir, "fig4_rank_accuracy_transferred_vs_base.png")
    fig3.savefig(p3_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Done] Figure 4C (Transferred vs Base) saved -> {p3_png}")

    # Copy primary figure
    import shutil
    shutil.copyfile(p3_png, os.path.join(savefigdir, "fig4_rank_accuracy_profiles.png"))
    return p3_png


def print_summary_table(df):
    """Print markdown-ready empirical summary of latest completed epoch metrics."""
    latest_epoch = df["epoch"].max()
    latest_df = df[df["epoch"] == latest_epoch]
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
    print(f"EMPIRICAL SUMMARY AT EPOCH {latest_epoch} (PHASE 2 - POST-GROWTH & RESAMPLING, ALL RUNS)")
    print("=" * 95)
    print(f"{'Condition':<18} | {'Transfer Mode':<12} | {'Params':<10} | {'Train Loss':<14} | {'Top-3k Acc':<14} | {'Max Rank (90%)':<16} | {'Seeds'}")
    print("-" * 95)
    for _, row in summary.iterrows():
        loss_str = f"{row['mean_loss']:.4f}±{row['std_loss']:.4f}" if pd.notnull(row['std_loss']) else f"{row['mean_loss']:.4f}"
        acc_str = f"{row['mean_top3k_acc']:.3f}±{row['std_top3k_acc']:.3f}" if pd.notnull(row['std_top3k_acc']) else f"{row['mean_top3k_acc']:.3f}"
        rank_str = f"{row['mean_max_rank_90']:.0f}±{row['std_max_rank_90']:.0f}" if pd.notnull(row['std_max_rank_90']) else f"{row['mean_max_rank_90']:.0f}"
        print(f"{row['condition']:<18} | {row['transfer_mode']:<12} | {row['num_params']:<10,d} | {loss_str:<14} | {acc_str:<14} | {rank_str:<16} | {row['n_seeds']}")
    print("=" * 95 + "\n")


def parse_args(argv=None):
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
