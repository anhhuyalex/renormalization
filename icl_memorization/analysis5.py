import os
import sys
import shutil
import glob
import math
import pickle
from collections import defaultdict
import copy

# Configure environment to locate TeX tools and required libraries
fmri_root = os.path.dirname(os.path.dirname(sys.executable))
fmri_bin = os.path.join(fmri_root, "bin")
fmri_lib = os.path.join(fmri_root, "lib")

os.environ["PATH"] = fmri_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = fmri_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import seaborn as sns
import torch

import gpt
import utils


# =============================================================================
# FIGURE 1: Experimental Setup (Sequences + Zipf Sampling)
# =============================================================================
def create_figure1():
    """Create publication-quality Figure 1 for NeurIPS camera-ready submission.

    Combines:
      (a) Sequence visualizations (4 example binary sequences of length 20)
      (b) Zipfian sampling distribution (Empirical vs Expected counts on log-log scale)
    Aligned side-by-side with bold subpanel labels 'a' and 'b'.
    """
    utils.set_plot_style()

    fig = plt.figure(figsize=(10.5, 3.6))

    outer_gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.12, 1.0],
        wspace=0.30,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.16,
    )

    # Panel (a): Sequences visualization
    left_gs = outer_gs[0].subgridspec(4, 1, hspace=0.95)

    seqs = np.array([
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    ])

    for i in range(4):
        ax = fig.add_subplot(left_gs[i])
        ax.imshow(seqs[i : i + 1], aspect="equal", cmap="rocket", extent=[0, 20, 0, 1])
        ax.set_title(f"Sequence {i}", fontsize=11.5, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Panel (b): Zipf distribution
    ax_b = fig.add_subplot(outer_gs[1])

    K = 1000
    Pk = 1.0 / np.arange(1, K + 1)
    Pk = Pk / Pk.sum()

    n = 100000
    np.random.seed(42)
    samples = np.random.choice(np.arange(1, K + 1), size=n, p=Pk)
    counts = np.bincount(samples, minlength=K + 1)[1:]

    ranks = np.arange(1, K + 1)
    expected = n * Pk
    mask = counts > 0

    ax_b.loglog(ranks[mask], counts[mask], ".", ms=3.0, color="C0", alpha=0.85, label="Empirical")
    ax_b.loglog(ranks, expected, "-", lw=1.8, color="C1", label="Expected")

    ax_b.set_xlabel("Rank", fontsize=15)
    ax_b.set_ylabel("Count", fontsize=15)
    ax_b.set_title("Zipf Sampling", fontsize=15, pad=6)

    ax_b.set_xlim(0.8, 1500)
    ax_b.set_ylim(4, 25000)

    ax_b.tick_params(which="both", direction="in", top=True, right=True)
    ax_b.legend(
        loc="upper right",
        frameon=True,
        edgecolor="0.6",
        facecolor="white",
        framealpha=0.9,
        fontsize=12,
        handletextpad=0.5,
        borderpad=0.4,
    )

    fig.text(0.035, 0.96, r"\textbf{a}", fontsize=24, fontweight="bold", va="top")
    fig.text(0.355, 0.96, r"\textbf{b}", fontsize=24, fontweight="bold", va="top")

    return fig


# =============================================================================
# FIGURE 2 HELPERS: Data Extraction, Power Law Fitting & Subpanels
# =============================================================================
class FastParamUnpickler(pickle.Unpickler):
    """Bypasses full tensor buffer allocation while preserving logs, args,

    and exact parameter counts from tensor shape metadata (~50x speedup).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_count = 0

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: None
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def dummy_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
                numel = math.prod(size)
                self.param_count += numel
                return numel
            return dummy_tensor
        return super().find_class(module, name)


def fit_power_law(x, y):
    """Fits power law: y = C * x^gamma via linear regression in log-log space."""
    mask = (x > 0) & (y > 0)
    x_val = x[mask]
    y_val = y[mask]

    if len(x_val) < 2:
        return np.nan, np.nan, None, None

    log_x = np.log10(x_val)
    log_y = np.log10(y_val)

    gamma, log_c = np.polyfit(log_x, log_y, 1)

    x_fit = np.logspace(log_x.min(), log_x.max(), 100)
    y_fit = (10 ** log_c) * (x_fit ** gamma)

    return gamma, 10 ** log_c, x_fit, y_fit


def plot_train_loss_panel(ax, df_data, title="Train Loss vs. Number of Parameters", d_min=None, d_max=None):
    ax.set_title(title, fontsize=15, pad=8)
    """Plots Train Loss vs Number of Parameters with markers for H and colormap for D."""
    H_values = sorted(df_data["num_heads"].unique())
    markers = ["o", "X", "s", "P", "D", "^", "v", "<", ">", "*", "h", "8"]
    marker_map = {h: markers[i % len(markers)] for i, h in enumerate(H_values)}

    D_values = sorted(df_data["num_hidden_features"].unique())
    cmap = mpl.cm.turbo
    dmin = min(D_values) if d_min is None else d_min
    dmax = max(D_values) if d_max is None else d_max
    norm = mpl.colors.LogNorm(vmin=dmin, vmax=dmax)

    for h in H_values:
        dh = df_data[df_data["num_heads"] == h]
        if len(dh) == 0:
            continue
        ax.scatter(
            dh["num_parameters"],
            dh["train_loss"],
            c=dh["num_hidden_features"],
            cmap=cmap,
            norm=norm,
            marker=marker_map[h],
            s=55,
            alpha=0.9,
            edgecolors="none",
        )

    ax.set_xlabel("Number of Parameters", fontsize=14)
    ax.set_ylabel("Train Loss", fontsize=14)
    ax.set_xscale("log")
    ax.tick_params(which="both", direction="in", top=True, right=True)

    # Colorbar for D
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$D$", fontsize=14)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10))

    # Legend for H
    style_handles = [
        Line2D([0], [0], marker=marker_map[h], linestyle="None", color="black",
               markerfacecolor="black", markersize=6.5, label=str(h))
        for h in H_values
    ]
    ax.legend(
        handles=style_handles,
        title=r"$H$",
        loc="upper right",
        ncol=2,
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10.5,
        handletextpad=0.2,
        borderpad=0.25,
        columnspacing=0.5,
    )


def plot_power_law_panel(ax, capacity_df, iters_list, title, is_12layer=True):
    """Plots Maximum rank memorized vs Number of parameters with power law fits."""
    ax.set_title(title, fontsize=15, pad=8)
    colors = ["C0", "C1", "C2"]

    for i, iters in enumerate(iters_list):
        iter_col = "iters" if "iters" in capacity_df.columns else "iters_checkpoint"
        rank_col = "max_rank" if "max_rank" in capacity_df.columns else "max_rank_at_saturation"
        sub = capacity_df[capacity_df[iter_col] == iters].dropna(
            subset=["num_parameters", rank_col]
        )
        sub = sub[(sub["num_parameters"] > 0) & (sub[rank_col] > 0)]

        # Panel (b) strictly uses the exact same failure filters as Panel (a)

        p = sub["num_parameters"].values
        r = sub[rank_col].values
        c = colors[i % len(colors)]

        gamma, C, x_fit, y_fit = fit_power_law(p, r)

        ax.plot(p, r, "o", color=c, ms=6, alpha=0.9)
        if x_fit is not None:
            ax.plot(x_fit, y_fit, "--", color=c, lw=1.8, label=rf"$\gamma = {gamma:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    if is_12layer:
        ax.set_xlim(1e4, 1e7)
    else:
        ax.set_xlim(2e2, 5e8)
    ax.set_ylim(1e1, 1.2e5)
    ax.set_xlabel("Number of parameters", fontsize=14)
    ax.set_ylabel("Maximum rank memorized", fontsize=14)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="upper left", frameon=True, fontsize=11.5)


def plot_broken_axis_panel(ax1, ax2, avg_df, norm, title=None):
    """Plots broken axis accuracy vs iteration with data switch marker and break slashes."""
    left_df = avg_df[(avg_df["iteration"] >= 0) & (avg_df["iteration"] <= 10000)].copy()
    right_df = avg_df[(avg_df["iteration"] >= 40000) & (avg_df["iteration"] <= 50000)].copy()

    common_kwargs = dict(
        y="accuracy",
        hue="num_parameters",
        palette="turbo",
        hue_norm=norm,
        legend=False,
        errorbar=None,
    )

    sns.lineplot(
        data=left_df.sort_values(["num_parameters", "iteration"]),
        x="iteration",
        ax=ax1,
        **common_kwargs,
    )
    sns.lineplot(
        data=right_df.sort_values(["num_parameters", "iteration"]),
        x="iteration",
        ax=ax2,
        **common_kwargs,
    )

    # Red dashed line for data switch
    ax2.axvline(x=40000, color="red", linestyle="--", linewidth=1.5)
    ax2.text(40250, 0.93, "Data Switch", color="red", fontsize=8.5)

    ax1.set_xlim(0, 10000)
    ax2.set_xlim(40000, 50000)
    ax1.set_ylim(0.45, 1.0)

    ax1.set_ylabel("Accuracy", fontsize=14)
    ax2.set_ylabel("")
    ax1.set_xlabel("")
    ax2.set_xlabel("")

    # Broken axis tick formatting: [0.0, 0.5, 1.0] and [4.0, 4.5, 5.0]
    ax1.set_xticks([0, 5000, 10000])
    ax2.set_xticks([40000, 45000, 50000])
    ax1.set_xticklabels(["0.0", "0.5", "1.0"])
    ax2.set_xticklabels(["4.0", "4.5", "5.0"])

    left_labels = ax1.get_xticklabels()
    right_labels = ax2.get_xticklabels()
    left_labels[-1].set_horizontalalignment("right")
    right_labels[0].set_horizontalalignment("left")

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(labelleft=False, left=False)

    # Draw broken axis diagonal slash marks
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Shared common x-axis label between the two subplots
    ax1.text(
        1.04,
        -0.10,
        r"Iteration $\times 10^4$",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    if title:
        ax1.set_title(title, x=1.04, y=1.0, pad=8, fontsize=15)


# Figure 2 bottom-row data provenance.  Panels (a) and (b) use the one-layer
# no-resampling experiment; panel (c) uses the corresponding GPT-2 resampling
# experiment used for the data-switch evaluation.
FIGURE2_BOTTOM_AB_EXPERIMENT = (
    "memo_jan4_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample"
)
FIGURE2_BOTTOM_C_EXPERIMENT = (
    "memo_jan4_zipf_gpt2_vary_num_hidden_features_num_heads_doresample/"
)
FIGURE2_BOTTOM_MAX_PARAMETERS = 7e7
FIGURE2_BOTTOM_AB_RECORD_GLOB = (
    "/scratch/qanguyen/gautam/cache/"
    f"{FIGURE2_BOTTOM_AB_EXPERIMENT}/*"
)
FIGURE2_BOTTOM_A_CACHE = "./cache/df_train_loss_1layer_jan4_noresample.csv"
FIGURE2_BOTTOM_B_CACHE = "./cache/df_capacity_1layer_jan4_noresample.csv"
# This evaluation cache is keyed by the resampled GPT-2 sequences named above.
# That experiment stores sequence payloads rather than model checkpoint logs,
# so its accuracy trajectory cannot be re-derived from that directory alone.
FIGURE2_BOTTOM_C_CACHE = "./cache/df_dataswitch_1layer_jan4_doresample.csv"


def filter_figure2_bottom_row_models(dataframe):
    """Exclude bottom-row models larger than the camera-ready cutoff."""
    kept_indices = []
    for index, row in dataframe.iterrows():
        if row["num_parameters"] > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        kept_indices.append(index)
    return dataframe.loc[kept_indices].copy()


def regenerate_figure2_bottom_ab_cache():
    """Rebuild Figure 2 bottom-row panels (a, b) from their raw records.

    The fast unpickler reads model tensor shapes without allocating model
    weights, then extracts epoch-wise training loss and 90%-accuracy capacity
    from ``FIGURE2_BOTTOM_AB_EXPERIMENT``.  Models above the camera-ready
    parameter cutoff are excluded before either cache is written.
    """
    loss_rows = []
    capacity_rows = []
    folders = sorted(glob.glob(FIGURE2_BOTTOM_AB_RECORD_GLOB))
    print(
        "Regenerating Figure 2 bottom-row caches from "
        f"{FIGURE2_BOTTOM_AB_EXPERIMENT} ({len(folders)} runs)"
    )

    for folder in folders:
        record_paths = [
            path for path in glob.glob(os.path.join(folder, "*.pkl"))
            if not path.endswith("_all_sequences.pkl")
        ]
        if len(record_paths) != 1:
            continue
        try:
            with open(record_paths[0], "rb") as file:
                record = FastParamUnpickler(file)
                data = record.load()
        except Exception as error:
            print(f"Skipping unreadable record {record_paths[0]}: {error}")
            continue

        model_parameters = sum(
            value for value in data.get("model", {}).values()
            if isinstance(value, (int, np.integer))
        )
        if model_parameters > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue

        args = data.get("args", {})
        num_heads = args.get("num_heads")
        num_hidden_features = args.get("num_hidden_features")
        for log in data.get("logs", []):
            epoch = log.get("epoch")
            if epoch is None or "train_loss" not in log:
                continue
            test_metrics = log.get("test_metrics", {})
            accuracy = test_metrics.get("accuracy")
            test_accuracy = (
                float(np.asarray(accuracy).mean()) if accuracy is not None else np.nan
            )
            loss_rows.append({
                "train_loss": log["train_loss"],
                "test_accuracy": test_accuracy,
                "num_parameters": model_parameters,
                "num_hidden_features": num_hidden_features,
                "num_heads": num_heads,
                "epoch": epoch,
            })

            if epoch not in [20, 50, 79]:
                continue
            sequence_ranks = test_metrics.get("sequence_rank")
            if accuracy is None or sequence_ranks is None:
                continue
            accuracy_array = np.asarray(accuracy)
            rank_array = np.asarray(sequence_ranks) * 10
            learned_ranks = rank_array[accuracy_array > 0.90]
            capacity_rows.append({
                "num_parameters": model_parameters,
                "num_heads": num_heads,
                "num_hidden_features": num_hidden_features,
                "epoch": epoch,
                "iters": epoch * 1000,
                "max_rank": int(learned_ranks.max()) if len(learned_ranks) else 0,
            })

    loss_frame = pd.DataFrame(loss_rows)
    capacity_frame = pd.DataFrame(capacity_rows)
    if loss_frame.empty or capacity_frame.empty:
        raise RuntimeError(
            "No usable Figure 2 bottom-row records were found in "
            f"{FIGURE2_BOTTOM_AB_RECORD_GLOB}."
        )

    loss_frame.to_csv(FIGURE2_BOTTOM_A_CACHE, index=False)
    capacity_frame.to_csv(FIGURE2_BOTTOM_B_CACHE, index=False)
    print(f"Wrote {len(loss_frame)} rows to {FIGURE2_BOTTOM_A_CACHE}")
    print(f"Wrote {len(capacity_frame)} rows to {FIGURE2_BOTTOM_B_CACHE}")


# =============================================================================
# FIGURE 2: Full Assembly (2 rows x 3 columns)
# =============================================================================
def create_figure2():
    """Create publication-quality Figure 2 for NeurIPS submission.
    Combines:
      Column (a): Final train loss vs parameter count (12-layer top, 1-layer bottom)
      Column (b): Maximum rank memorized vs parameter count with power law fits
                  (12-layer top at 60k, 120k, 190k; 1-layer bottom at 20k, 50k, 79k)
      Column (c): Average accuracy over first 3000 ranked sequences across data switch
                  with broken axis [0, 1.0] and [4.0, 5.0] x 10^4 (12-layer top, 1-layer bottom)
    """
    utils.set_plot_style()
    print(
        "Figure 2 bottom-row panels (a, b) experiment: "
        f"{FIGURE2_BOTTOM_AB_EXPERIMENT}"
    )
    print(
        "Figure 2 bottom-row panel (c) experiment: "
        f"{FIGURE2_BOTTOM_C_EXPERIMENT}"
    )

    fig = plt.figure(figsize=(19, 11))
    outer_gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.05, 1.35],
        height_ratios=[1.0, 1.0],
        wspace=0.32,
        hspace=0.45,
        left=0.05,
        right=0.96,
        top=0.91,
        bottom=0.07,
    )

    # -------------------------------------------------------------------------
    # COLUMN 1: Panel (a) - Train Loss vs. Parameters
    # -------------------------------------------------------------------------
    ax_a0 = fig.add_subplot(outer_gs[0, 0])
    ax_a1 = fig.add_subplot(outer_gs[1, 0])

    # 12-layer train loss data (freshly extracted from memo_apr19)
    df_tl_12 = pd.read_csv("./cache/df_train_loss_12layer.csv")
    ok_12 = df_tl_12[~df_tl_12["failed"]]
    plot_train_loss_panel(
        ax_a0,
        ok_12,
        title="Train Loss vs. Number of Parameters",
        d_min=min(ok_12["num_hidden_features"]),
        d_max=max(ok_12["num_hidden_features"]),
    )

    # 1-layer train loss data from FIGURE2_BOTTOM_AB_EXPERIMENT.
    df_tl_1 = pd.read_csv(FIGURE2_BOTTOM_A_CACHE)
    df_tl_1 = df_tl_1[df_tl_1["epoch"] == 50]
    df_tl_1 = filter_figure2_bottom_row_models(df_tl_1)
    plot_train_loss_panel(
        ax_a1,
        df_tl_1,
        title="Train Loss vs. Number of Parameters",
        d_min=min(df_tl_1["num_hidden_features"]),
        d_max=max(df_tl_1["num_hidden_features"]),
    )

    # -------------------------------------------------------------------------
    # COLUMN 2: Panel (b) - Max Rank Memorized vs. Parameters
    # -------------------------------------------------------------------------
    ax_b0 = fig.add_subplot(outer_gs[0, 1])
    ax_b1 = fig.add_subplot(outer_gs[1, 1])

    # 12-layer capacity data (freshly extracted from memo_apr19, excluding memo_may1)
    df_cap_12 = pd.read_csv("./cache/df_capacity_12layer.csv")
    df_cap_12_ok = df_cap_12[~df_cap_12["failed"]] if "failed" in df_cap_12.columns else df_cap_12
    plot_power_law_panel(
        ax_b0,
        df_cap_12_ok,
        iters_list=[60000, 120000, 190000],
        title="Max Learned Rank vs. Number of Parameters",
        is_12layer=True,
    )

    # 1-layer capacity data from FIGURE2_BOTTOM_AB_EXPERIMENT.
    df_cap_1 = pd.read_csv(FIGURE2_BOTTOM_B_CACHE)
    df_cap_1 = filter_figure2_bottom_row_models(df_cap_1)
    plot_power_law_panel(
        ax_b1,
        df_cap_1,
        iters_list=[20000, 50000, 79000],
        title="Max Learned Rank vs. Number of Parameters",
        is_12layer=False,
    )

    # -------------------------------------------------------------------------
    # COLUMN 3: Panel (c) - Meta-Memorization Broken Axis Lineplots
    # -------------------------------------------------------------------------
    gs_c0 = outer_gs[0, 2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    ax_c0_left = fig.add_subplot(gs_c0[0])
    ax_c0_right = fig.add_subplot(gs_c0[1], sharey=ax_c0_left)

    gs_c1 = outer_gs[1, 2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    ax_c1_left = fig.add_subplot(gs_c1[0])
    ax_c1_right = fig.add_subplot(gs_c1[1], sharey=ax_c1_left)

    # 12-layer data switch data (exclusively from memo_may1_zipf_gpt2_vary_num_hidden_features_num_heads_doresample)
    avg_df_12 = pd.read_csv("./cache/df_dataswitch_12layer_may1.csv")

    # 1-layer data-switch panel from FIGURE2_BOTTOM_C_EXPERIMENT.
    avg_df_1 = pd.read_csv(FIGURE2_BOTTOM_C_CACHE)
    avg_df_1 = filter_figure2_bottom_row_models(avg_df_1)

    norm_c0 = LogNorm(
        vmin=avg_df_12["num_parameters"].min(),
        vmax=avg_df_12["num_parameters"].max(),
    )
    norm_c1 = LogNorm(
        vmin=avg_df_1["num_parameters"].min(),
        vmax=avg_df_1["num_parameters"].max(),
    )

    plot_broken_axis_panel(ax_c0_left, ax_c0_right, avg_df_12, norm=norm_c0,
                           title="Average Accuracy Over First 3000 Ranked Sequences")
    plot_broken_axis_panel(ax_c1_left, ax_c1_right, avg_df_1, norm=norm_c1,
                           title="Average Accuracy Over First 3000 Ranked Sequences")

    # Colorbars for Column (c)
    sm_c0 = mpl.cm.ScalarMappable(cmap="turbo", norm=norm_c0)
    sm_c0.set_array([])
    cbar_c0 = fig.colorbar(sm_c0, ax=[ax_c0_left, ax_c0_right], pad=0.02)
    cbar_c0.set_label("Number of Parameters", fontsize=13)
    cbar_c0.formatter = mticker.LogFormatterSciNotation()
    cbar_c0.update_ticks()

    sm_c1 = mpl.cm.ScalarMappable(cmap="turbo", norm=norm_c1)
    sm_c1.set_array([])
    cbar_c1 = fig.colorbar(sm_c1, ax=[ax_c1_left, ax_c1_right], pad=0.02)
    cbar_c1.set_label("Number of Parameters", fontsize=13)
    cbar_c1.formatter = mticker.LogFormatterSciNotation()
    cbar_c1.update_ticks()

    # -------------------------------------------------------------------------
    # Row Headers: 12-layer models (top row) & 1-layer models (bottom row)
    # -------------------------------------------------------------------------
    fig.text(0.5, 0.985, r"\textbf{12-layer models}", fontsize=20, ha="center", va="top")
    fig.text(0.5, 0.505, r"\textbf{1-layer models}", fontsize=20, ha="center", va="top")

    # -------------------------------------------------------------------------
    # Subpanel Labels: a, b, c
    # -------------------------------------------------------------------------
    fig.text(0.035, 0.96, r"\textbf{a}", fontsize=24, fontweight="bold", va="top")
    fig.text(0.355, 0.96, r"\textbf{b}", fontsize=24, fontweight="bold", va="top")
    fig.text(0.665, 0.96, r"\textbf{c}", fontsize=24, fontweight="bold", va="top")

    return fig


# =============================================================================
# FIGURE 5: Learned Attention Patterns
# =============================================================================
# This is the trained one-layer, 16-head model used for the attention-pattern
# visualization.  Keep the checkpoint selection explicit so that the figure is
# reproducible and cannot silently switch to a different experiment.
FIGURE5_RECORD_GLOB = (
    "/scratch/qanguyen/gautam/cache/"
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample/"
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample_"
    "transformer_K_100000_L_100_hidden_64_nheads_16*"
)


def create_figure5(sequence_idx=0, num_heads_to_plot=8,
                   record_glob=FIGURE5_RECORD_GLOB):
    """Create the camera-ready learned-attention visualization (Figure 5).

    The panel shows causal attention maps from the first held-out sequence of
    the 16-head, width-64 one-layer model.  A common [0, 1] colour scale makes
    attention magnitudes directly comparable across heads; the strict upper
    triangle is white because causal attention cannot attend to future keys.

    Args:
        sequence_idx: Index in the saved sequence batch to visualize.
        num_heads_to_plot: Number of heads to draw, up to eight (two rows of
            four panels).
        record_glob: Glob identifying the experiment directory containing the
            model record and its ``_all_sequences.pkl`` data file.

    Returns:
        The Matplotlib figure.  Saving is deliberately handled by ``main`` so
        callers can also embed the panel in a larger composition.
    """
    utils.set_plot_style()
    # Figure 5 has no mathematical text.  Use Matplotlib's embedded text
    # instead of a TeX subprocess so the camera-ready asset is portable across
    # rendering hosts.
    mpl.rcParams["text.usetex"] = False

    experiment_dirs = sorted(glob.glob(record_glob))
    if len(experiment_dirs) != 1:
        raise RuntimeError(
            "Figure 5 requires exactly one checkpoint directory matching "
            f"{record_glob!r}; found {len(experiment_dirs)}."
        )

    experiment_dir = experiment_dirs[0]
    experiment_name = os.path.basename(experiment_dir)
    record_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    data_path = os.path.join(experiment_dir, f"{experiment_name}_all_sequences.pkl")
    if not os.path.isfile(record_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Figure 5 checkpoint is incomplete. Expected both "
            f"{record_path!r} and {data_path!r}."
        )

    with open(record_path, "rb") as file:
        record = utils.CPU_Unpickler(file).load()
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    args = record["args"]
    model = gpt.OneLayerAttention(
        args["len_context"],
        args["num_heads"],
        args["num_hidden_features"],
        args["vocab_size"],
        args["num_mlp_layers"],
    )
    model.load_state_dict(record["model"])
    model.eval()

    sequences = data["sequences"][-1]
    if not 0 <= sequence_idx < len(sequences):
        raise IndexError(
            f"sequence_idx={sequence_idx} is outside the saved batch of "
            f"{len(sequences)} sequences."
        )

    # Run only the selected sequence: this avoids an unnecessary full-batch
    # forward pass while retaining the batch dimension expected by the model.
    selected_sequence = sequences[sequence_idx : sequence_idx + 1]
    with torch.no_grad():
        _, attention = model.get_attention_weights(selected_sequence, None)

    if attention.ndim != 4:
        raise ValueError(
            "Expected attention weights with shape (batch, head, query, key), "
            f"but received {tuple(attention.shape)}."
        )
    num_heads_to_plot = min(num_heads_to_plot, attention.shape[1], 8)
    if num_heads_to_plot < 1:
        raise ValueError("The checkpoint has no attention heads to plot.")

    # Sized for a double-column NeurIPS figure.  Axis-number labels are kept
    # smaller than the descriptive labels so the eight dense panels stay clear.
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.1, 3.75),
        facecolor="white",
        gridspec_kw={"hspace": 0.30, "wspace": 0.48},
    )
    axes = axes.ravel()

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad(color="white")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    image = None
    for head in range(num_heads_to_plot):
        attention_map = attention[0, head].detach().cpu().numpy()
        future_mask = np.triu(np.ones_like(attention_map, dtype=bool), k=1)
        masked_attention_map = np.ma.array(attention_map, mask=future_mask)

        image = axes[head].imshow(
            masked_attention_map,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axes[head].set_title(f"Head {head + 1}", fontsize=9.5, pad=3)
        axes[head].set_xlabel("Key position", fontsize=9, labelpad=2)
        axes[head].set_ylabel("Query position", fontsize=9, labelpad=2)
        axes[head].set_xticks([0, 50])
        axes[head].set_yticks([0, 25, 50, 75])
        axes[head].tick_params(
            axis="both", which="major", labelsize=7.25, length=2.5, width=0.7,
            pad=1.5,
        )
        for spine in axes[head].spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[num_heads_to_plot:]:
        axis.set_visible(False)

    # Reserve the right margin before placing the colourbar.  Applying this
    # adjustment afterwards would shift the last-column panels underneath it.
    fig.subplots_adjust(left=0.075, right=0.87, bottom=0.10, top=0.94)

    # A single colourbar, rather than one per panel, preserves panel area and
    # makes the shared normalization apparent at a glance.
    colorbar = fig.colorbar(
        image,
        ax=axes[:num_heads_to_plot].tolist(),
        fraction=0.024,
        pad=0.018,
        aspect=30,
        ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    colorbar.set_label("Attention weight", fontsize=9, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.25, length=2.5, width=0.7, pad=1.5)
    colorbar.outline.set_linewidth(0.8)

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    savefigdir = "./figs_camera_ready"
    utils.ensure_dir(savefigdir)

    # 1. Figure 1
    print("Generating Figure 1...")
    fig1 = create_figure1()
    for name in ["Figure1_combined", "Figure1"]:
        fig1.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig1.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig1.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {savefigdir}")
    plt.close(fig1)

    # 2. Figure 2
    print("Generating Figure 2...")
    fig2 = create_figure2()
    for name in ["Figure2_combined", "Figure2"]:
        fig2.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig2.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig2.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 2 to {savefigdir}")
    plt.close(fig2)

    # 3. Figure 5
    print("Generating Figure 5...")
    fig5 = create_figure5()
    for extension in ["pdf", "eps"]:
        fig5.savefig(
            f"{savefigdir}/Figure5.{extension}",
            format=extension,
            bbox_inches="tight",
            facecolor="white",
        )
    fig5.savefig(
        f"{savefigdir}/Figure5.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved Figure 5 to {savefigdir}")
    plt.close(fig5)


if __name__ == "__main__":
    main()
