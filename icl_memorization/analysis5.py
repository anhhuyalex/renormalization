import os
import sys
import shutil
import glob
import argparse
import math
import pickle
import re
from multiprocessing import Pool
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
        top=0.80,
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

    axes_a = []
    for i in range(4):
        ax = fig.add_subplot(left_gs[i])
        ax.imshow(seqs[i : i + 1], aspect="equal", cmap="rocket", extent=[0, 20, 0, 1])
        ax.set_title(f"Sequence {i}", fontsize=11.5, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes_a.append(ax)

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

    fig.canvas.draw()
    pos_a = axes_a[0].get_position()
    pos_b = ax_b.get_position()

    t_b = ax_b.title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    t_a = axes_a[0].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())

    # Consistent panel label options matching Figure 4:
    dx_a = -0.025
    dx_b = -0.070
    y_label = max(t_a.y1, t_b.y1) + 0.040

    fig.text(pos_a.x0 + dx_a, y_label, r"\textbf{a}", fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_b.x0 + dx_b, y_label, r"\textbf{b}", fontsize=22, fontweight="bold", va="bottom", ha="left")

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
    cmap = plt.get_cmap("rocket")
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
        palette="rocket",
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


# Figure 2 top-row and bottom-row data provenance.
# Panels (a) and (b) use the no-resampling experiments;
# panel (c) uses the resampling experiments used for data-switch evaluation.
FIGURE2_TOP_AB_EXPERIMENTS = (
    "memo_apr19_zipf_gpt2_vary_num_hidden_features_num_heads_noresample",
)
FIGURE2_TOP_C_EXPERIMENTS = (
    "memo_nov10_zipf_gpt2_vary_num_hidden_features_num_heads_doresample",
)
FIGURE2_TOP_AB_RECORD_GLOBS = [
    f"/scratch/qanguyen/gautam/cache/{experiment}/*"
    for experiment in (
        [FIGURE2_TOP_AB_EXPERIMENTS]
        if isinstance(FIGURE2_TOP_AB_EXPERIMENTS, str)
        else FIGURE2_TOP_AB_EXPERIMENTS
    )
]
FIGURE2_TOP_C_RECORD_GLOBS = [
    f"/scratch/qanguyen/gautam/cache/{experiment}/*"
    for experiment in (
        [FIGURE2_TOP_C_EXPERIMENTS]
        if isinstance(FIGURE2_TOP_C_EXPERIMENTS, str)
        else FIGURE2_TOP_C_EXPERIMENTS
    )
]
# Retain singular aliases for backward compatibility
FIGURE2_TOP_EXPERIMENT = (
    FIGURE2_TOP_AB_EXPERIMENTS[0]
    if isinstance(FIGURE2_TOP_AB_EXPERIMENTS, (tuple, list))
    else FIGURE2_TOP_AB_EXPERIMENTS
)
FIGURE2_TOP_RECORD_GLOB = FIGURE2_TOP_AB_RECORD_GLOBS[0]
FIGURE2_TOP_C_EXPERIMENT = (
    FIGURE2_TOP_C_EXPERIMENTS[0]
    if isinstance(FIGURE2_TOP_C_EXPERIMENTS, (tuple, list))
    else FIGURE2_TOP_C_EXPERIMENTS
)
FIGURE2_TOP_C_RECORD_GLOB = FIGURE2_TOP_C_RECORD_GLOBS[0]
FIGURE2_TOP_A_CACHE = "./cache/df_train_loss_12layer.csv"
FIGURE2_TOP_B_CACHE = "./cache/df_capacity_12layer.csv"
FIGURE2_TOP_C_CACHE = "./cache/df_dataswitch_12layer.csv"
FIGURE2_BOTTOM_AB_EXPERIMENTS = (
    "memo_jan24_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample",
    "memo_jan25_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample",
    "memo_jan26_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample",
)


FIGURE2_BOTTOM_C_EXPERIMENT = (
    "memo_jan4_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample"
)
FIGURE2_BOTTOM_MAX_PARAMETERS = 7e7
FIGURE2_BOTTOM_AB_RECORD_GLOBS = [
    f"/scratch/qanguyen/gautam/cache/{experiment}/*"
    for experiment in FIGURE2_BOTTOM_AB_EXPERIMENTS
]
FIGURE2_BOTTOM_C_RECORD_GLOB = (
    "/scratch/qanguyen/gautam/cache/"
    f"{FIGURE2_BOTTOM_C_EXPERIMENT}/*"
)
FIGURE2_BOTTOM_A_CACHE = "./cache/df_train_loss_1layer_jan4_noresample.csv"
FIGURE2_BOTTOM_B_CACHE = "./cache/df_capacity_1layer_jan4_noresample.csv"
# This evaluation cache is rebuilt from the Jan 20 one-layer resampling logs.
FIGURE2_BOTTOM_C_CACHE = "./cache/df_dataswitch_1layer_jan20_doresample.csv"


class FastFigure2TopUnpickler(pickle.Unpickler):
    """Load Figure 2 logs while retaining only tensor shape information."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: None
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def dummy_tensor(storage, storage_offset, size, stride,
                             requires_grad, backward_hooks, metadata=None):
                class DummyTensor:
                    def numel(self):
                        return math.prod(size)
                return DummyTensor()
            return dummy_tensor
        return super().find_class(module, name)


def _extract_figure2_top_run(folder):
    """Extract the three top-row Figure 2 data tables from one raw run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith("_all_sequences.pkl")
    ]
    if len(record_paths) != 1:
        return None
    try:
        with open(record_paths[0], "rb") as file:
            data = FastFigure2TopUnpickler(file).load()
    except Exception:
        return None

    logs = data.get("logs", [])
    if len(logs) < 60:
        return None
    args = data.get("args", {})
    num_hidden_features = args.get("num_hidden_features", 1)
    num_heads = args.get("num_heads", 1)
    width = num_hidden_features * num_heads
    model_parameters = sum(
        parameter.numel() for parameter in data.get("model", {}).values()
        if hasattr(parameter, "numel")
    )
    if model_parameters == 0:
        num_layers = args.get("num_layers", 12)
        context_length = args.get("len_context", 100)
        vocab_size = args.get("vocab_size", 2)
        # GPT has token and position embeddings, ``num_layers`` transformer
        # blocks, a final layer norm, and a vocabulary-sized output head.
        model_parameters = (
            num_layers * (12 * width ** 2 + 13 * width)
            + (context_length + 2 * vocab_size + 2) * width
            + vocab_size
        )

    final_loss = logs[-1].get("train_loss")
    if final_loss is None:
        return None
    failed = (
        (model_parameters > 3e6 and final_loss > 0.21)
        or (model_parameters > 1e6 and final_loss > 0.4)
    )
    capacity_rows = []
    data_switch_rows = []
    for log in logs:
        epoch = log.get("epoch")
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_rank = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_rank is None:
            continue
        accuracy = np.asarray(accuracy)
        sequence_rank = np.asarray(sequence_rank) * 10

        if epoch in [60, 120, 190]:
            learned_ranks = sequence_rank[accuracy > 0.90]
            if len(learned_ranks):
                capacity_rows.append({
                    "num_parameters": model_parameters,
                    "epoch": epoch,
                    "iters": epoch * 1000,
                    "max_rank": int(learned_ranks.max()),
                    "failed": failed,
                })

        if epoch is not None and (0 <= epoch <= 10 or 40 <= epoch <= 50):
            first_3000 = sequence_rank <= 3000
            if np.any(first_3000):
                data_switch_rows.append({
                    "num_parameters": model_parameters,
                    "iteration": epoch * 1000,
                    "accuracy": float(accuracy[first_3000].mean()),
                    "failed": failed,
                })

    return {
        "loss": {
            "folder": os.path.basename(folder),
            "num_parameters": model_parameters,
            "num_heads": num_heads,
            "num_hidden_features": num_hidden_features * num_heads,
            "train_loss": final_loss,
            "epoch": logs[-1].get("epoch", len(logs) - 1),
            "failed": failed,
        },
        "capacity": capacity_rows,
        "data_switch": data_switch_rows,
    }


def _extract_figure2_top_c_run(folder):
    """Extract panel-c data-switch accuracy from one GPT-2 resampling run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith("_all_sequences.pkl")
    ]
    if len(record_paths) != 1:
        return []
    try:
        with open(record_paths[0], "rb") as file:
            data = FastFigure2TopUnpickler(file).load()
    except Exception:
        return []

    logs = data.get("logs", [])
    if not logs:
        return []
    args = data.get("args", {})
    num_hidden_features = args.get("num_hidden_features", 1)
    num_heads = args.get("num_heads", 1)
    width = num_hidden_features * num_heads
    model_parameters = sum(
        parameter.numel() for parameter in data.get("model", {}).values()
        if hasattr(parameter, "numel")
    )
    if model_parameters == 0:
        num_layers = args.get("num_layers", 12)
        context_length = args.get("len_context", 100)
        vocab_size = args.get("vocab_size", 2)
        model_parameters = (
            num_layers * (12 * width ** 2 + 13 * width)
            + (context_length + 2 * vocab_size + 2) * width
            + vocab_size
        )

    final_loss = logs[-1].get("train_loss")
    failed = False
    if final_loss is not None:
        failed = (
            (model_parameters > 3e6 and final_loss > 0.21)
            or (model_parameters > 1e6 and final_loss > 0.4)
        )

    data_switch_rows = []
    for log in logs:
        epoch = log.get("epoch")
        if epoch is None or not (0 <= epoch <= 10 or 40 <= epoch <= 50):
            continue
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_rank = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_rank is None:
            continue
        accuracy = np.asarray(accuracy)
        sequence_rank = np.asarray(sequence_rank) * 10
        first_3000 = sequence_rank <= 3000
        if np.any(first_3000):
            data_switch_rows.append({
                "num_parameters": model_parameters,
                "iteration": epoch * 1000,
                "accuracy": float(accuracy[first_3000].mean()),
                "failed": failed,
            })
    return data_switch_rows


def regenerate_figure2_top_ab_cache():
    """Rebuild the top-row panels (a, b) Figure 2 caches from raw GPT-2 records."""
    all_folders = sorted({
        folder
        for record_glob in FIGURE2_TOP_AB_RECORD_GLOBS
        for folder in glob.glob(record_glob)
    })
    if not all_folders:
        raise FileNotFoundError(
            f"No Figure 2 top-row (a, b) records match any of: {FIGURE2_TOP_AB_RECORD_GLOBS}."
        )
    print(
        "Regenerating Figure 2 top-row (a, b) caches from "
        f"{', '.join(FIGURE2_TOP_AB_EXPERIMENTS)} ({len(all_folders)} runs)"
    )
    with Pool(processes=min(20, len(all_folders))) as pool:
        results = [
            result for result in pool.imap_unordered(_extract_figure2_top_run, all_folders)
            if result is not None
        ]
    if not results:
        raise RuntimeError("No usable Figure 2 top-row (a, b) records were found.")

    loss_frame = pd.DataFrame([result["loss"] for result in results])
    capacity_frame = pd.DataFrame(
        row for result in results for row in result["capacity"]
    )
    if loss_frame.empty or capacity_frame.empty:
        raise RuntimeError("The extracted Figure 2 top-row (a, b) data is incomplete.")
    os.makedirs("./cache", exist_ok=True)
    loss_frame.to_csv(FIGURE2_TOP_A_CACHE, index=False)
    capacity_frame.to_csv(FIGURE2_TOP_B_CACHE, index=False)
    print(f"Wrote {len(loss_frame)} rows to {FIGURE2_TOP_A_CACHE}")
    print(f"Wrote {len(capacity_frame)} rows to {FIGURE2_TOP_B_CACHE}")


def regenerate_figure2_top_c_cache():
    """Rebuild the top-row panel (c) Figure 2 cache from raw GPT-2 resampling records."""
    all_folders = sorted({
        folder
        for record_glob in FIGURE2_TOP_C_RECORD_GLOBS
        for folder in glob.glob(record_glob)
    })
    if not all_folders:
        raise FileNotFoundError(
            f"No Figure 2 top-row panel (c) records match any of: {FIGURE2_TOP_C_RECORD_GLOBS}."
        )
    print(
        "Regenerating Figure 2 top-row panel (c) cache from "
        f"{', '.join(FIGURE2_TOP_C_EXPERIMENTS)} ({len(all_folders)} runs)"
    )
    data_switch_rows = []
    with Pool(processes=min(20, len(all_folders))) as pool:
        for run_rows in pool.imap_unordered(_extract_figure2_top_c_run, all_folders):
            data_switch_rows.extend(run_rows)
    data_switch_frame = pd.DataFrame(data_switch_rows)
    if data_switch_frame.empty:
        raise RuntimeError("The extracted Figure 2 top-row panel (c) data is empty.")
    os.makedirs("./cache", exist_ok=True)
    data_switch_frame.to_csv(FIGURE2_TOP_C_CACHE, index=False)
    print(f"Wrote {len(data_switch_frame)} rows to {FIGURE2_TOP_C_CACHE}")


def regenerate_figure2_top_cache():
    """Rebuild all top-row Figure 2 caches from raw GPT-2 records."""
    regenerate_figure2_top_ab_cache()
    regenerate_figure2_top_c_cache()


def filter_figure2_bottom_row_models(dataframe):
    """Exclude bottom-row models larger than the camera-ready cutoff."""
    kept_indices = []
    for index, row in dataframe.iterrows():
        if row["num_parameters"] > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        kept_indices.append(index)
    return dataframe.loc[kept_indices].copy()


def _extract_figure2_bottom_ab_run(folder):
    """Extract one raw one-layer run without materializing its model weights."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith("_all_sequences.pkl")
    ]
    if len(record_paths) != 1:
        return [], [], None
    try:
        with open(record_paths[0], "rb") as file:
            record = FastParamUnpickler(file)
            data = record.load()
    except Exception as error:
        return [], [], f"{record_paths[0]}: {error}"

    model_parameters = sum(
        value for value in data.get("model", {}).values()
        if isinstance(value, (int, np.integer))
    )
    if model_parameters > FIGURE2_BOTTOM_MAX_PARAMETERS:
        return [], [], None

    loss_rows = []
    capacity_rows = []
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
    return loss_rows, capacity_rows, None


def _one_layer_parameter_count_from_folder(folder):
    """Read the one-layer width from its run name without opening its pickle."""
    match = re.search(r"_hidden_(\d+)_nheads_(\d+)_", os.path.basename(folder))
    if match is None:
        return None
    hidden_features, num_heads = map(int, match.groups())
    width = hidden_features * num_heads
    # OneLayerAttention with context length 100, vocabulary size 2, and three
    # MLP layers: 6 * width^2 + 114 * width + 2 parameters.
    return 6 * width ** 2 + 114 * width + 2


def regenerate_figure2_bottom_ab_cache():
    """Rebuild Figure 2 bottom-row panels (a, b) from their raw records.

    The fast unpickler reads model tensor shapes without allocating model
    weights, then extracts epoch-wise training loss and 90%-accuracy capacity
    from ``FIGURE2_BOTTOM_AB_EXPERIMENTS``.  Models above the camera-ready
    parameter cutoff are excluded before either cache is written.
    """
    loss_rows = []
    capacity_rows = []
    all_folders = sorted({
        folder
        for record_glob in FIGURE2_BOTTOM_AB_RECORD_GLOBS
        for folder in glob.glob(record_glob)
    })
    if not all_folders:
        raise FileNotFoundError(
            "No Figure 2 bottom-row records match any of: "
            f"{FIGURE2_BOTTOM_AB_RECORD_GLOBS}."
        )
    folders = []
    for folder in all_folders:
        parameter_count = _one_layer_parameter_count_from_folder(folder)
        if parameter_count is not None and parameter_count > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        folders.append(folder)
    print(
        "Regenerating Figure 2 bottom-row caches from "
        f"{', '.join(FIGURE2_BOTTOM_AB_EXPERIMENTS)} "
        f"({len(folders)} of {len(all_folders)} runs)"
    )

    if not folders:
        raise RuntimeError(
            "All Figure 2 bottom-row records exceed the "
            f"{FIGURE2_BOTTOM_MAX_PARAMETERS:g}-parameter limit."
        )
    num_workers = min(12, len(folders))
    with Pool(processes=num_workers) as pool:
        for run_loss_rows, run_capacity_rows, error in pool.imap_unordered(
            _extract_figure2_bottom_ab_run, folders
        ):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            loss_rows.extend(run_loss_rows)
            capacity_rows.extend(run_capacity_rows)

    loss_frame = pd.DataFrame(loss_rows)
    capacity_frame = pd.DataFrame(capacity_rows)
    if loss_frame.empty or capacity_frame.empty:
        raise RuntimeError(
            "No usable Figure 2 bottom-row records were found in "
            f"{FIGURE2_BOTTOM_AB_RECORD_GLOBS}."
        )

    loss_frame.to_csv(FIGURE2_BOTTOM_A_CACHE, index=False)
    capacity_frame.to_csv(FIGURE2_BOTTOM_B_CACHE, index=False)
    print(f"Wrote {len(loss_frame)} rows to {FIGURE2_BOTTOM_A_CACHE}")
    print(f"Wrote {len(capacity_frame)} rows to {FIGURE2_BOTTOM_B_CACHE}")


def _extract_figure2_bottom_c_run(folder):
    """Extract panel-c accuracy values from one Jan 20 resampling run."""
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith("_all_sequences.pkl")
    ]
    if len(record_paths) != 1:
        return [], None
    try:
        with open(record_paths[0], "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception as error:
        return [], f"{record_paths[0]}: {error}"

    model_parameters = sum(
        value for value in data.get("model", {}).values()
        if isinstance(value, (int, np.integer))
    )
    if model_parameters > FIGURE2_BOTTOM_MAX_PARAMETERS:
        return [], None

    rows = []
    for log in data.get("logs", []):
        epoch = log.get("epoch")
        if epoch is None or not (0 <= epoch <= 10 or 40 <= epoch <= 50):
            continue
        test_metrics = log.get("test_metrics", {})
        accuracy = test_metrics.get("accuracy")
        sequence_ranks = test_metrics.get("sequence_rank")
        if accuracy is None or sequence_ranks is None:
            continue
        accuracy = np.asarray(accuracy)
        sequence_ranks = np.asarray(sequence_ranks) * 10
        first_3000 = sequence_ranks <= 3000
        if not np.any(first_3000):
            continue
        rows.append({
            "num_parameters": model_parameters,
            "iteration": epoch * 1000,
            "accuracy": float(accuracy[first_3000].mean()),
        })
    return rows, None


def regenerate_figure2_bottom_c_cache():
    """Rebuild Figure 2 bottom panel (c) from the Jan 20 resampling records."""
    all_folders = sorted(glob.glob(FIGURE2_BOTTOM_C_RECORD_GLOB))
    folders = []
    for folder in all_folders:
        parameter_count = _one_layer_parameter_count_from_folder(folder)
        if parameter_count is not None and parameter_count > FIGURE2_BOTTOM_MAX_PARAMETERS:
            continue
        folders.append(folder)
    if not folders:
        raise FileNotFoundError(
            "No Figure 2 bottom panel (c) records match "
            f"{FIGURE2_BOTTOM_C_RECORD_GLOB}."
        )
    print(
        "Regenerating Figure 2 bottom panel (c) cache from "
        f"{FIGURE2_BOTTOM_C_EXPERIMENT} ({len(folders)} of {len(all_folders)} runs)"
    )

    rows = []
    with Pool(processes=min(12, len(folders))) as pool:
        for run_rows, error in pool.imap_unordered(_extract_figure2_bottom_c_run, folders):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            rows.extend(run_rows)
    cache_frame = pd.DataFrame(rows)
    if cache_frame.empty:
        raise RuntimeError("No usable Figure 2 bottom panel (c) records were found.")
    cache_frame.to_csv(FIGURE2_BOTTOM_C_CACHE, index=False)
    print(f"Wrote {len(cache_frame)} rows to {FIGURE2_BOTTOM_C_CACHE}")


def ensure_figure2_caches(force_regenerate=False):
    """Create missing Figure 2 caches, or replace them when forced."""
    top_ab_caches = [FIGURE2_TOP_A_CACHE, FIGURE2_TOP_B_CACHE]
    bottom_ab_caches = [FIGURE2_BOTTOM_A_CACHE, FIGURE2_BOTTOM_B_CACHE]
    if force_regenerate or any(not os.path.isfile(path) for path in top_ab_caches):
        regenerate_figure2_top_ab_cache()
    if force_regenerate or not os.path.isfile(FIGURE2_TOP_C_CACHE):
        regenerate_figure2_top_c_cache()
    if force_regenerate or any(not os.path.isfile(path) for path in bottom_ab_caches):
        regenerate_figure2_bottom_ab_cache()
    if force_regenerate or not os.path.isfile(FIGURE2_BOTTOM_C_CACHE):
        regenerate_figure2_bottom_c_cache()


# =============================================================================
# FIGURE 4: Cache Extraction
# =============================================================================
FIGURE4_LINEAR_PROBE_EXPERIMENT = "memo_mar9_zipf_onelayerattention_linearprobe"
FIGURE4_FORK_PROGRESS_EXPERIMENT = (
    "memo_apr3_zipf_onelayerattention_lr_1e3_swapmlp_eval_fork_progress"
)
FIGURE4_LINEAR_PROBE_RECORD_GLOB = (
    "/scratch/qanguyen/gautam/cache/"
    f"{FIGURE4_LINEAR_PROBE_EXPERIMENT}/*"
)
FIGURE4_FORK_PROGRESS_RECORD_GLOB = (
    "/scratch/qanguyen/gautam/cache/"
    f"{FIGURE4_FORK_PROGRESS_EXPERIMENT}/*"
)
FIGURE4_LINEAR_PROBE_CACHE = "./cache/df_fig4_linearprobe.csv"
FIGURE4_FORK_PROGRESS_CACHE = "./cache/df_fig4_fork_progress.csv"


def _figure4_record_path(folder):
    """Return the checkpoint path used by the original Figure 4 extractor."""
    subdir = os.path.basename(folder)
    path = os.path.join(folder, f"{subdir}.pkl")
    return path if os.path.isfile(path) else None


def _one_layer_parameter_count_from_args(args):
    """Count parameters without allocating a OneLayerAttention model."""
    hidden_width = args["num_heads"] * args["num_hidden_features"]
    mlp_width = args.get("num_hidden_features_mlp") or hidden_width
    num_mlp_layers = args["num_mlp_layers"]
    vocab_size = args["vocab_size"]
    context_length = args["len_context"]

    # Token/position embeddings, two layer norms, QKV projection, MLP, and
    # language-model head.  This is the same count as state_dict().values()
    # for gpt.OneLayerAttention, but avoids allocating its weight tensors.
    return (
        (vocab_size + context_length) * hidden_width
        + 4 * hidden_width
        + 3 * hidden_width ** 2 + 3 * hidden_width
        + hidden_width * mlp_width + mlp_width
        + (num_mlp_layers - 1) * (mlp_width ** 2 + mlp_width)
        + mlp_width * vocab_size + vocab_size
    )


def _extract_figure4_linear_probe_run(task):
    """Extract Figure 4 linear-probe rows from one checkpoint."""
    irun, folder = task
    record_path = _figure4_record_path(folder)
    if record_path is None:
        return [], None
    try:
        with open(record_path, "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception as error:
        return [], f"{record_path}: {error}"

    args = data.get("args", {})
    model_parameters = sum(
        value for value in data.get("model", {}).values()
        if isinstance(value, (int, np.integer))
    )
    if not model_parameters:
        return [], f"{record_path}: missing model parameters"

    rows = []
    for log in data.get("logs", []):
        metrics = log.get("test_metrics", {})
        probe_value = metrics.get("probe_novel_sequences_0")
        if not isinstance(probe_value, (float, np.floating)):
            continue
        try:
            rows.append({
                "num_heads": args["num_heads"],
                "num_hidden_features": args["num_hidden_features"],
                "num_parameters": model_parameters,
                "probe_novel_sequences_0": probe_value,
                "probe_novel_sequences_1": metrics["probe_novel_sequences_1"],
                "probe_novel_sequences_2": metrics["probe_novel_sequences_2"],
                "probe_novel_sequences_3": metrics["probe_novel_sequences_3"],
                "epoch": log["epoch"],
                "irun": irun,
                "iteration": log["epoch"] * args["num_iters_per_epoch"],
            })
        except (KeyError, TypeError) as error:
            return rows, f"{record_path}: incomplete linear-probe log ({error})"
    return rows, None


def _extract_figure4_fork_progress_run(task):
    """Extract Figure 4 fork-progress rows from one checkpoint."""
    irun, folder = task
    record_path = _figure4_record_path(folder)
    if record_path is None:
        return [], None
    try:
        with open(record_path, "rb") as file:
            data = FastParamUnpickler(file).load()
        args = data["args"]
        model_parameters = _one_layer_parameter_count_from_args(args)
    except Exception as error:
        return [], f"{record_path}: {error}"

    rows = []
    for log in data.get("logs", []):
        probe_metrics = log.get("probe_metrics", {})
        test_metrics = log.get("test_metrics", {})
        if "accuracy" not in probe_metrics or "mlp_probe_metrics" not in test_metrics:
            continue
        try:
            rows.append({
                "num_heads": args["num_heads"],
                "num_hidden_features": args["num_hidden_features"],
                "probe_accuracy": float(
                    np.stack(probe_metrics["accuracy"]).flatten()[-10:].mean()
                ),
                "num_parameters": model_parameters,
                "probe_novel_sequences_0": test_metrics["mlp_probe_metrics"]["accuracy"][0],
                "probe_novel_sequences_1": test_metrics["mlp_probe_metrics"]["accuracy"][1],
                "probe_novel_sequences_2": test_metrics["mlp_probe_metrics"]["accuracy"][2],
                "probe_novel_sequences_3": test_metrics["mlp_probe_metrics"]["accuracy"][3],
                "epoch": log["epoch"],
                "irun": irun,
            })
        except (KeyError, IndexError, TypeError, ValueError) as error:
            return rows, f"{record_path}: incomplete fork-progress log ({error})"
    return rows, None


def _regenerate_figure4_cache(record_glob, experiment, extractor, cache_path):
    """Run a Figure 4 extractor and save its complete table."""
    folders = sorted(glob.glob(record_glob))
    if not folders:
        raise FileNotFoundError(f"No Figure 4 records match {record_glob}.")
    print(f"Regenerating Figure 4 cache from {experiment} ({len(folders)} runs)")

    rows = []
    tasks = list(enumerate(folders))
    with Pool(processes=min(12, len(tasks))) as pool:
        for run_rows, error in pool.imap_unordered(extractor, tasks):
            if error is not None:
                print(f"Skipping unreadable record {error}")
            rows.extend(run_rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No usable Figure 4 records were found in {record_glob}.")
    os.makedirs("./cache", exist_ok=True)
    frame.to_csv(cache_path, index=False)
    print(f"Wrote {len(frame)} rows to {cache_path}")


def regenerate_figure4_caches():
    """Rebuild both data tables required by generate_figure4.py."""
    _regenerate_figure4_cache(
        FIGURE4_LINEAR_PROBE_RECORD_GLOB,
        FIGURE4_LINEAR_PROBE_EXPERIMENT,
        _extract_figure4_linear_probe_run,
        FIGURE4_LINEAR_PROBE_CACHE,
    )
    _regenerate_figure4_cache(
        FIGURE4_FORK_PROGRESS_RECORD_GLOB,
        FIGURE4_FORK_PROGRESS_EXPERIMENT,
        _extract_figure4_fork_progress_run,
        FIGURE4_FORK_PROGRESS_CACHE,
    )


def ensure_figure4_caches(force_regenerate=False):
    """Create missing Figure 4 caches, or replace both when forced."""
    cache_paths = [FIGURE4_LINEAR_PROBE_CACHE, FIGURE4_FORK_PROGRESS_CACHE]
    if force_regenerate or any(not os.path.isfile(path) for path in cache_paths):
        regenerate_figure4_caches()


# =============================================================================
# FIGURE 4PRE: Saturation Appearances vs Sequence Rank
# =============================================================================
FIGURE4PRE_EXPERIMENT = (
    "memo_jan20_zipf_onelayerattention_vary_num_hidden_features_num_heads_doresample"
)
FIGURE4PRE_RECORD_GLOB = (
    "/scratch/qanguyen/gautam/cache/"
    f"{FIGURE4PRE_EXPERIMENT}/*"
)
FIGURE4PRE_CACHE = "./cache/df_fig4pre_saturation.csv"


def _extract_figure4pre_folder_saturation(folder):
    record_paths = [
        path for path in glob.glob(os.path.join(folder, "*.pkl"))
        if not path.endswith("_all_sequences.pkl")
    ]
    if len(record_paths) != 1:
        return []
    try:
        with open(record_paths[0], "rb") as file:
            data = FastParamUnpickler(file).load()
    except Exception:
        return []
    if "logs" not in data or len(data["logs"]) < 10:
        return []
    model_params = sum(
        value for value in data.get("model", {}).values()
        if isinstance(value, (int, np.integer))
    )
    if model_params > 1e6 and data["logs"][-1].get("train_loss", 0) > 0.3:
        return []

    rows = []
    THRESHOLD = 0.9
    for log in data["logs"]:
        if "num_apppearances" not in log or "test_metrics" not in log:
            continue
        appearances = log["num_apppearances"][::10]
        test_metrics = log["test_metrics"]
        sequence_ranks = np.asarray(test_metrics["sequence_rank"]) * 10
        accuracy = np.asarray(test_metrics["accuracy"])
        epoch = log["epoch"]
        switch_number = int(epoch // (np.round(len(data["logs"]) / 2.0)) + 1.0)

        mask = accuracy > THRESHOLD
        if not np.any(mask):
            continue
        for app, rank, acc in zip(
            np.asarray(appearances)[mask], sequence_ranks[mask], accuracy[mask]
        ):
            rows.append({
                "num_appearances": int(app),
                "sequence_rank": int(rank),
                "accuracy": float(acc),
                "num_parameters": model_params,
                "epoch": int(epoch),
                "switch_number": int(switch_number),
            })
    return rows


def regenerate_figure4pre_cache():
    """Extract and rebuild the Figure 4pre saturation cache."""
    folders = sorted(glob.glob(FIGURE4PRE_RECORD_GLOB))
    if not folders:
        raise FileNotFoundError(
            f"No Figure 4pre records match {FIGURE4PRE_RECORD_GLOB}."
        )
    print(
        f"Regenerating Figure 4pre saturation cache from {FIGURE4PRE_EXPERIMENT} "
        f"({len(folders)} runs)..."
    )
    with Pool(processes=min(12, len(folders))) as pool:
        all_results = pool.map(_extract_figure4pre_folder_saturation, folders)

    all_rows = [row for run_rows in all_results if run_rows for row in run_rows]
    if not all_rows:
        raise RuntimeError("No usable Figure 4pre saturation records found.")
    df = pd.DataFrame(all_rows)
    results = (
        df.sort_values("epoch", ascending=True)
        .drop_duplicates(subset=["num_parameters", "switch_number", "sequence_rank"])
        .sort_values(["num_parameters", "switch_number", "sequence_rank"])
        .reset_index(drop=True)
    )
    results["sequence_rank_plus_one"] = results["sequence_rank"] + 1
    os.makedirs("./cache", exist_ok=True)
    results.to_csv(FIGURE4PRE_CACHE, index=False)
    print(f"Wrote {len(results)} rows to {FIGURE4PRE_CACHE}")


def ensure_figure4pre_cache(force_regenerate=False):
    if force_regenerate or not os.path.isfile(FIGURE4PRE_CACHE):
        regenerate_figure4pre_cache()


def create_figure4pre(force_regenerate_cache=False):
    """Create publication-quality Figure 4pre: saturation appearances vs rank."""
    ensure_figure4pre_cache(force_regenerate=force_regenerate_cache)
    results = pd.read_csv(FIGURE4PRE_CACHE)

    utils.set_plot_style()
    params_to_analyze = [25399298, 56973314]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, num_params in zip(axes, params_to_analyze):
        for switch_number in [1, 2]:
            sub = results[
                (results["num_parameters"] == num_params)
                & (results["switch_number"] == switch_number)
            ].copy()
            sub["sequence_rank_plus_one"] = pd.to_numeric(sub["sequence_rank_plus_one"])
            sub = sub.sort_values("sequence_rank_plus_one")
            label = "Pre-switch" if switch_number == 1 else "Post-switch"
            sns.lineplot(
                data=sub,
                x="sequence_rank_plus_one",
                y="num_appearances",
                linewidth=2.0,
                alpha=0.85,
                label=label,
                marker="+",
                markeredgecolor="black",
                ax=ax,
            )
        ax.set_title(f"{num_params} Parameters", fontsize=20, pad=10)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 100000)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 100000)
        ax.set_xlabel("Sequence rank", fontsize=20)
        ax.set_ylabel("Saturation appearances", fontsize=20)
        ax.legend(fontsize=14, loc="upper right")

    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()

    t_a = axes[0].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    t_b = axes[1].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())

    dx = -0.025
    y_label = max(t_a.y1, t_b.y1) + 0.035

    fig.text(pos_a.x0 + dx, y_label, r"	extbf{a}", fontsize=24, fontweight="bold", va="bottom", ha="left")
    fig.text(pos_b.x0 + dx, y_label, r"	extbf{b}", fontsize=24, fontweight="bold", va="bottom", ha="left")

    return fig


def force_regenerate_all_figure_caches():
    """Rebuild every cache used by the camera-ready figures."""
    print("Figure 1: no data cache to regenerate.")
    ensure_figure2_caches(force_regenerate=True)
    ensure_figure4_caches(force_regenerate=True)
    ensure_figure4pre_cache(force_regenerate=True)
    print("Figure 5: no data cache to regenerate.")


# =============================================================================
# FIGURE 2: Full Assembly (2 rows x 3 columns)
# =============================================================================
def create_figure2(force_regenerate_cache=False):
    """Create publication-quality Figure 2 for NeurIPS submission.
    Combines:
      Column (a): Final train loss vs parameter count (12-layer top, 1-layer bottom)
      Column (b): Maximum rank memorized vs parameter count with power law fits
                  (12-layer top at 60k, 120k, 190k; 1-layer bottom at 20k, 50k, 79k)
      Column (c): Average accuracy over first 3000 ranked sequences across data switch
                  with broken axis [0, 1.0] and [4.0, 5.0] x 10^4 (12-layer top, 1-layer bottom)
    """
    ensure_figure2_caches(force_regenerate=force_regenerate_cache)
    utils.set_plot_style()
    print(
        "Figure 2 top-row panels (a, b) experiments: "
        f"{', '.join(FIGURE2_TOP_AB_EXPERIMENTS)}"
    )
    print(
        "Figure 2 top-row panel (c) experiments: "
        f"{', '.join(FIGURE2_TOP_C_EXPERIMENTS)}"
    )
    print(
        "Figure 2 bottom-row panels (a, b) experiments: "
        f"{', '.join(FIGURE2_BOTTOM_AB_EXPERIMENTS)}"
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
    df_tl_12 = pd.read_csv(FIGURE2_TOP_A_CACHE)
    ok_12 = df_tl_12[~df_tl_12["failed"]]
    plot_train_loss_panel(
        ax_a0,
        ok_12,
        title="Train Loss vs. Number of Parameters",
        d_min=min(ok_12["num_hidden_features"]),
        d_max=max(ok_12["num_hidden_features"]),
    )

    # 1-layer train loss data from FIGURE2_BOTTOM_AB_EXPERIMENTS.
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
    df_cap_12 = pd.read_csv(FIGURE2_TOP_B_CACHE)
    df_cap_12_ok = df_cap_12[~df_cap_12["failed"]] if "failed" in df_cap_12.columns else df_cap_12
    plot_power_law_panel(
        ax_b0,
        df_cap_12_ok,
        iters_list=[60000, 120000, 190000],
        title="Max Learned Rank vs. Number of Parameters",
        is_12layer=True,
    )

    # 1-layer capacity data from FIGURE2_BOTTOM_AB_EXPERIMENTS.
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

    # 12-layer data switch data from FIGURE2_TOP_C_EXPERIMENTS (memo_nov10_zipf_gpt2_vary_num_hidden_features_num_heads_doresample)
    avg_df_12 = pd.read_csv(FIGURE2_TOP_C_CACHE)

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
    sm_c0 = mpl.cm.ScalarMappable(cmap="rocket", norm=norm_c0)
    sm_c0.set_array([])
    cbar_c0 = fig.colorbar(sm_c0, ax=[ax_c0_left, ax_c0_right], pad=0.02)
    cbar_c0.set_label("Number of Parameters", fontsize=13)
    cbar_c0.formatter = mticker.LogFormatterSciNotation()
    cbar_c0.update_ticks()

    sm_c1 = mpl.cm.ScalarMappable(cmap="rocket", norm=norm_c1)
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
        figsize=(7.6, 4.0),
        facecolor="white",
        gridspec_kw={"hspace": 0.12, "wspace": 0.24},
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
        if head >= 4:
            axes[head].set_xlabel("Key position", fontsize=9, labelpad=2)
        if head % 4 == 0:
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
    fig.subplots_adjust(left=0.065, right=0.88, bottom=0.10, top=0.94)

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


def create_figure_s9(sequence_idx=9, num_heads_to_plot=16,
                     record_glob=FIGURE5_RECORD_GLOB):
    """Create the camera-ready Supplemental Figure 9 visualization (all 16 heads).

    Caption:
        Figure S9: Learned attention maps in a trained one-layer transformer for
        Sequence 10 (16 heads, 64 features per head). Each panel shows attention
        maps of one head, evaluated on the lowest-ranked sequence. Rows are query
        positions and columns are causal key positions.

    Args:
        sequence_idx: Sequence index to visualize (0-indexed; default 9 corresponds
            to Sequence 10, the lowest-ranked sequence in the evaluation set).
        num_heads_to_plot: Number of heads to plot (up to 16, arranged 4x4).
        record_glob: Glob matching the 16-head width-64 checkpoint.

    Returns:
        The Matplotlib figure object.
    """
    utils.set_plot_style()
    mpl.rcParams["text.usetex"] = False

    experiment_dirs = sorted(glob.glob(record_glob))
    if len(experiment_dirs) != 1:
        raise RuntimeError(
            "Figure S9 requires exactly one checkpoint directory matching "
            f"{record_glob!r}; found {len(experiment_dirs)}."
        )

    experiment_dir = experiment_dirs[0]
    experiment_name = os.path.basename(experiment_dir)
    record_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    data_path = os.path.join(experiment_dir, f"{experiment_name}_all_sequences.pkl")
    if not os.path.isfile(record_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Figure S9 checkpoint is incomplete. Expected both "
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

    selected_sequence = sequences[sequence_idx : sequence_idx + 1]
    with torch.no_grad():
        _, attention = model.get_attention_weights(selected_sequence, None)

    if attention.ndim != 4:
        raise ValueError(
            "Expected attention weights with shape (batch, head, query, key), "
            f"but received {tuple(attention.shape)}."
        )
    num_heads_to_plot = min(num_heads_to_plot, attention.shape[1], 16)
    if num_heads_to_plot < 1:
        raise ValueError("The checkpoint has no attention heads to plot.")

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(7.6, 7.6),
        facecolor="white",
        gridspec_kw={"hspace": 0.14, "wspace": 0.24},
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
        if head >= 12:
            axes[head].set_xlabel("Key position", fontsize=9, labelpad=2)
        if head % 4 == 0:
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

    fig.suptitle(
        "Attention patterns for Sequence 10 - 64 features, 16 heads",
        fontsize=12,
        y=0.965,
        fontweight="normal",
    )

    fig.subplots_adjust(left=0.065, right=0.88, bottom=0.06, top=0.925)

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


def create_figure_s10(head_idx=4, num_sequences=10,
                      record_glob=FIGURE5_RECORD_GLOB):
    """Create the camera-ready Supplemental Figure 10 visualization.

    Shows attention maps for a single attention head across multiple sequences:
    Head 5 (0-indexed head_idx=4) evaluated on Sequences 1 through 10.

    Suptitle:
        Attention patterns for Sequences 1 to 10 (Head 5 in a model with 64 features, 16 heads)

    Args:
        head_idx: 0-indexed head index (default 4 corresponds to Head 5).
        num_sequences: Number of held-out sequences to visualize (default 10, arranged 2x5).
        record_glob: Glob pattern for the checkpoint.

    Returns:
        The Matplotlib figure object.
    """
    utils.set_plot_style()
    mpl.rcParams["text.usetex"] = False

    experiment_dirs = sorted(glob.glob(record_glob))
    if len(experiment_dirs) != 1:
        raise RuntimeError(
            "Figure S10 requires exactly one checkpoint directory matching "
            f"{record_glob!r}; found {len(experiment_dirs)}."
        )

    experiment_dir = experiment_dirs[0]
    experiment_name = os.path.basename(experiment_dir)
    record_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    data_path = os.path.join(experiment_dir, f"{experiment_name}_all_sequences.pkl")
    if not os.path.isfile(record_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Figure S10 checkpoint is incomplete. Expected both "
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
    if len(sequences) < num_sequences:
        raise IndexError(
            f"num_sequences={num_sequences} exceeds available sequences ({len(sequences)})."
        )

    selected_sequences = sequences[:num_sequences]
    with torch.no_grad():
        _, attention = model.get_attention_weights(selected_sequences, None)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(9.2, 4.0),
        facecolor="white",
        gridspec_kw={"hspace": 0.14, "wspace": 0.24},
    )
    axes = axes.ravel()

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad(color="white")
    norm = mpl.colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    image = None
    for i in range(num_sequences):
        attn_map = attention[i, head_idx].detach().cpu().numpy()
        future_mask = np.triu(np.ones_like(attn_map, dtype=bool), k=1)
        masked_attention_map = np.ma.array(attn_map, mask=future_mask)

        image = axes[i].imshow(
            masked_attention_map,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axes[i].set_title(f"Sequence {i + 1}", fontsize=9.5, pad=3)
        if i >= 5:  # bottom row
            axes[i].set_xlabel("Key position", fontsize=9, labelpad=2)
        if i % 5 == 0:  # leftmost column
            axes[i].set_ylabel("Query position", fontsize=9, labelpad=2)
        axes[i].set_xticks([0, 50])
        axes[i].set_yticks([0, 25, 50, 75])
        axes[i].tick_params(
            axis="both", which="major", labelsize=7.25, length=2.5, width=0.7,
            pad=1.5,
        )
        for spine in axes[i].spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[num_sequences:]:
        axis.set_visible(False)

    fig.suptitle(
        "Attention patterns for Sequences 1 to 10 (Head 5 in a model with 64 features, 16 heads)",
        fontsize=11.5,
        y=0.965,
        fontweight="normal",
    )

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.90)

    colorbar = fig.colorbar(
        image,
        ax=axes[:num_sequences].tolist(),
        fraction=0.020,
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
def main(force_regenerate_cache=False):
    savefigdir = "./figs_camera_ready"
    utils.ensure_dir(savefigdir)

    if force_regenerate_cache:
        force_regenerate_all_figure_caches()
    else:
        # Figure 4 is generated by generate_figure4.py, but its two input
        # tables are maintained here alongside the other camera-ready caches.
        ensure_figure4_caches()

    # 1. Figure 1
    print("Generating Figure 1...")
    fig1 = create_figure1()
    for name in ["Figure1_combined", "Figure1"]:
        fig1.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig1.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig1.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {savefigdir}")
    plt.close(fig1)

    # 2. Figure 4pre
    print("Generating Figure 4pre...")
    fig4pre = create_figure4pre(force_regenerate_cache=force_regenerate_cache)
    for name in ["Figure4pre_combined", "Figure4pre"]:
        fig4pre.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig4pre.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig4pre.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 4pre to {savefigdir}")
    plt.close(fig4pre)

    # 3. Figure 2
    print("Generating Figure 2...")
    fig2 = create_figure2()
    for name in ["Figure2_combined", "Figure2"]:
        fig2.savefig(f"{savefigdir}/{name}.pdf", format="pdf", bbox_inches="tight")
        fig2.savefig(f"{savefigdir}/{name}.eps", format="eps", bbox_inches="tight")
        fig2.savefig(f"{savefigdir}/{name}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"Saved Figure 2 to {savefigdir}")
    plt.close(fig2)

    # 4. Figure 5
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

    # 5. Supplemental Figure 9
    print("Generating Figure Supplemental 9...")
    fig_s9 = create_figure_s9()
    for name in ["FigureS9", "FigureS9_combined", "Figure_Supplemental_9"]:
        for extension in ["pdf", "eps"]:
            fig_s9.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s9.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 9 to {savefigdir}")
    plt.close(fig_s9)

    # 6. Supplemental Figure 10
    print("Generating Figure Supplemental 10...")
    fig_s10 = create_figure_s10()
    for name in ["FigureS10", "FigureS10_combined", "Figure_Supplemental_10"]:
        for extension in ["pdf", "eps"]:
            fig_s10.savefig(
                f"{savefigdir}/{name}.{extension}",
                format=extension,
                bbox_inches="tight",
                facecolor="white",
            )
        fig_s10.savefig(
            f"{savefigdir}/{name}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"Saved Figure Supplemental 10 to {savefigdir}")
    plt.close(fig_s10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the camera-ready figures."
    )
    parser.add_argument(
        "--force_regenerate_cache",
        action="store_true",
        help=(
            "Rebuild every cache that has enough raw data, before generating "
            "the figures."
        ),
    )
    cli_args = parser.parse_args()
    main(force_regenerate_cache=cli_args.force_regenerate_cache)
