import os
import sys
import glob
import math
import pickle
from collections import defaultdict
import copy

fmri_root = os.path.dirname(os.path.dirname(sys.executable))
fmri_bin = os.path.join(fmri_root, "bin")
fmri_lib = os.path.join(fmri_root, "lib")
os.environ["PATH"] = fmri_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = fmri_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import seaborn as sns
import utils

utils.set_plot_style()

def build_figure2():
    # Figsize proportional to the 2x3 layout
    fig = plt.figure(figsize=(19, 10.5))

    outer_gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.05, 1.35],
        height_ratios=[1.0, 1.0],
        wspace=0.32,
        hspace=0.38,
        left=0.05,
        right=0.96,
        top=0.92,
        bottom=0.08,
    )

    # -------------------------------------------------------------------------
    # Panel (a): Train Loss vs. Parameters (Top: 12-layer, Bottom: 1-layer)
    # -------------------------------------------------------------------------
    markers = ["o", "X", "s", "P", "D", "^", "v", "<", ">", "*", "h", "8"]

    # Top: 12-layer
    ax_a0 = fig.add_subplot(outer_gs[0, 0])
    # Bottom: 1-layer
    ax_a1 = fig.add_subplot(outer_gs[1, 0])

    # -------------------------------------------------------------------------
    # Panel (b): Max Rank vs. Parameters (Top: 12-layer, Bottom: 1-layer)
    # -------------------------------------------------------------------------
    ax_b0 = fig.add_subplot(outer_gs[0, 1])
    ax_b1 = fig.add_subplot(outer_gs[1, 1])

    # 12-layer capacity data
    df_cap_12 = pd.read_csv("maxrank_vs_numparams_gpt2_apr24.csv")
    colors = ["C0", "C1", "C2"]
    ax_b0.set_title("12-layer models", fontsize=18, pad=8)
    for i, iters in enumerate([60000, 120000, 190000]):
        sub = df_cap_12[df_cap_12["iters_checkpoint"] == iters].dropna(subset=["num_parameters", "max_rank_at_saturation"])
        sub = sub[(sub["num_parameters"] > 0) & (sub["max_rank_at_saturation"] > 0)]
        if iters == 120000:
            sub = sub[sub["num_parameters"] <= 6e6]
        p = sub["num_parameters"].values
        r = sub["max_rank_at_saturation"].values
        c = colors[i]
        log_p = np.log10(p)
        log_r = np.log10(r)
        gamma, log_c = np.polyfit(log_p, log_r, 1)
        ax_b0.plot(p, r, "o", color=c, ms=6, alpha=0.9)
        x_fit = np.logspace(log_p.min(), log_p.max(), 100)
        y_fit = (10 ** log_c) * (x_fit ** gamma)
        ax_b0.plot(x_fit, y_fit, "--", color=c, lw=1.8, label=rf"$\gamma = {gamma:.2f}$")

    ax_b0.set_xscale("log")
    ax_b0.set_yscale("log")
    ax_b0.set_xlim(1e4, 1e7)
    ax_b0.set_ylim(1e1, 1.2e5)
    ax_b0.set_xlabel("Number of parameters", fontsize=14)
    ax_b0.set_ylabel("Maximum rank memorized", fontsize=14)
    ax_b0.tick_params(which="both", direction="in", top=True, right=True)
    ax_b0.legend(loc="upper left", frameon=True, fontsize=11)

    # 1-layer capacity data
    df_cap_1 = pd.read_csv("maxrank_vs_numparams_onelayer_apr30.csv")
    ax_b1.set_title("1-layer models", fontsize=18, pad=8)
    for i, iters in enumerate([20000, 50000, 79000]):
        sub = df_cap_1[df_cap_1["iters_checkpoint"] == iters].dropna(subset=["num_parameters", "max_rank_at_saturation"])
        sub = sub[(sub["num_parameters"] > 0) & (sub["max_rank_at_saturation"] > 0)]
        p = sub["num_parameters"].values
        r = sub["max_rank_at_saturation"].values
        c = colors[i]
        log_p = np.log10(p)
        log_r = np.log10(r)
        gamma, log_c = np.polyfit(log_p, log_r, 1)
        ax_b1.plot(p, r, "o", color=c, ms=6, alpha=0.9)
        x_fit = np.logspace(log_p.min(), log_p.max(), 100)
        y_fit = (10 ** log_c) * (x_fit ** gamma)
        ax_b1.plot(x_fit, y_fit, "--", color=c, lw=1.8, label=rf"$\gamma = {gamma:.2f}$")

    ax_b1.set_xscale("log")
    ax_b1.set_yscale("log")
    ax_b1.set_xlim(1e4, 1e7)
    ax_b1.set_ylim(1e1, 1.2e5)
    ax_b1.set_xlabel("Number of parameters", fontsize=14)
    ax_b1.set_ylabel("Maximum rank memorized", fontsize=14)
    ax_b1.tick_params(which="both", direction="in", top=True, right=True)
    ax_b1.legend(loc="upper left", frameon=True, fontsize=11)

    # -------------------------------------------------------------------------
    # Panel (c): Meta-Memorization Broken Axis (Top: 12-layer, Bottom: 1-layer)
    # -------------------------------------------------------------------------
    gs_c0 = outer_gs[0, 2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    ax_c0_left = fig.add_subplot(gs_c0[0])
    ax_c0_right = fig.add_subplot(gs_c0[1], sharey=ax_c0_left)

    gs_c1 = outer_gs[1, 2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    ax_c1_left = fig.add_subplot(gs_c1[0])
    ax_c1_right = fig.add_subplot(gs_c1[1], sharey=ax_c1_left)

    # Subpanel labels a, b, c
    fig.text(0.04, 0.96, r"\textbf{a}", fontsize=24, fontweight="bold", va="top")
    fig.text(0.35, 0.96, r"\textbf{b}", fontsize=24, fontweight="bold", va="top")
    fig.text(0.67, 0.96, r"\textbf{c}", fontsize=24, fontweight="bold", va="top")

    print("Layout created successfully")
    return fig

build_figure2()
