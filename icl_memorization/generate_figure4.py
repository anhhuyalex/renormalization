import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit
import utils

# Setup paths and environment
os.environ['PATH'] = '/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/mnt/cup/labs/norman/qanguyen/patdiff_seq/fmri/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Apply plot style
utils.set_plot_style()
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})

def add_spines(ax, linewidth=1.2, color="black"):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(linewidth)
        ax.spines[side].set_edgecolor(color)

def set_linear_rescaled_ticks(
    ax,
    x_values,
    y_values,
    x_scale=1e4,
    y_scale=1e7,
    x_tick_step=1.5,
    y_tick_step=1.0,
    show_yticklabels=True,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    ncols = len(x_values)
    nrows = len(y_values)

    x_min = x_values.min() / x_scale
    x_max = x_values.max() / x_scale
    y_min = y_values.min() / y_scale
    y_max = y_values.max() / y_scale

    x_ticks_lab = np.arange(
        0,
        np.floor(x_max / x_tick_step) * x_tick_step + 0.5 * x_tick_step,
        x_tick_step,
    )
    y_ticks_lab = np.arange(
        np.ceil(y_min / y_tick_step) * y_tick_step,
        y_max + 0.5 * y_tick_step,
        y_tick_step,
    )

    if len(x_ticks_lab) == 0 or not np.isclose(x_ticks_lab[0], 0):
        x_ticks_lab = np.insert(x_ticks_lab, 0, 0.0)
    if len(y_ticks_lab) == 0:
        y_ticks_lab = np.array([y_min, y_max])

    x_ticks_lab = x_ticks_lab[(x_ticks_lab >= x_min) & (x_ticks_lab <= x_max)]
    y_ticks_lab = y_ticks_lab[(y_ticks_lab >= y_min) & (y_ticks_lab <= y_max)]

    x_pos = np.interp(
        x_ticks_lab * x_scale,
        [x_values.min(), x_values.max()],
        [0.5, ncols - 0.5],
    )
    y_pos = np.interp(
        y_ticks_lab * y_scale,
        [y_values.min(), y_values.max()],
        [0.5, nrows - 0.5],
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{v:.1f}".rstrip("0").rstrip(".") for v in x_ticks_lab],
        rotation=90,
    )

    ax.set_yticks(y_pos)
    if show_yticklabels:
        ax.set_yticklabels(
            [f"{v:.1f}".rstrip("0").rstrip(".") for v in y_ticks_lab],
            rotation=0,
        )
    else:
        ax.set_yticklabels([])

print("Loading datasets...")
# Top row data
df_post = pd.read_csv('./figs/Figure6/df_first_sequences_loader_post_swap.csv')
df_first = pd.read_csv('./figs/Figure6/df_first_sequences_loader.csv')
df_second = pd.read_csv('./figs/Figure6/df_second_sequences_loader.csv')

# Bottom row data
df_de = pd.read_csv('./cache/df_fig4_linearprobe.csv')
final_it = df_de['iteration'].max()
df_de["iteration"] = pd.to_numeric(df_de["iteration"], errors="coerce")
df_de["probe_novel_sequences_0"] = pd.to_numeric(df_de["probe_novel_sequences_0"], errors="coerce")
df_de["probe_novel_sequences_3"] = pd.to_numeric(df_de["probe_novel_sequences_3"], errors="coerce")
df_de["num_parameters"] = pd.to_numeric(df_de["num_parameters"], errors="coerce")

df_final = df_de[df_de["iteration"] == final_it]
best_idx = df_final.groupby("num_parameters")["probe_novel_sequences_0"].idxmax()
best_runs = df_final.loc[best_idx, ["num_parameters", "irun"]]

df_best = (
    df_de.merge(best_runs, on=["num_parameters", "irun"], how="inner")
       .sort_values(["num_parameters", "iteration"])
       .reset_index(drop=True)
)
params_de = np.sort(df_best["num_parameters"].unique())

df_f = pd.read_csv('./cache/df_fig4_fork_progress.csv')
eps = np.finfo(float).eps
p = df_f["probe_accuracy"].clip(eps, 1 - eps)
df_f["logit_probe_accuracy"] = logit(p)

print("Building Figure 4...")
fig = plt.figure(figsize=(15.5, 10.5))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.035], wspace=0.25, hspace=0.48)

# Create axes
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
cbar_top_ax = fig.add_subplot(gs[0, 3])

ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])
cbar_bottom_ax = fig.add_subplot(gs[1, 3])

# ----------------- TOP ROW -----------------
top_datasets = [
    (ax_a, df_second, "Second Sequences Accuracy", "a", True),
    (ax_b, df_first, "First Sequences Accuracy", "b", False),
    (ax_c, df_post, "First Sequences Post-Swap Accuracy", "c", False),
]

for ax, raw_df, title, panel_lbl, is_first in top_datasets:
    df_sub = pd.DataFrame(
        raw_df[(raw_df["num_parameters"] > 2e6) & (raw_df["num_parameters"] < 6e7)]
    ).copy()
    df_sub["rank"] = pd.to_numeric(df_sub["rank"], errors="coerce")
    df_sub["num_parameters"] = pd.to_numeric(df_sub["num_parameters"], errors="coerce")
    df_sub["is_correct"] = pd.to_numeric(df_sub["is_correct"], errors="coerce")

    pivot_df = (
        df_sub.pivot_table(
            columns="rank",
            index="num_parameters",
            values="is_correct",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    sns.heatmap(
        pivot_df,
        cmap="YlGnBu",
        vmin=0.5,
        vmax=0.8,
        ax=ax,
        cbar=(panel_lbl == "c"),
        cbar_ax=cbar_top_ax if (panel_lbl == "c") else None,
        cbar_kws={"label": "Accuracy"} if (panel_lbl == "c") else None,
    )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(r"Rank ($\times 10^4$)", fontsize=13)
    ax.set_ylabel(r"Number of Parameters ($\times 10^7$)" if is_first else "", fontsize=13)

    set_linear_rescaled_ticks(
        ax,
        x_values=pivot_df.columns.to_numpy(),
        y_values=pivot_df.index.to_numpy(),
        x_scale=1e4,
        y_scale=1e7,
        x_tick_step=1.5,
        y_tick_step=1.0,
        show_yticklabels=is_first,
    )

    add_spines(ax, linewidth=1.2)

add_spines(cbar_top_ax, linewidth=1.2)
cbar_top_ax.set_ylabel("Accuracy", fontsize=13)

# ----------------- BOTTOM ROW -----------------
# Norms and colormaps
cmap = plt.cm.turbo
norm_de = mpl.colors.LogNorm(vmin=params_de.min(), vmax=params_de.max())
param_to_color_de = {p: cmap(norm_de(p)) for p in params_de}

norm_f = mpl.colors.LogNorm(vmin=df_f["num_parameters"].min(), vmax=df_f["num_parameters"].max())

# Panel d: Attention linear decodability vs iteration
sns.lineplot(
    data=df_best,
    x="iteration",
    y="probe_novel_sequences_0",
    hue="num_parameters",
    hue_order=params_de,
    palette=param_to_color_de,
    marker="+",
    markeredgecolor="black",
    markersize=4,
    linewidth=2.2,
    estimator=None,
    sort=False,
    ax=ax_d,
)
ax_d.get_legend().remove()
ax_d.set_xlabel("Iteration", fontsize=13)
ax_d.set_ylabel("Linear Decodability", fontsize=13)
ax_d.set_title("Linear Decodability vs. Iteration", fontsize=14, pad=10)
ax_d.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
ax_d.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax_d.set_ylim(-0.05, 0.9)
add_spines(ax_d, linewidth=1.2)

# Panel e: MLP linear decodability vs iteration
sns.lineplot(
    data=df_best,
    x="iteration",
    y="probe_novel_sequences_3",
    hue="num_parameters",
    hue_order=params_de,
    palette=param_to_color_de,
    marker="+",
    markeredgecolor="black",
    markersize=4,
    linewidth=2.2,
    estimator=None,
    sort=False,
    ax=ax_e,
)
ax_e.get_legend().remove()
ax_e.set_xlabel("Iteration", fontsize=13)
ax_e.set_ylabel("Linear Decodability", fontsize=13)
ax_e.set_title("Linear Decodability for MLPs vs. Iteration", fontsize=14, pad=10)
ax_e.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
ax_e.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax_e.set_ylim(-0.05, 0.9)
add_spines(ax_e, linewidth=1.2)

# Panel f: Linear Decodability vs Accuracy Logit
sns.scatterplot(
    data=df_f,
    x="probe_novel_sequences_0",
    y="logit_probe_accuracy",
    hue="num_parameters",
    palette="turbo",
    hue_norm=norm_f,
    legend=False,
    marker="+",
    s=110,
    linewidth=2.2,
    edgecolor="black",
    ax=ax_f,
)
ax_f.set_xlabel("Linear Decodability", fontsize=13)
ax_f.set_ylabel("Logit of Probe Accuracy", fontsize=13)
ax_f.set_title("Linear Decodability vs. Accuracy Logit", fontsize=14, pad=10)
ax_f.set_xlim(-0.05, 1.0)
ax_f.set_ylim(-0.1, 2.55)
add_spines(ax_f, linewidth=1.2)

# Bottom Colorbar on cbar_bottom_ax
sm_bottom = mpl.cm.ScalarMappable(norm=norm_f, cmap="turbo")
sm_bottom.set_array([])
cbar_bottom = fig.colorbar(sm_bottom, cax=cbar_bottom_ax)
cbar_bottom.set_label("Number of Parameters", fontsize=13)
cbar_bottom.formatter = mpl.ticker.LogFormatterSciNotation()
cbar_bottom.update_ticks()
add_spines(cbar_bottom_ax, linewidth=1.2)

# Precise panel label placement in figure coordinates
# Get positions of subplots after layout
fig.canvas.draw()
pos_a = ax_a.get_position()
pos_b = ax_b.get_position()
pos_c = ax_c.get_position()
pos_d = ax_d.get_position()
pos_e = ax_e.get_position()
pos_f = ax_f.get_position()

y_top_label = pos_a.y1 + 0.025
y_bot_label = pos_d.y1 + 0.025

# Move labels slightly to the right, sitting right at the top-left shoulder of each respective panel:
# dx = -0.012 in figure coordinates (~0.186 inches to the left of the subplot spine)
dx = -0.012

fig.text(pos_a.x0 + dx, y_top_label, r"\textbf{a}", fontsize=22, fontweight='bold', va='bottom', ha='left')
fig.text(pos_b.x0 + dx, y_top_label, r"\textbf{b}", fontsize=22, fontweight='bold', va='bottom', ha='left')
fig.text(pos_c.x0 + dx, y_top_label, r"\textbf{c}", fontsize=22, fontweight='bold', va='bottom', ha='left')

fig.text(pos_d.x0 + dx, y_bot_label, r"\textbf{d}", fontsize=22, fontweight='bold', va='bottom', ha='left')
fig.text(pos_e.x0 + dx, y_bot_label, r"\textbf{e}", fontsize=22, fontweight='bold', va='bottom', ha='left')
fig.text(pos_f.x0 + dx, y_bot_label, r"\textbf{f}", fontsize=22, fontweight='bold', va='bottom', ha='left')

# Ensure output directories exist
utils.ensure_dir("./figs/Figure4")
utils.ensure_dir("figs_camera_ready/Figure4")

# Save outputs efficiently
print("Saving Figure 4...")
master_png = "./figs/Figure4/Figure4.png"
master_pdf = "./figs/Figure4/Figure4.pdf"
master_eps = "./figs/Figure4/Figure4.eps"

fig.savefig(master_png, format="png", dpi=300, bbox_inches="tight")
fig.savefig(master_pdf, format="pdf", bbox_inches="tight")
fig.savefig(master_eps, format="eps", bbox_inches="tight")

# Fast copy to all targets
targets = [
    "./figs/Figure4/Figure4_combined",
    "figs_camera_ready/Figure4/Figure4",
    "figs_camera_ready/Figure4/Figure4_combined",
    "figs_camera_ready/Figure4",
]

for t in targets:
    shutil.copyfile(master_png, f"{t}.png")
    shutil.copyfile(master_pdf, f"{t}.pdf")
    shutil.copyfile(master_eps, f"{t}.eps")

# Also copy to artifacts dir for review
shutil.copyfile(master_png, "/usr/people/qanguyen/.gemini/antigravity-ide/brain/3b22a695-64a5-404f-adf4-41ef71f94d4f/Figure4_combined.png")

print("All Figure 4 files successfully generated!")
