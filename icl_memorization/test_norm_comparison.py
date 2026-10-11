import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit

# 1. Load data
df_de = pd.read_csv('./cache/df_fig4_linearprobe.csv')
final_it = df_de['iteration'].max()
df_final = df_de[df_de['iteration'] == final_it]
best_idx = df_final.groupby('num_parameters')['probe_novel_sequences_0'].idxmax()
best_runs = df_final.loc[best_idx, ['num_parameters', 'irun']]
df_best = df_de.merge(best_runs, on=['num_parameters', 'irun'], how='inner').sort_values(['num_parameters', 'iteration']).reset_index(drop=True)
params_de = np.sort(df_best['num_parameters'].unique())

df_f = pd.read_csv('./cache/df_fig4_fork_progress.csv')
eps = np.finfo(float).eps
p = df_f["probe_accuracy"].clip(eps, 1 - eps)
df_f["logit_probe_accuracy"] = logit(p)
params_f = np.sort(df_f['num_parameters'].unique())

print("params_de:", params_de.min(), params_de.max())
print("params_f:", params_f.min(), params_f.max())

# Test plot with shared norm vs separate norms
for mode in ['shared', 'separate']:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    if mode == 'shared':
        norm = mpl.colors.LogNorm(vmin=min(params_de.min(), params_f.min()), vmax=max(params_de.max(), params_f.max()))
        norm_de = norm
        norm_f = norm
    else:
        norm_de = mpl.colors.LogNorm(vmin=params_de.min(), vmax=params_de.max())
        norm_f = mpl.colors.LogNorm(vmin=params_f.min(), vmax=params_f.max())
    
    cmap = plt.get_cmap("rocket")
    param_to_color_de = {p: cmap(norm_de(p)) for p in params_de}
    
    # d
    sns.lineplot(
        data=df_best, x="iteration", y="probe_novel_sequences_0",
        hue="num_parameters", hue_order=params_de, palette=param_to_color_de,
        marker="+", markeredgecolor="black", markersize=4, linewidth=2.5,
        estimator=None, sort=False, ax=axes[0]
    )
    axes[0].get_legend().remove()
    axes[0].set_title(f"d ({mode})")
    
    # e
    sns.lineplot(
        data=df_best, x="iteration", y="probe_novel_sequences_3",
        hue="num_parameters", hue_order=params_de, palette=param_to_color_de,
        marker="+", markeredgecolor="black", markersize=4, linewidth=2.5,
        estimator=None, sort=False, ax=axes[1]
    )
    axes[1].get_legend().remove()
    axes[1].set_title(f"e ({mode})")
    
    # f
    sns.scatterplot(
        data=df_f, x="probe_novel_sequences_0", y="logit_probe_accuracy",
        hue="num_parameters", palette="rocket", hue_norm=norm_f,
        legend=False, marker="+", s=120, linewidth=2.5, edgecolor="black",
        ax=axes[2]
    )
    axes[2].set_title(f"f ({mode})")
    
    sm = mpl.cm.ScalarMappable(norm=norm_f, cmap="rocket")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], pad=0.02)
    cbar.set_label("Number of Parameters")
    
    fig.savefig(f"test_norm_{mode}.png", dpi=150)
    plt.close()
print("Saved test_norm_shared.png and test_norm_separate.png")
