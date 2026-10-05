import os
import sys
import glob
import math
import pickle
from collections import defaultdict

# Locate environment tools
fmri_root = os.path.dirname(os.path.dirname(sys.executable))
fmri_bin = os.path.join(fmri_root, "bin")
fmri_lib = os.path.join(fmri_root, "lib")
os.environ["PATH"] = fmri_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = fmri_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Fast Unpickler that skips allocating full model tensor arrays in RAM,
# providing ~50x faster unpickling speed on network scratch storage while
# preserving logs, arguments, and exact tensor shapes for parameter counts.
class FastParamUnpickler(pickle.Unpickler):
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


def load_raw_dataset(cache_pattern, target_epochs, min_logs=10, max_params_filter=None):
    """Regenerate capacity data from raw training checkpoint pickles."""
    folders = glob.glob(cache_pattern)
    print(f"Scanning {len(folders)} folders in {cache_pattern[:60]}...")

    records = []
    target_set = set(target_epochs)

    for i, folder in enumerate(folders):
        pkl_files = glob.glob(os.path.join(folder, "*"))
        if not pkl_files:
            continue
        fname = pkl_files[0]
        try:
            with open(fname, "rb") as fp:
                unpickler = FastParamUnpickler(fp)
                data = unpickler.load()
        except Exception:
            continue

        if "logs" not in data or len(data["logs"]) < min_logs:
            continue

        # Get parameter count: use accumulated size from state_dict if present, else fallback to args
        num_parameters = unpickler.param_count
        if num_parameters == 0 and "args" in data:
            args = data["args"]
            num_parameters = args.get("num_heads", 1) * args.get("num_hidden_features", 1) * args.get("num_layers", 1) * 12

        if max_params_filter and num_parameters > max_params_filter:
            continue

        # Extract logs for the target epochs
        for l in data["logs"]:
            epoch = l.get("epoch")
            if epoch not in target_set:
                continue

            test_metrics = l.get("test_metrics", {})
            if "accuracy" not in test_metrics or "sequence_rank" not in test_metrics:
                continue

            accuracy = np.asarray(test_metrics["accuracy"])
            sequence_ranks = np.asarray(test_metrics["sequence_rank"]) * 10

            records.append({
                "num_parameters": num_parameters,
                "epoch": epoch,
                "iters": epoch * 1000,
                "sequence_ranks": sequence_ranks,
                "accuracy": accuracy,
            })

    print(f"Loaded {len(records)} checkpoint evaluation records.")
    return records


def compute_max_ranks_per_checkpoint(records, threshold=0.9):
    """Compute maximum memorized sequence rank at threshold accuracy."""
    data_by_iter = defaultdict(lambda: {"num_parameters": [], "max_rank": []})

    # Group by (num_parameters, iters)
    grouped = defaultdict(list)
    for r in records:
        key = (r["num_parameters"], r["iters"])
        grouped[key].append(r)

    for (num_params, iters), r_list in grouped.items():
        max_rank = 0
        for r in r_list:
            acc = r["accuracy"]
            ranks = r["sequence_ranks"]
            met = ranks[acc > threshold]
            if len(met) > 0:
                max_rank = max(max_rank, met.max())

        if max_rank > 0:
            data_by_iter[iters]["num_parameters"].append(num_params)
            data_by_iter[iters]["max_rank"].append(max_rank)

    # Convert to sorted arrays
    result = {}
    for iters, d in data_by_iter.items():
        p = np.array(d["num_parameters"])
        r = np.array(d["max_rank"])
        idx = np.argsort(p)
        result[iters] = {
            "num_parameters": p[idx],
            "max_rank": r[idx],
        }
    return result


def fit_power_law(x, y):
    """Fit power law y = C * x^gamma via linear regression in log-log space."""
    mask = (x > 0) & (y > 0)
    x_val = x[mask]
    y_val = y[mask]

    if len(x_val) < 2:
        return np.nan, np.nan, None, None

    log_x = np.log10(x_val)
    log_y = np.log10(y_val)

    # Linear fit: log_y = gamma * log_x + log_c
    gamma, log_c = np.polyfit(log_x, log_y, 1)

    x_fit = np.logspace(log_x.min(), log_x.max(), 100)
    y_fit = (10 ** log_c) * (x_fit ** gamma)

    return gamma, 10 ** log_c, x_fit, y_fit


def plot_capacity(data_dict, title, target_iters, subpanel_label=None, save_prefix=None):
    """Plot Maximum rank memorized vs Number of parameters with power law fits."""
    utils.set_plot_style()

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["C0", "C1", "C2", "C3"]

    for i, iters in enumerate(target_iters):
        if iters not in data_dict:
            print(f"Warning: checkpoint {iters} not found in data.")
            continue

        p = data_dict[iters]["num_parameters"]
        r = data_dict[iters]["max_rank"]
        c = colors[i % len(colors)]

        # Power law fit
        gamma, C, x_fit, y_fit = fit_power_law(p, r)

        # Plot empirical points
        ax.plot(p, r, "o", color=c, ms=6, alpha=0.9)

        # Plot dashed power law fit
        if x_fit is not None:
            ax.plot(
                x_fit,
                y_fit,
                "--",
                color=c,
                lw=1.8,
                label=rf"$\gamma = {gamma:.2f}$",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 1e7)
    ax.set_ylim(1e1, 1.2e5)

    ax.set_xlabel("Number of parameters", fontsize=15)
    ax.set_ylabel("Maximum rank memorized", fontsize=15)
    ax.set_title(title, fontsize=18, pad=10)

    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="upper left", frameon=True, fontsize=12)

    if subpanel_label:
        fig.text(0.02, 0.95, rf"\textbf{{{subpanel_label}}}", fontsize=22, fontweight="bold", va="top")

    if save_prefix:
        utils.ensure_dir(os.path.dirname(save_prefix))
        fig.savefig(f"{save_prefix}.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(f"{save_prefix}.png", format="png", dpi=300, bbox_inches="tight")
        print(f"Saved: {save_prefix}.pdf and .png")

    return fig, ax


def main():
    # -------------------------------------------------------------
    # 1. 12-Layer Models (GPT-2)
    # Checkpoints: 60k, 120k, 190k iterations (epochs 60, 120, 190)
    # -------------------------------------------------------------
    gpt2_cache = "/scratch/qanguyen/gautam/cache/memo_apr19_zipf_gpt2_vary_num_hidden_features_num_heads_noresample/*"
    gpt2_epochs = [60, 120, 190]
    gpt2_iters = [60000, 120000, 190000]

    records_12 = load_raw_dataset(gpt2_cache, target_epochs=gpt2_epochs, min_logs=60)
    data_12 = compute_max_ranks_per_checkpoint(records_12, threshold=0.9)
    plot_capacity(
        data_12,
        title="12-layer models",
        target_iters=gpt2_iters,
        subpanel_label="b",
        save_prefix="./figs/Figure1/max_rank_vs_params_12layer",
    )

    # -------------------------------------------------------------
    # 2. 1-Layer Models
    # Checkpoints: 20k, 50k, 79k iterations (epochs 20, 50, 79)
    # -------------------------------------------------------------
    onelayer_cache = "/scratch/qanguyen/gautam/cache/memo_jan4_zipf_onelayerattention_vary_num_hidden_features_num_heads_noresample/*"
    onelayer_epochs = [20, 50, 79]
    onelayer_iters = [20000, 50000, 79000]

    records_1 = load_raw_dataset(onelayer_cache, target_epochs=onelayer_epochs, min_logs=20)
    data_1 = compute_max_ranks_per_checkpoint(records_1, threshold=0.9)
    plot_capacity(
        data_1,
        title="1-layer models",
        target_iters=onelayer_iters,
        subpanel_label=None,
        save_prefix="./figs/Figure1/max_rank_vs_params_1layer",
    )


if __name__ == "__main__":
    main()
