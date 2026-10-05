import os
import sys
import time
import traceback
from collections import defaultdict
from glob import glob
import numpy as np
import pandas as pd
import torch
import utils
import gpt

device = torch.device("cpu")

# 1. memo_mar9_zipf_onelayerattention_linearprobe
csv_d_e = "./cache/df_fig4_linearprobe.csv"
if not os.path.exists(csv_d_e):
    print("Extracting memo_mar9_zipf_onelayerattention_linearprobe...")
    t0 = time.time()
    f = '/scratch/qanguyen/gautam/cache/memo_mar9_zipf_onelayerattention_linearprobe/*'
    folders = glob(f)
    print(f"Found {len(folders)} folders")
    df = defaultdict(list)
    num_runs_per_param = defaultdict(int)
    for irun, folder in enumerate(folders):
        subdir = folder.split('/')[-1]
        fname = folder + '/' + subdir + '.pkl'
        try:
            with open(fname, 'rb') as fp:
                data = utils.CPU_Unpickler(fp).load()
        except Exception:
            continue
        model = data["model"]
        num_heads = data["args"]["num_heads"]
        num_hidden_features = data["args"]["num_hidden_features"]
        num_parameters = int(np.sum([torch.prod(torch.tensor(v.shape)).item() for k, v in model.items()]))
        num_runs_per_param[num_parameters] += 1
        for log_id, l in enumerate(data["logs"]):
            if "test_metrics" not in l or "probe_novel_sequences_0" not in l["test_metrics"]:
                continue
            if not isinstance(l["test_metrics"]["probe_novel_sequences_0"], float):
                continue
            df["num_heads"].append(num_heads)
            df["num_hidden_features"].append(num_hidden_features)
            df["num_parameters"].append(num_parameters)
            df["probe_novel_sequences_0"].append(l["test_metrics"]["probe_novel_sequences_0"])
            df["probe_novel_sequences_1"].append(l["test_metrics"]["probe_novel_sequences_1"])
            df["probe_novel_sequences_2"].append(l["test_metrics"]["probe_novel_sequences_2"])
            df["probe_novel_sequences_3"].append(l["test_metrics"]["probe_novel_sequences_3"])
            df["iteration"].append(l["epoch"] * data["args"]["num_iters_per_epoch"])
            df["irun"].append(irun)
    df = pd.DataFrame(df)
    df.to_csv(csv_d_e, index=False)
    print(f"Saved {csv_d_e} with shape {df.shape} in {time.time() - t0:.2f}s")
else:
    print(f"Already exists: {csv_d_e}")

# 2. memo_apr3_zipf_onelayerattention_lr_1e3_swapmlp_eval_fork_progress
csv_f = "./cache/df_fig4_fork_progress.csv"
if not os.path.exists(csv_f):
    print("Extracting memo_apr3_zipf_onelayerattention_lr_1e3_swapmlp_eval_fork_progress...")
    t0 = time.time()
    f = '/scratch/qanguyen/gautam/cache/memo_apr3_zipf_onelayerattention_lr_1e3_swapmlp_eval_fork_progress/*'
    folders = glob(f)
    print(f"Found {len(folders)} folders")
    df = defaultdict(list)
    num_runs_per_param = defaultdict(int)
    for irun, folder in enumerate(folders):
        subdir = folder.split('/')[-1]
        fname = folder + '/' + subdir + '.pkl'
        try:
            with open(fname, 'rb') as fp:
                data = utils.CPU_Unpickler(fp).load()
        except Exception:
            continue
        try:
            get_model = lambda: gpt.OneLayerAttention(
                data["args"]["len_context"],
                data["args"]["num_heads"],
                data["args"]["num_hidden_features"],
                data["args"]["vocab_size"],
                data["args"]["num_mlp_layers"],
                is_initialize_attention_weights_to_zero=data["args"]["is_initialize_attention_weights_to_zero"],
            ).to(device)
            model = get_model().state_dict()
            num_parameters = int(np.sum([torch.prod(torch.tensor(v.shape)).item() for k, v in model.items()]))
        except Exception:
            # Fallback if args missing
            num_heads = data["args"]["num_heads"]
            num_hidden_features = data["args"]["num_hidden_features"]
            # approximate or compute from weights if available
            continue
        num_heads = data["args"]["num_heads"]
        num_hidden_features = data["args"]["num_hidden_features"]
        num_runs_per_param[num_parameters] += 1
        for log_id, l in enumerate(data["logs"]):
            if "probe_metrics" not in l or "accuracy" not in l["probe_metrics"]:
                continue
            if "test_metrics" not in l or "mlp_probe_metrics" not in l["test_metrics"]:
                continue
            df["num_heads"].append(num_heads)
            df["num_hidden_features"].append(num_hidden_features)
            df["probe_accuracy"].append(np.stack(l["probe_metrics"]["accuracy"]).flatten()[-10:].mean())
            df["num_parameters"].append(num_parameters)
            df["probe_novel_sequences_0"].append(l["test_metrics"]["mlp_probe_metrics"]["accuracy"][0])
            df["probe_novel_sequences_1"].append(l["test_metrics"]["mlp_probe_metrics"]["accuracy"][1])
            df["probe_novel_sequences_2"].append(l["test_metrics"]["mlp_probe_metrics"]["accuracy"][2])
            df["probe_novel_sequences_3"].append(l["test_metrics"]["mlp_probe_metrics"]["accuracy"][3])
            df["epoch"].append(l["epoch"])
            df["irun"].append(irun)
    df = pd.DataFrame(df)
    df.to_csv(csv_f, index=False)
    print(f"Saved {csv_f} with shape {df.shape} in {time.time() - t0:.2f}s")
else:
    print(f"Already exists: {csv_f}")
