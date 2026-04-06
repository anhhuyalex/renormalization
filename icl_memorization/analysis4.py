import pickle 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import utils
from collections import defaultdict
from glob import glob

import traceback
import gpt
# fname = './cache/memo_may10_zipf_onelayerattention_lr_1e-3/memo_may10_zipf_onelayerattention_lr_1e-3_transformer_K_1000_L_100_hidden_8_nheads_20_nlayers_4_1746999806.3694336/memo_may10_zipf_onelayerattention_lr_1e-3_transformer_K_1000_L_100_hidden_8_nheads_20_nlayers_4_1746999806.3694336.pkl' 
f = '/scratch/qanguyen/gautam/cache/memo_apr3_zipf_onelayerattention_lr_1e-3_swapmlp_eval/*'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = defaultdict(list)
folders = glob(f) 
print(len(folders))
def get_sequences(sequences):
    # turn into TensorDataset
    sequences_ds = torch.utils.data.TensorDataset(torch.as_tensor(sequences))
    sequences_loader = torch.utils.data.DataLoader(sequences_ds, batch_size=100, shuffle=False)
    return sequences_loader
df_first_sequences_loader = defaultdict(list)
df_second_sequences_loader = defaultdict(list)
df_first_sequences_loader_post_swap = defaultdict(list)
for folder in folders:
    
    subdir = folder.split('/')[-1]
    fname = folder + '/' + subdir + '.pkl'
    sequences_pkl = folder + '/' + subdir + '_all_sequences.pkl'
    try:
        with open(fname, 
                'rb') as f:
            data = utils.CPU_Unpickler(f).load()
        with open(sequences_pkl, 'rb') as f:
            sequences = utils.CPU_Unpickler(f).load()
        num_heads = data["args"]["num_heads"]
        num_hidden_features = data["args"]["num_hidden_features"]   
        if "model_state_dict" not in data:
            print(f"Skipping {subdir}: missing model_state_dict")
            continue
        # count model parameters
        num_parameters = sum(v.numel() for v in data["model_state_dict"].values())
        # if num_heads < 10: continue 
        # if num_hidden_features < 20: continue 
        first_sequences_loader = get_sequences(sequences['sequences'][0][::10])
        second_sequences_loader = get_sequences(sequences['sequences'][1][::10])
    except Exception as e:
        traceback.print_exc()
        continue
    # print("args", data["args"]["num_mlp_layers"], [l["train_loss"] for l in data["logs"]])
    print("data keys", data.keys()) 
    print("sequences keys", sequences.keys())
    print("sequences",  ((sequences['sequences'])[0]).shape)
    print("model", data["args"]["lr"], num_heads, num_hidden_features, 
          "len logs", len(data["logs"]), "epochs", (data["logs"][-1]["epoch"]+1)*50)
    
    # plot train loss
    # fig, axs = plt.subplots(1, 1, figsize=(5, 6))
    # num_iters_per_epoch = 1000
    # axs.plot(np.arange(0,len(data["logs"])*num_iters_per_epoch,num_iters_per_epoch), [i["train_loss"] for i in data["logs"]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    len_context = data["args"]["len_context"]
    vocab_size = data["args"]["vocab_size"]
    num_mlp_layers = data["args"]["num_mlp_layers"]
    get_model = lambda: gpt.OneLayerAttention(len_context, 
                                  num_heads, 
                                  num_hidden_features, 
                                  vocab_size, 
                                  num_mlp_layers).to(device)
    model = get_model()
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    with torch.no_grad():
        for idx, (seqs,) in enumerate(second_sequences_loader):
            output = model(seqs.to(device), idx)
            B, N, D = output.shape
            preds_query = output[:, :len_context-1,:].reshape(-1, D)
            seqs_query = seqs[:,1:len_context].reshape(-1).to(device)
            # print("preds_query", preds_query.shape, "seqs_query", seqs_query.shape) 
            is_correct = (preds_query.argmax(dim=-1) == seqs_query).float().reshape(B, N-1)
            df_second_sequences_loader["is_correct"].extend(is_correct.mean(dim=-1).cpu().tolist())
            df_second_sequences_loader["rank"].extend((idx*np.arange(0, B)*10).tolist())
            df_second_sequences_loader["num_parameters"].extend([num_parameters] * B)
            # print("second_sequences_loader is_correct", idx, is_correct.mean(dim=-1))
    
    model.eval()
    with torch.no_grad():
        for idx, (seqs,) in enumerate(first_sequences_loader):
            output = model(seqs.to(device), idx)
            B, N, D = output.shape
            preds_query = output[:, :len_context-1,:].reshape(-1, D)
            seqs_query = seqs[:,1:len_context].reshape(-1).to(device)
            is_correct = (preds_query.argmax(dim=-1) == seqs_query).float().reshape(B, N-1)
            df_first_sequences_loader["is_correct"].extend(is_correct.mean(dim=-1).cpu().tolist())
            df_first_sequences_loader["rank"].extend((idx*np.arange(0, B)*10).tolist())
            df_first_sequences_loader["num_parameters"].extend([num_parameters] * B)
    # print("linear modules pre swap", model.linear_modules.state_dict()['1.weight'])
    # print("cattn pre swap", model.c_attn.state_dict()['weight'])
    model.linear_modules.load_state_dict(data["mlp_state_dict"]["pre_switch"]["linear_modules"])
    model.lm_head.load_state_dict(data["mlp_state_dict"]["pre_switch"]["lm_head"])
    # print("cattn post swap", model.c_attn.state_dict()['weight'])
    # print("linear modules post swap", model.linear_modules.state_dict()['1.weight'])
    model.eval()
    with torch.no_grad():
        for idx, (seqs,) in enumerate(first_sequences_loader):
            output = model(seqs.to(device), idx)
            B, N, D = output.shape
            preds_query = output[:, :len_context-1,:].reshape(-1, D)
            seqs_query = seqs[:,1:len_context].reshape(-1).to(device)
            is_correct = (preds_query.argmax(dim=-1) == seqs_query).float().reshape(B, N-1)
            df_first_sequences_loader_post_swap["is_correct"].extend(is_correct.mean(dim=-1).cpu().tolist())
            df_first_sequences_loader_post_swap["rank"].extend((idx*np.arange(0, B)*10).tolist())
            df_first_sequences_loader_post_swap["num_parameters"].extend([num_parameters] * B)

df_second_sequences_loader = pd.DataFrame(df_second_sequences_loader)
df_first_sequences_loader = pd.DataFrame(df_first_sequences_loader)
df_first_sequences_loader_post_swap = pd.DataFrame(df_first_sequences_loader_post_swap)

savefigdir = "./figs/Figure7"
out = f"{savefigdir}/Figure1a_sequences_visualization"
utils.ensure_dir(savefigdir)
# save 
df_second_sequences_loader.to_csv(f"{savefigdir}/df_second_sequences_loader.csv", index=False)
# save df_first_sequences_loader to csv
df_first_sequences_loader.to_csv(f"{savefigdir}/df_first_sequences_loader.csv", index=False)
# save df_first_sequences_loader_post_swap to csv
df_first_sequences_loader_post_swap.to_csv(f"{savefigdir}/df_first_sequences_loader_post_swap.csv", index=False)