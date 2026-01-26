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
import gpt 
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()

# fname = './cache/memo_may10_zipf_onelayerattention_lr_1e-3/memo_may10_zipf_onelayerattention_lr_1e-3_transformer_K_1000_L_100_hidden_8_nheads_20_nlayers_4_1746999806.3694336/memo_may10_zipf_onelayerattention_lr_1e-3_transformer_K_1000_L_100_hidden_8_nheads_20_nlayers_4_1746999806.3694336.pkl' 
# f = './cache/memo_may26_zipf_onelayerattention_lr_1e-3_vary_num_hidden_features_num_heads/*'
df = defaultdict(list)
folder_names = [args.folder]
print("folder_names", folder_names)
for folder in folder_names:
    
    subdir = folder.split('/')[-1]
    fname = folder + '/' + subdir + '.pkl'
    try:
        with open(fname, 
                'rb') as f:
            data = utils.CPU_Unpickler(f).load()
        
    except Exception as e:
        print("error", e)
    print("args num_heads", data["args"]["num_heads"], "args num_hidden_features", data["args"]["num_hidden_features"])
    # plot train loss
    # fig, axs = plt.subplots(1, 1, figsize=(5, 6))
    # num_iters_per_epoch = 1000
    # axs.plot(np.arange(0,len(data["logs"])*num_iters_per_epoch,num_iters_per_epoch), [i["train_loss"] for i in data["logs"]])
    model = gpt.OneLayerAttention(data["args"]["len_context"], 
                                  data["args"]["num_heads"], 
                                  data["args"]["num_hidden_features"], 
                                  data["args"]["vocab_size"], 
                                  data["args"]["num_mlp_layers"]) 
    model.load_state_dict(data["model"])
    model.eval()
    sequences_fname = folder + '/' + subdir + '_all_sequences.pkl'
    with open(sequences_fname, 'rb') as f:
        sequences = utils.CPU_Unpickler(f).load()
        if len(sequences["sequences"]) < 2: continue
    print("sequences", len(sequences["sequences"]), sequences["sequences"][-1].shape)
    batch_size = 256
    last_sequences = sequences["sequences"][-1] # shape: (B, T)
    
    attn_weights_list = []
    # post switch accuracy
    for i_b in range(0, len(sequences["sequences"][-1]), batch_size):
        print("i_b", i_b, "batch_size", batch_size, torch.tensor(np.array(sequences["sequences"][-1][i_b:i_b+batch_size])).shape)
        # output = model(torch.tensor(np.array(sequences["sequences"][1][i_b:i_b+batch_size])), None)
        with torch.no_grad():
            attn, attn_weights = model.get_attention_weights(torch.tensor(np.array(sequences["sequences"][-1][i_b:i_b+batch_size])), None)
            attn_weights_list.append(attn_weights)
        # break 
    attn_weights = torch.cat(attn_weights_list, dim=0)
    # print("attn_weights", attn_weights.shape)
    bit_pair = (0,0)
    position_pair = (0,1)
    mask = np.logical_and(last_sequences[:, position_pair[0]] == bit_pair[0], last_sequences[:, position_pair[1]] == bit_pair[1]).float()
    # print("mask", mask[:attn_weights.shape[0]])
    plt.imshow(mask.reshape(100,1000))
    avg_attention_mask_bit_pair = {}
    # Loop through possibilities of bits
    for bit_pair in [(0,0), (0,1), (1,0), (1,1)]:
        avg_attention_mask = torch.zeros(data["args"]["num_heads"], last_sequences.shape[1], last_sequences.shape[1]) # shape: (H, T, T)
        position_pairs = itertools.combinations(range(last_sequences.shape[1]), 2) # combinations of 2 positions
        # Loop through all positions
        for position_pair in position_pairs:
            position_pair = sorted(position_pair, reverse=True) # make sure position_pair[0] is the larger position since attention weights are lower triangular
            # Get sequences with the bit pair at the two positions
            mask = torch.logical_and(last_sequences[:, position_pair[0]] == bit_pair[0], last_sequences[:, position_pair[1]] == bit_pair[1]).bool()
            
            # Get the attention weights of the masked sequences
            masked_attn_weights = (attn_weights)[mask[:attn_weights.shape[0]]]
            # print("masked_attn_weights", masked_attn_weights.shape)
            avg_attention_mask[:, position_pair[0], position_pair[1]] = masked_attn_weights.mean(dim=0)[:, position_pair[0], position_pair[1]]
            # print("masked_attn_weights", masked_attn_weights.mean(dim=0)[:, position_pair[0], position_pair[1]])
        for i in range(data["args"]["num_heads"]):
            plt.imshow(avg_attention_mask[i].detach().cpu().numpy())
            plt.title(f"Average attention mask for bit pair {bit_pair} and head {i}")
            plt.savefig(f"./figs/num_heads_{data['args']['num_heads']}_num_hidden_features_{data['args']['num_hidden_features']}_bit_pair_{bit_pair}_head_{i}_lr_{data['args']['lr']}.png")
            plt.show()
        avg_attention_mask_bit_pair[bit_pair] = avg_attention_mask
    # save as pickle 
    with open(folder + '/' + subdir + '_avg_attention_mask_bit_pair.pkl', 'wb') as f:
        pickle.dump(avg_attention_mask_bit_pair, f)
        
        
    
    