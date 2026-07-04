import argparse
import glob
import io
import math
import os
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class OneLayerAttention(nn.Module):
    def __init__(self, len_context, num_heads, num_hidden_features, vocab_size,
                 num_mlp_layers, is_initialize_attention_weights_to_zero=None,
                 num_hidden_features_mlp=None):
        super().__init__()
        self.len_context = len_context
        self.num_heads = num_heads
        self.num_hidden_features = num_hidden_features * num_heads
        self.num_hidden_features_mlp = num_hidden_features_mlp if num_hidden_features_mlp is not None else self.num_hidden_features
        self.wte = nn.Embedding(vocab_size, self.num_hidden_features)
        self.wpe = nn.Embedding(len_context, self.num_hidden_features)
        self.pre_attn_ln = LayerNorm(self.num_hidden_features, bias=True)
        self.ln_f = LayerNorm(self.num_hidden_features, bias=True)
        self.c_attn = nn.Linear(self.num_hidden_features, self.num_hidden_features * 3, bias=True)
        linear_modules = [nn.Linear(self.num_hidden_features, self.num_hidden_features_mlp)]
        for _ in range(num_mlp_layers - 1):
            linear_modules.append(nn.Linear(self.num_hidden_features_mlp, self.num_hidden_features_mlp))
        self.linear_modules = nn.ModuleList(linear_modules)
        self.lm_head = nn.Linear(self.num_hidden_features_mlp, vocab_size, bias=True)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2))
            if is_initialize_attention_weights_to_zero == "zero_all_attn_weights":
                if "c_attn" in pn:
                    torch.nn.init.zeros_(p)
                    p.requires_grad = False
            elif is_initialize_attention_weights_to_zero == "zero_all_attn_allow_learning":
                if "c_attn" in pn:
                    torch.nn.init.zeros_(p)
                    p.requires_grad = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets):
        B, T = idx.size()
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.wte(idx) + self.wpe(pos)
        _, _, C = x.size()
        q, k, v = self.c_attn(x).split(self.num_hidden_features, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        x = x.transpose(1, 2).contiguous().reshape(B, T, C)
        x = self.ln_f(x)
        x = F.relu(self.linear_modules[0](x))
        for module in self.linear_modules[1:]:
            x = F.relu(module(x)) + x
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str,
                    default="figs_one_shot/memo_may25_zipf_onelayerattention_large_noresample_transformer_K_500000_L_100_hidden_256_nheads_15_nlayers_12_1780095637.9333496",
                    help="Path to a single run folder inside the cache group dir")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Override learning rate (default: use saved args.lr)")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Override batch size")
parser.add_argument("--iter_num", type=int, default=None,
                    help="Load a specific checkpoint iter (default: latest)")
parser.add_argument("--no_cuda", action="store_true", default=False)
parser.add_argument("--num_extra_epochs", type=int, default=1,
                    help="Additional training epochs")
cli = parser.parse_args()

device = torch.device("cuda" if (not cli.no_cuda and torch.cuda.is_available()) else "cpu")
print("device:", device)

# ---------------------------------------------------------------------------
# Locate files
# ---------------------------------------------------------------------------
folder = cli.folder.rstrip("/")
all_pkls = glob.glob(f"{folder}/*.pkl")

# record file: not a model_iter, not all_sequences
record_file = [p for p in all_pkls
               if "model_iter" not in p and "all_sequences" not in p]
assert len(record_file) == 1, f"Expected 1 record file, got {record_file}"
record_file = record_file[0]

sequences_file = [p for p in all_pkls if "all_sequences" in p]
assert len(sequences_file) == 1, f"Expected 1 sequences file, got {sequences_file}"
sequences_file = sequences_file[0]

model_checkpoints = [p for p in all_pkls if "model_iter" in p]
# sort by iter number
def _iter_num(path):
    m = re.search(r"model_iter_(\d+)", path)
    return int(m.group(1)) if m else -1

model_checkpoints.sort(key=_iter_num)
if cli.iter_num is not None:
    matched = [p for p in model_checkpoints if _iter_num(p) == cli.iter_num]
    assert matched, f"No checkpoint for iter {cli.iter_num}. Available: {[_iter_num(p) for p in model_checkpoints]}"
    selected_ckpt = matched[0]
else:
    selected_ckpt = model_checkpoints[-1]
start_iter = _iter_num(selected_ckpt)
print(f"Record file : {record_file}")
print(f"Sequences   : {sequences_file}")
print(f"Checkpoint  : {selected_ckpt}  (iter {start_iter})")

# ---------------------------------------------------------------------------
# Load record (args + logs so far)
# ---------------------------------------------------------------------------
with open(record_file, "rb") as f:
    record = CPU_Unpickler(f).load()

args_dict = record["args"]

# Build a namespace from the saved dict
class Args:
    pass
args = Args()
args.__dict__.update(args_dict)

if cli.lr is not None:
    args.lr = cli.lr
if cli.batch_size is not None:
    args.batch_size = cli.batch_size

print("Loaded args:", vars(args))

# ---------------------------------------------------------------------------
# Load sequences
# ---------------------------------------------------------------------------
with open(sequences_file, "rb") as f:
    all_sequences_across_switches = CPU_Unpickler(f).load()

# The first entry holds the original training sequences
sequences_tensor = all_sequences_across_switches["sequences"][0]
if not isinstance(sequences_tensor, torch.Tensor):
    sequences_tensor = torch.tensor(sequences_tensor)
print(f"Sequences shape: {sequences_tensor.shape}")  # (K, len_context)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Sequence(torch.utils.data.Dataset):
    def __init__(self, sequences, K, len_context, len_data,
                 sequence_sampling_distribution="zipf"):
        self.sequences = sequences
        self.K = K
        self.len_context = len_context
        self.len_data = len_data
        self.sequence_sampling_distribution = sequence_sampling_distribution
        if sequence_sampling_distribution == "zipf":
            self.p = 1.0 / np.arange(1, K + 1)
            self.p /= self.p.sum()
        self.skip = False

    def __len__(self):
        return self.len_data

    def __getitem__(self, task):
        if self.sequence_sampling_distribution == "uniform":
            i = task % self.K
        else:  # zipf
            i = np.random.choice(self.K, p=self.p)
        return self.sequences[i].long(), i


num_iters_per_epoch = getattr(args, "num_iters_per_epoch", 1000)
batch_size = args.batch_size
sequence_sampling_distribution = getattr(args, "sequence_sampling_distribution", "zipf")
K = args.K
len_context = args.len_context

train_dataset = Sequence(
    sequences=sequences_tensor,
    K=K,
    len_context=len_context,
    len_data=batch_size * num_iters_per_epoch,
    sequence_sampling_distribution=sequence_sampling_distribution,
)
 

use_cuda = not cli.no_cuda and torch.cuda.is_available()
loader_kwargs = {"batch_size": batch_size, "num_workers": 4,
                 "pin_memory": use_cuda}
train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, **loader_kwargs
) 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
num_hidden_features_mlp = getattr(args, "num_hidden_features_mlp",
                                  args.num_heads * args.num_hidden_features)
model = OneLayerAttention(
    len_context=len_context,
    num_heads=args.num_heads,
    num_hidden_features=args.num_hidden_features,
    vocab_size=getattr(args, "vocab_size", 2),
    num_mlp_layers=getattr(args, "num_mlp_layers", 3),
    num_hidden_features_mlp=num_hidden_features_mlp,
    is_initialize_attention_weights_to_zero=getattr(
        args, "is_initialize_attention_weights_to_zero", None),
).to(device)

with open(selected_ckpt, "rb") as f:
    state_dict = CPU_Unpickler(f).load()
model.load_state_dict(state_dict)
print(f"Loaded model weights from iter {start_iter}")

num_parameters = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_parameters:,}")

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
optimizer_name = getattr(args, "optimizer", "Adam")
if optimizer_name == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 1e-5))
else:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 1e-5))

criterion = nn.NLLLoss(reduction="none")

# ---------------------------------------------------------------------------
# Reconstruct num_appearances from existing logs
# ---------------------------------------------------------------------------
num_apppearances = np.zeros(K)
for log in record["logs"]:
    if "appearances" in log:
        for batch_appearances in log["appearances"]:
            counts = np.bincount(np.asarray(batch_appearances).ravel(),
                                 minlength=K)
            num_apppearances += counts

print(f"Reconstructed num_appearances sum: {num_apppearances.sum():.0f}")

# ---------------------------------------------------------------------------
# Training loop (resumed)
# ---------------------------------------------------------------------------

for rel_iter in range(args.num_extra_epochs):
    epoch = start_iter + rel_iter + 1

    model.train()
    for seq, task in train_loader:
        optimizer.zero_grad()
        seq, task = seq.to(device), task.to(device)
        output = model(seq, task)
        B, N, D = output.shape
        logsoftmax = F.log_softmax(output[:, :-1, :], dim=-1).reshape(B * (N - 1), D)
        logsoftmaxloss = criterion(logsoftmax, seq[:, 1:].reshape(B * (N - 1)))
        loss = logsoftmaxloss.reshape(B, N - 1).mean()
        loss.backward()
        # optimizer.step() # inference mode, no need to update the model
    print(f"iter {epoch}")

print("Done.")
