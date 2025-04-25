# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python (fmri)
#     language: python
#     name: fmri
# ---

# +
import argparse
import os
import copy
import time
from enum import Enum
import importlib

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import Subset
import attention
# import webdataset as wds

import datetime
import utils
import numpy as np
import math
import einops
import random
import pandas as pd

import wandb 
import sys 
import glob
from collections import defaultdict

# +
parser = argparse.ArgumentParser(description='GMM L2L Training with Sequence Model')
parser.add_argument('--data', metavar='DIR', nargs='?', default='./data',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--cache', default='./cache',
                    help='path to cached files (e.g. for previous random weights)')
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_group_name",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--num_iter_per_epoch', default=100, type=int,  
                    help='number of iters per epoch')
parser.add_argument('--num_iters', default=5e5, type=int,  
                    help='number of iters')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')                         
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='SGD', type=str, 
                    choices = ['SGD', 'Adam'],
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlp',
                    help='model architecture (default: mlp)')
parser.add_argument('--gpt_bias', default="True", type=str,
                    help='whether to include bias in GPT')
parser.add_argument('--num_hidden_features', default=1, type=int,
                    help='num_hidden_features')
parser.add_argument('--num_layers', default=1, type=int,
                    help='num_layers in transformer')
parser.add_argument('--num_heads', default=1, type=int,
                    help='num_heads in transformer')
parser.add_argument('--len_context', default=1, type=int,
                    help='number of in-context images in sequence')
parser.add_argument('--SLURM_ARRAY_TASK_ID', default=1, type=int,
                    help='SLURM_ARRAY_TASK_ID')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  
 
parser.add_argument('--K', default=1, type=int, 
                    help='number of tasks')
parser.add_argument('--sequence_sampling_distribution', type=str,
                    default="uniform", 
                    choices = ["uniform", "zipf"],
                    help='sequence_sampling_distribution')
parser.add_argument('--is_resample_tasks', default="False", type=str,
                    help='whether to resample tasks')
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store') 
parser.add_argument('--resume', type=str, default=None,
                    help='resume from checkpoint')
    

# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    arch = "pytorch_transformer"
    # arch = "transformer"
    SLURM_ARRAY_TASK_ID = 0 
    optimizer = "SGD"
    lr = 1e-3
    num_iters = int(5e5 )
    gpt_bias = "True"
    len_context = 100
    K = 1000
    sequence_sampling_distribution = "uniform"
    jupyter_args = f"--data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --num_iters ${num_iters} --arch gpt --gpt_bias ${gpt_bias} --num_hidden_features 8 --num_layers 4 --len_context ${len_context} --K ${K} --sequence_sampling_distribution ${sequence_sampling_distribution} --no-wandb_log --wandb_project l2l --wandb_group_name t"
    # replace $ with '' in jupyter_args 
    jupyter_args = jupyter_args.replace("$","")
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    # %load_ext autoreload 
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    # %autoreload 2 

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

def set_zipf_exp_params(args):
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
    args.num_iters = 80000
    
def set_zipf_exp_params_resample(args): 
    set_zipf_exp_params(args)
    args.is_resample_tasks = "True"
    args.num_iters = 20000

def set_zipf_exp_params_forget(args):
    set_zipf_exp_params(args)
    args.is_resample_tasks = "Forget"
    args.num_iters = 40000

if args.wandb_group_name == "memo_feb26_uniformdist_modelsize":
    num_heads = [1] + list(range(2, 17, 2)) # len: 9
    num_layers = [1] + list(range(2, 17, 2)) # len: 9 
    args.num_heads = num_heads[args.SLURM_ARRAY_TASK_ID % len(num_heads)] 
    args.num_layers = num_layers[args.SLURM_ARRAY_TASK_ID // len(num_heads)]
    print("SLURM_ARRAY_TASK_ID",args.SLURM_ARRAY_TASK_ID, "num_heads", args.num_heads, "num_layers", args.num_layers)
    args.sequence_sampling_distribution = "uniform"
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_8_num_layers_12":
    args.num_heads = 8
    args.num_layers = 12
    set_zipf_exp_params(args)
    # args.resume = './cache/memo_apr6_zipf_num_heads_8_num_layers_12/memo_apr6_zipf_num_heads_8_num_layers_12_transformer_K_100000_L_100_hidden_8_nheads_8_nlayers_12_1744050810.647857.pkl'
elif args.wandb_group_name == "memo_apr10_zipf_num_heads_8_num_layers_12_resample":
    args.num_heads = 8
    args.num_layers = 12
    set_zipf_exp_params_resample(args)
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_8_num_layers_12_forget":
    args.num_heads = 8
    args.num_layers = 12
    set_zipf_exp_params_forget(args)
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_16_num_layers_24":
    args.num_heads = 16
    args.num_layers = 24
    set_zipf_exp_params(args)
    # args.resume = './cache/memo_apr6_zipf_num_heads_16_num_layers_24/memo_apr6_zipf_num_heads_16_num_layers_24_transformer_K_100000_L_100_hidden_8_nheads_16_nlayers_24_1744134825.3819082.pkl'
elif args.wandb_group_name == "memo_apr10_zipf_num_heads_16_num_layers_24_resample":
    args.num_heads = 16
    args.num_layers = 24
    set_zipf_exp_params_resample(args)
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_16_num_layers_24_forget":
    args.num_heads = 16
    args.num_layers = 24
    set_zipf_exp_params_forget(args)
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_24_num_layers_36":
    args.num_heads = 24
    args.num_layers = 36
    set_zipf_exp_params(args)
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4":
    args.num_heads = 24
    args.num_layers = 36
    args.lr = 1e-4
    set_zipf_exp_params(args)
    # args.resume = './cache/memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4/memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4_transformer_K_100000_L_100_hidden_8_nheads_24_nlayers_36_1744050810.8180943.pkl'
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-5":
    args.num_heads = 24
    args.num_layers = 36
    args.lr = 1e-5
    set_zipf_exp_params(args)
elif args.wandb_group_name == "memo_apr6_zipf_num_heads_24_num_layers_36_lr_1e-4_forget":
    args.num_heads = 24
    args.num_layers = 36
    args.lr = 1e-4
    set_zipf_exp_params_forget(args)
elif args.wandb_group_name == "memo_apr10_zipf_num_heads_24_num_layers_36_resample_lr_1e-4":
    args.num_heads = 24
    args.num_layers = 36
    set_zipf_exp_params_resample(args)
    args.lr = 1e-4
    
# assert args.K % args.L == 0, "K must be divisible by L"
if args.seed is None:
    args.seed = np.random.randint(0, 10000000)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local Rank for distributed training
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)
print("args:\n",vars(args))
# setup weights and biases (optional)
if local_rank==0 and args.wandb_log: # only use main process for wandb logging
    print(f"wandb {args.wandb_project} run")
    wandb_model_name = f"{args.fileprefix}"
    wandb_config = vars(args)
    os.environ["WANDB_MODE"] = "offline" # turn off wandb logging for now
    os.environ["WANDB_API_KEY"] = "a421cbcaff87506c1eadd3b9e4d6424996432e38"
    print("wandb_id:",wandb_model_name)
    wandb.login(relogin=True, key = os.environ["WANDB_API_KEY"])
    wandb.init(
        project=args.wandb_project,
        name=wandb_model_name,
        config=wandb_config,
        resume="allow",
        group=args.wandb_group_name
    )
    wandb.config.local_file_dir = wandb.run.dir 
else:
    record = {
        "args": vars(args),
        "logs": []
    }
if args.resume is not None:
    with open(args.resume, 'rb') as f:
        record = pickle.load(f)
    args = record['args']
    

# +

class Sequence(torch.utils.data.Dataset):
    def __init__(self, K,   
                 len_context = 1,
                len_data = 60000, skip=False,
                sequence_sampling_distribution = "uniform"):

        # if K < 40000:
        self.len_context = len_context
        self.K = K 
        self.len_data = len_data
        self.sequence_sampling_distribution = sequence_sampling_distribution
        if skip == False:
            self.generate_sequences()
        if args.sequence_sampling_distribution == "zipf":
            self.p = 1.0 / np.arange(1, self.K + 1)
            self.p /= np.sum(self.p)
        else:
            assert self.sequence_sampling_distribution == "uniform", f"sequence_sampling_distribution must be uniform or zipf, got {self.sequence_sampling_distribution}"
        self.skip = skip
    def generate_sequences(self):
        self.sequences = torch.randint(0, 2, (self.K, self.len_context)) 
        
    def __len__(self):
        return self.len_data

    def __getitem__(self, task: int):
        if (self.sequence_sampling_distribution == "uniform") or (self.skip == True):
            i = task % self.K
        elif self.sequence_sampling_distribution == "zipf":
            i = np.random.choice(self.K, p= self.p)
        samples = self.sequences[i]
        # samples = torch.randint(0, 2, (self.len_context,))
        return samples.type(torch.long), i 



# +
# importlib.reload(gpt)
import gpt
criterion = nn.NLLLoss(reduction="none")
# define the model, optimizer, and scheduler, and criterion
if args.arch == "gpt":
    import gpt 
    config = gpt.GPTConfig(
        block_size = args.len_context,
        n_embd=args.num_heads * args.num_hidden_features,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        bias = args.gpt_bias == "True"
    )
    model = gpt.GPT(config, criterion).to(device)

if args.optimizer == 'SGD': 
    optimizer = torch.optim.SGD(model.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
else:
    raise ValueError("optimizer not recognized")
iters_per_epoch = 1000
# scheduler = StepLR(optimizer, step_size=50, gamma=0.7)
# scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
#                        total_steps=args.epochs * iters_per_epoch, 
#                        pct_start=0.5,
#                        steps_per_epoch=iters_per_epoch, epochs=args.epochs)
# -

# define the dataset
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.batch_size}
use_cuda = not args.no_cuda and torch.cuda.is_available()
if use_cuda:
    cuda_kwargs = {'num_workers': args.workers,
                    "shuffle": True,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
train_dataset = Sequence(K=args.K, 
                         len_data = 1000 * args.num_iter_per_epoch,
                         len_context=args.len_context,
                         sequence_sampling_distribution = args.sequence_sampling_distribution
                         )
iwl_dataset = Sequence(K=args.K, len_context=args.len_context,  skip=True, 
                       len_data = args.K // 10,
                       sequence_sampling_distribution = "uniform")
iwl_dataset.sequences = train_dataset.sequences[::10] # take every 10th sequence from train_dataset
print ("sequences.sequence_sampling_distribution", iwl_dataset.sequence_sampling_distribution)
train_sampler = None
val_sampler = None 
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            sampler=train_sampler, 
                                            **train_kwargs)  
# At test time we don't shuffle and sample uniformly
iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})


# +
# # test dataset construction 
# ihistogram = []
# for _, (seq, i) in enumerate(train_loader):
#     ihistogram.extend(i.tolist()) 
#     if _ > 100: break 
# # print ("train_dataset", train_dataset.p, train_dataset.K)
# print ("ihistogram",ihistogram)
# import matplotlib.pyplot as plt
# # plot histogram, should be zipf distribution
# # plt.hist(ihistogram, bins = 20)
# plt.bar(np.arange(args.K), np.bincount(ihistogram, minlength=args.K))
# plt.plot(np.arange(args.K), train_dataset.p * len(ihistogram), "r")
# # plt.xlim(0, 30)
# plt.semilogy()
# plt.title("Histogram of tasks in training set, should be zipf distribution")
# plt.legend(["expected: p(zipf) * n_samps", "empirical"])
# plt.xlabel("seq rank")
# plt.ylabel("freq")
# plt.show()

# +

def validate_gradient_descent(epoch, val_loader, model, args, criterion, device):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
   
    test_losses = [utils.AverageMeter('Loss', ':.4e') for _ in range(args.len_context)]
    test_top1 = [utils.AverageMeter('Acc@1', ':6.2f') for _ in range(args.len_context)]
    model.eval() # switch to eval mode
    
    with torch.no_grad():
        for i, (seq, task) in enumerate(val_loader):
            seq, task = seq.to(device), task.to(device) 
            
            
            output = model(seq, task) # shape: B, N, D
            # print ("seq", seq.shape, "task", task.shape, "output", output.shape )
            preds = output 
            B, N, D = output.shape
             
            
            # print ("test preds", preds.shape, "task", task.shape)
            
            # [test_losses[_].update(loss[_].item(), target.size(0)) for _ in range(N)]
            
            for i_seq in range(args.len_context-1): 
                preds_query = preds[:, i_seq,:].reshape(-1, D)
                seqs_query = seq[:,(i_seq+1)].reshape(-1)
                losstest = criterion(preds_query, seqs_query)
                acc1 = utils.accuracy(preds_query, seqs_query, topk=[1])
                test_losses[i_seq].update(losstest.item(), seq.size(0))
                test_top1[i_seq].update(acc1[0].item(), seq.size(0))
                
            if i > 10: break
            # print ("acc1", acc1, "loss", loss)
            # test_top1[seq_len].update(acc1[0], target.size(0))
            # acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
            # test_top1[seq_len].update(acc1, target.size(0))

    return test_losses, test_top1


def validate_gradient_descent_zipf(epoch, val_loader, model, args, criterion, device):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
   
    sequence_rank = []
    test_losses = [] 
    test_top1 = []
    test_metrics = defaultdict(list)
    model.eval() # switch to eval mode
    sequence_ranks = torch.zeros((args.K,args.len_context-1), dtype=torch.long).to(device)
    lengths = torch.zeros((args.K,args.len_context-1), dtype=torch.long).to(device)
    logsoftmaxlosses = torch.zeros((args.K,args.len_context-1), dtype=torch.float).to(device) 
    accuracys = torch.zeros((args.K,args.len_context-1), dtype=torch.float).to(device)
    with torch.no_grad():
        for i, (seq, task) in enumerate(val_loader):
            seq, task = seq.to(device), task.to(device) 
            
            # print("seq", seq.shape, "task", task)
            output = model(seq, task) # shape: B, N, D
            # print ("seq", seq.shape, "task", task.shape, "output", output.shape )
            preds = output 
            B, N, D = output.shape
             
            for i_seq in range(args.len_context-1): 
                preds_query = preds[:, i_seq,:].reshape(-1, D)
                seqs_query = seq[:,(i_seq+1)].reshape(-1)
                # losstest = criterion(preds_query, seqs_query)
                # acc1 = utils.accuracy(preds_query, seqs_query, topk=[1]) 
                # print ("preds_max", preds_query.argmax(dim=-1).shape, "seqs_query", seqs_query.shape)
                is_correct = (preds_query.argmax(dim=-1) == seqs_query)
                logsoftmax = F.log_softmax(preds_query, dim=1)
                logsoftmaxloss = F.nll_loss(logsoftmax, seqs_query, reduction="none") 
                
                # if args.K < 10000: # for small K, we can save the whole tensor 
                # test_metrics["sequence_rank"].append(task.detach().cpu().numpy())
                sequence_ranks[task, i_seq] = task 
                lengths[task, i_seq] = i_seq + 1 
                logsoftmaxlosses[task, i_seq] = logsoftmaxloss 
                accuracys[task, i_seq] = is_correct.float() 

    test_metrics["sequence_rank"] = sequence_ranks[:,0].detach().cpu().numpy() # average over positions
    test_metrics["length"] = 50
    test_metrics["logsoftmaxloss"] = logsoftmaxlosses.mean(dim=1).detach().cpu().numpy()
    test_metrics["accuracy"] = accuracys.mean(dim=1).detach().cpu().numpy()
     
    return test_metrics
 


# +
import json
import pickle
# import matplotlib.pyplot as plt
if os.path.exists(f"./cache/{args.wandb_group_name}") == False:
    os.makedirs(f"./cache/{args.wandb_group_name}", exist_ok=True)
exp_name = f"./cache/{args.wandb_group_name}/{args.wandb_group_name}_{args.fileprefix}_K_{args.K}_L_{args.len_context}_hidden_{args.num_hidden_features}_nheads_{args.num_heads}_nlayers_{args.num_layers}_{time.time()}.pkl"
print("Saving to", exp_name)
num_iters_per_epoch = 50
num_apppearances = np.zeros(args.K)
test_every = 1 # np.log10(args.K).astype(int) * 2
all_sequences_across_switches = {
    "sequences": [],
    "switch_start_iter": [],
}
# save all sequences in pickle file
with open(f"{exp_name[:-4]}_all_sequences.pkl", "wb") as f:
    pickle.dump(train_dataset.sequences, f)
for iter in range(args.num_iters // num_iters_per_epoch):
    # Switch the sequences half way through the training
    if iter == args.num_iters // num_iters_per_epoch / 2 and args.is_resample_tasks == "True": # resample the tasks
        all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences))
        all_sequences_across_switches["switch_start_iter"].append([iter] * len(train_dataset.sequences))
        train_dataset.generate_sequences()
        if args.sequence_sampling_distribution == "zipf":
            iwl_dataset.sequences = train_dataset.sequences[::10] # take every 10th sequence from train_dataset
            iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})
        num_apppearances = np.zeros(args.K)
        
    # Switch the sequences several times throughout the training
    # Save the sequences and the start iter of the switch, so that
    # we can plot forgetting curves across different switches
    elif args.is_resample_tasks == "Forget" and iter % 100 == 0:
        train_dataset.generate_sequences()
        all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences[::50]))
        all_sequences_across_switches["switch_start_iter"].append([iter] * len(train_dataset.sequences[::50]))
        num_apppearances = np.zeros(args.K)
        if args.sequence_sampling_distribution == "zipf":
            iwl_dataset.sequences = torch.cat(all_sequences_across_switches["sequences"], dim=0)
            iwl_dataset.len_data = len(iwl_dataset.sequences)
            print("iter", iter, "len(iwl_dataset.sequences)", iwl_dataset.sequences.shape)
            iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})
    logs = {
        "num_apppearances": copy.deepcopy(num_apppearances),
    }
    if args.sequence_sampling_distribution == "zipf"  :
        if (iter % test_every == 0):
            test_metrics = validate_gradient_descent_zipf(iter, iwl_test_loader, model, args, criterion, device)
        else:
            test_metrics = {}
    else:
        test_losses, test_top1 = validate_gradient_descent(iter, train_loader, model, args, criterion, device)
        
    
    losses = utils.AverageMeter('Loss', ':.4e')
    ridge_losses = utils.AverageMeter('Ridge Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    
    
    model.train() # switch to train mode
    # train iteration
    appearances = []
    loss_per_appearance = []
    for i, (seq, task) in enumerate(train_loader):
        optimizer.zero_grad()
        seq, task = seq.to(device), task.to(device)
        batch_num_appearances = torch.bincount(task, minlength=args.K).detach().cpu().numpy()
        num_apppearances += batch_num_appearances # update the number of appearances of each task
        
        # print ("seq", seq.shape, "task", task.shape, batch_num_appearances[:20])
        output = model(seq, task)
        B, N, D = output.shape
        preds = output # shape: (B, L, 2)
        # loss function: cross-entropy loss
        # output at position i should be the input at position i+1
        
        # Write a function to compute the binary cross-entropy for each position in the sequence
        # But don't compute the mean, keep the vector dimension
        # loss = criterion(preds[:,:-1,:].reshape(B * (N-1), D), seq[:,1:].reshape(B * (N-1)))
        logsoftmax = F.log_softmax(preds[:,:-1,:], dim=-1).reshape(B * (N-1), D)
        logsoftmaxloss = criterion(logsoftmax, seq[:,1:].reshape(B * (N-1)))
        logsoftmaxloss = logsoftmaxloss.reshape(B, N-1).mean(dim=-1) # shape: (B,)
        appearances.append(task.detach().cpu().numpy())
        loss_per_appearance.append(logsoftmaxloss.detach().cpu().numpy())
        loss = logsoftmaxloss.mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), seq.size(0)) 
        
    # scheduler.step()
    # print ("loss" , loss, "preds", preds.shape, "target", seq.shape)
    
    logs.update({
            "train_loss": losses.avg,
            "epoch": iter,
            "lr": optimizer.param_groups[0]['lr'],  
            "loss_per_appearance": (loss_per_appearance),
            "appearances": (appearances),
        })
    print("iter", iter, "loss", losses.avg, logs)
    if iter == args.num_iters - 1: 
        if args.sequence_sampling_distribution == "zipf":
            test_metrics = validate_gradient_descent_zipf(iter, iwl_test_loader, model, args, criterion, device)
        else:
            test_losses, test_top1 = validate_gradient_descent(iter, train_loader, model, args, criterion, device)            
    if args.sequence_sampling_distribution == "zipf":
        logs["test_metrics"] = test_metrics
    elif args.sequence_sampling_distribution == "uniform":
        for i in range(args.len_context):
            logs[f"test_loss_{i}"] = test_losses[i].avg
            logs[f"test_top1_{i}"] = test_top1[i].avg
    
    if args.wandb_log:
        wandb.log(logs)
    else:
        # wandb.log(logs)
        record["logs"].append(logs)
    
 
    # save phi_xt_list_epoch 

    if iter % 10 == 0 and args.wandb_log != True:
        record["model"] = model.state_dict()
        with open(exp_name, "wb") as f:
            pickle.dump(record, f)
        # use json
        # with open(exp_name, "w") as f:
            # json.dump(record, f)
  
if args.wandb_log != True:
    with open(exp_name, "wb") as f:
        pickle.dump(record, f)
sys.exit(0)
