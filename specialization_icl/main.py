#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import attention
import webdataset as wds

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


# In[ ]:


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
parser.add_argument('--epochs', default=90, type=int,  
                    help='number of total epochs to run')
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
parser.add_argument('--num_hidden_features', default=1, type=int,
                    help='num_hidden_features')
parser.add_argument('--len_context', default=1, type=int,
                    help='number of in-context images in sequence')
parser.add_argument('--SLURM_ARRAY_TASK_ID', default=1, type=int,
                    help='SLURM_ARRAY_TASK_ID')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  
parser.add_argument('--D', default=63, type=int, 
                    help='number of features in each input')
parser.add_argument('--K', default=1, type=int, 
                    help='number of tasks')
parser.add_argument('--coarse_graining', default="abstop", type=str,
                    help='coarse graining method')
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store') 
    

# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    arch = "pytorch_transformer"
    # arch = "transformer"
    jupyter_args = f"--data ./cache --fileprefix transformer1layer_lr_0.01_no_posenc --position_encoding False --SLURM_ARRAY_TASK_ID 5 --batch-size 128 --optimizer SGD --lr 0.01 --wd 1e-10 --epsilon 0.0 --burstiness 1 --init_scale_qkv 1.0 --scale_target 1.0 --is_all_tasks_seen True --is_equalize_classes True --L 2  --epochs 1000 --arch causal_transformer_embed --is_layer_norm True --num_hidden_features 512 --is_train_random_length_seqs False --len_context 100 --wandb_log --wandb_project l2l --is_temperature_fixed False --wandb_group_name gmm_sep19_exp5_equalize0s1"
    
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

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
    wandb.login(host='https://stability.wandb.io') # need to configure wandb environment beforehand
    wandb_model_name = f"{args.fileprefix}_K_{args.K}_num_tasks_{args.num_tasks}"
    wandb_config = vars(args)
    
    print("wandb_id:",wandb_model_name)
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


# In[ ]:


class Sequence(torch.utils.data.Dataset):
    def __init__(self, K, D,  
                 len_context = 1,
                len_data = 60000):

        # if K < 40000:
        self.len_context = len_context
        self.D = D
    
        # x = rng.standard_normal((K, D)) * (1.0 / np.sqrt(D)) # shape: (K, D) 
        true_betas = torch.randn((K, D)) * (1.0 / np.sqrt(D)) # shape: (K, D)
        self.true_betas = true_betas
        self.K = K 
        self.D = D
        
    def __len__(self):
        return self.len_data

    def __getitem__(self, task: int):
        task_ind = torch.randint(0, self.K, (1,)).item()
        beta_incontext = self.true_betas[task_ind] # shape: (1, D)
        x = torch.randn((self.len_context, self.D)) * (1.0 / np.sqrt(self.D)) # shape: (self.len_context, D) 
        y = torch.matmul(x, beta_incontext.T) # shape: (self.len_context, 1) 
        # concat x and y 
        samples = torch.cat([x, y], axis = 1) # shape: (self.len_context, D+1)
        xtest = torch.randn((1, self.D)) * (1.0 / np.sqrt(self.D)) # test x
        ytest = torch.matmul(xtest, beta_incontext.T) 
        test_samples = torch.cat([xtest, 0], axis = 1) # test samples, shape (1, D+1)
        samples = torch.cat([samples, test_samples], axis = 0) # shape: (self.len_context+1, D+1)
          
        return samples.type(torch.float32), ytest.type(torch.float32), beta_incontext.type(torch.float32)  


# In[ ]:


importlib.reload(attention)
# define the model, optimizer, and scheduler, and criterion
if args.arch == "causal_transformer_embed":
    nheads = 1 # np.clip(args.num_hidden_features // 8, 1, 8)
    model = attention.CausalTransformerOneMinusOneEmbed(x_dim=args.D+1,                   
                                  mlp_dim=args.num_hidden_features
                                  ).to(device)

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

criterion = nn.MSELoss().to(device)


# In[ ]:


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

train_dataset = Sequence(K=args.K, D=args.D, len_context=args.len_context) 
# iwl_dataset = Sequence(K=args.K, D=args.D, len_context=args.len_context, len_data = 1000)
# iwl_dataset.true_betas = train_dataset.true_betas
icl_test_dataset = Sequence(K=args.K, D=args.D, len_context=args.len_context, len_data = 1000)

iwl_test_dataset = Sequence(K=args.K, D=args.D, len_context=args.len_context, len_data = 1000)
iwl_test_dataset.true_betas = train_dataset.true_betas

train_sampler = None
val_sampler = None 
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            sampler=train_sampler, 
                                            **train_kwargs) 
icl_test_loader = torch.utils.data.DataLoader(icl_test_dataset,
                                            sampler=val_sampler,
                                            **test_kwargs)  
iwl_test_loader = torch.utils.data.DataLoader(iwl_test_dataset,
                                            sampler=val_sampler,
                                            **test_kwargs)


# In[ ]:


def validate_gradient_descent(epoch, val_loader, model, args, criterion, device, coarse_graining="standard"):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
   
    test_losses = utils.AverageMeter('Loss', ':.4e') 
    
    model.eval() # switch to eval mode
    
    with torch.no_grad():
        for i, (seq, target, true_beta) in enumerate(val_loader):
            seq, target = seq.to(device), target.to(device)
            if coarse_graining == "absbot":
                # true_beta: shape (B, D)
                argsort_beta_visible = torch.argsort(torch.abs(true_beta), dim=-1)[:, :args.D_visible] # sort each row of true_beta by absolute value, shape (B, D_visible)
                test_beta_visible = true_beta[argsort_beta_visible] # take bottom D_visible betas, shape (B, D_visible)
                sigma_test_xi = torch.pow(args.sigma_xi ** 2 + torch.matmul(true_beta.unsqueeze(1), true_beta.unsqueeze(2))                                         - torch.matmul(test_beta_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)), 0.5)
                x_test_visible = seq[:, -1, :-1].squeeze(1)[argsort_beta_visible] # shape (B, D_visible)
                # target = x_test_visible  @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
                target = torch.matmul(x_test_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)).squeeze(2) + torch.randn_like(target) * sigma_test_xi # shape (B, 1)
            elif coarse_graining == "abstop":
                argsort_beta_visible = torch.argsort(torch.abs(true_beta), dim=-1)[:, -args.D_visible:] # sort each row of true_beta by absolute value, shape (B, D_visible)
                test_beta_visible = true_beta[argsort_beta_visible] # take top D_visible betas, shape (B, D_visible) 
                sigma_test_xi = torch.pow(args.sigma_xi ** 2 + torch.matmul(true_beta.unsqueeze(1), true_beta.unsqueeze(2))                                         - torch.matmul(test_beta_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)), 0.5)
                x_test_visible = seq[:, -1, :-1].squeeze(1)[argsort_beta_visible] # shape (B, D_visible)
                # target = x_test_visible  @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
                target = torch.matmul(x_test_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)).squeeze(2) + torch.randn_like(target) * sigma_test_xi # shape (B, 1)
            elif coarse_graining == "standard":
                pass
            output = model(seq)  
            loss = criterion(output, target)
            test_losses.update(loss.item(), target.size(0))
            # acc1 = utils.accuracy(output, seq_target, topk=[1])
            # test_top1[seq_len].update(acc1[0], target.size(0))
            # acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
            # test_top1[seq_len].update(acc1, target.size(0))

    return test_losses 


# In[ ]:


import pickle
# import matplotlib.pyplot as plt
exp_name = f"./cache/{args.wandb_group_name}_K_{args.K}_{time.time()}.pkl"
for epoch in range(args.epochs):
    model.train() # switch to train mode
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

 
    for i, (seq, target, _) in enumerate(train_loader):
        optimizer.zero_grad()
        seq, target = seq.to(device), target.to(device)
        output = model(seq) 
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), target.size(0)) 
        # acc1 = utils.accuracy(output, (seq_target), topk=[1])
        # print ("output", output.shape, output[0], seq_target[0], loss, acc1, model.temperature)
        # top1.update(acc1[0], target.size(0))
        # acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
        # top1.update(acc1, target.size(0))
 
    # step scheduler
    # scheduler.step()

    # save metrics
    # print("output",  torch.argsort(output, dim=-1), "target", target )
    # print("Current average loss", losses.avg, top1.avg, "epoch", epoch) 
    # seen_val_losses, seen_val_top1 = validate_gradient_descent(icl_loader, seen_projs_permutations_loader, model, args, criterion, device)
    icl_indistribution_losses = validate_gradient_descent(epoch, icl_test_loader, model, args, criterion, device, coarse_graining="standard")
    icl_outdistribution_losses = validate_gradient_descent(epoch, icl_test_loader, model, args, criterion, device, coarse_graining=args.coarse_graining)
    iwl_indistribution_losses = validate_gradient_descent(epoch, iwl_test_loader, model, args, criterion, device, coarse_graining="standard")
    iwl_outdistribution_losses = validate_gradient_descent(epoch, iwl_test_loader, model, args, criterion, device, coarse_graining=args.coarse_graining)
    
    # Compute unseen val loss
    # unseen_val_losses, unseen_val_top1 = validate_gradient_descent(icl_loader, seen_projs_permutations_loader, model, args, criterion, device)
    logs = {
            "train_loss": losses.avg,
            "epoch": epoch,
            "icl_indistribution_loss": icl_indistribution_losses.avg,
            "icl_outdistribution_loss": icl_outdistribution_losses.avg,
            "iwl_indistribution_loss": iwl_indistribution_losses.avg,
            "iwl_outdistribution_loss": iwl_outdistribution_losses.avg,
        }
    
    print(logs) 
    if args.wandb_log:
        wandb.log(logs)
    else:
        record["logs"].append(logs)
    
 
    # save phi_xt_list_epoch 

    # if epoch % 10 == 0:
    #     if args.arch == "phenomenologicalcausal_transformer":
    #         with open(f"./cache/{args.wandb_group_name}_betastd_{args.beta_std}_{time.time()}.pkl", "wb") as f:
    #             pickle.dump(record, f)
    #     else:
    #         with open(exp_name, "wb") as f:
    #             pickle.dump(record, f)
  
if args.wandb_log != True:
    with open(exp_name, "wb") as f:
        pickle.dump(record, f)
