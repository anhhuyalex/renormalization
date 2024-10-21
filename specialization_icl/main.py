#!/usr/bin/env python
# coding: utf-8

# In[108]:


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


# In[109]:


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
parser.add_argument('--num_layers', default=1, type=int,
                    help='num_layers in transformer')
parser.add_argument('--len_context', default=1, type=int,
                    help='number of in-context images in sequence')
parser.add_argument('--SLURM_ARRAY_TASK_ID', default=1, type=int,
                    help='SLURM_ARRAY_TASK_ID')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  
parser.add_argument('--D_sum', default=1000, type=int, help='number of visible+ hidden features')
parser.add_argument('--D_visible_frac', default=1.0, type=float, help='fraction of features visible') 
parser.add_argument('--K', default=1, type=int, 
                    help='number of tasks')
parser.add_argument('--coarse_graining', default="abstop", type=str,
                    help='coarse graining method')
parser.add_argument('--sigma_xi', default=1.0, type=float, help='noise level')
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store') 
    

# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    arch = "pytorch_transformer"
    # arch = "transformer"
    jupyter_args = f"--data ./cache --fileprefix transformer1layer  --SLURM_ARRAY_TASK_ID 5 --batch-size 128 --optimizer SGD --lr 1e-2 --wd 1e-10  --epochs 200 --arch gpt --num_hidden_features 256 --num_layers 1 --len_context 16 --K 100 --D_sum 8 --D_visible_frac 3 --sigma_xi 0.5 --coarse_graining abstop --no-wandb_log --wandb_project renormalization --wandb_group_name t"
    
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

args.D_visible = int(args.D_visible_frac) # just using D=8 max(int(args.D_visible_frac * args.D_sum),1)
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
    wandb_model_name = f"{args.fileprefix}_K_{args.K}_D_{args.D_sum}_L_{args.len_context}_hidden_{args.num_hidden_features}_coarse_{args.coarse_graining}"
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


# In[103]:


class Sequence(torch.utils.data.Dataset):
    def __init__(self, K, D,  
                 len_context = 1,
                 scale=0.5,
                len_data = 60000, skip_generating_betas=False):

        # if K < 40000:
        self.len_context = len_context
        self.D = D
    
        # x = rng.standard_normal((K, D)) * (1.0 / np.sqrt(D)) # shape: (K, D) 
        self.scale = scale
        if skip_generating_betas == False:
            true_betas = torch.randn((K, D)) * scale #* (1.0 / np.sqrt(D)) # shape: (K, D)
            self.true_betas = true_betas
        self.K = K 
        self.D = D
        self.len_data = len_data
        
    def __len__(self):
        return self.len_data

    def __getitem__(self, task: int):
        task_ind = torch.randint(0, self.K, (1,)).item()
        beta_incontext = self.true_betas[task_ind].unsqueeze(1) # shape: (D, 1)
        x = torch.randn((self.len_context, self.D)) * self.scale  # shape: (self.len_context, D) * (1.0 / np.sqrt(self.D))
        noise = torch.randn((self.len_context, 1)) * args.sigma_xi
        y = torch.matmul(x, beta_incontext) + noise

        # concat x and y 
        samples = x#torch.cat([x, y], axis = 1) # shape: (self.len_context, D+1)
        # ytest = samples[-1, -1].clone() 
        # samples[-1, -1] = 0.0 # remove ytest from samples 
         
          
        return samples.type(torch.float32), y.type(torch.float32), beta_incontext.type(torch.float32)  


# In[104]:


# importlib.reload(gpt)
import gpt
criterion = nn.MSELoss().to(device)
# define the model, optimizer, and scheduler, and criterion
if args.arch == "causal_transformer_embed":
    nheads = 1 # np.clip(args.num_hidden_features // 8, 1, 8)
    model = attention.MultiLayerTransformer(x_dim=args.D_sum,                   
                                  mlp_dim=args.num_hidden_features, 
                                  num_layers = args.num_layers
                                  ).to(device)
if args.arch == "gpt":
    import gpt 
    config = gpt.GPTConfig(
        block_size = args.len_context,
        input_size = args.D_sum,
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



# In[105]:


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

train_dataset = Sequence(K=args.K, D=args.D_sum, len_context=args.len_context) 
# iwl_dataset = Sequence(K=args.K, D=args.D_sum, len_context=args.len_context, len_data = 1000)
# iwl_dataset.true_betas = train_dataset.true_betas
icl_test_dataset = Sequence(K=1000, D=args.D_sum, len_context=args.len_context, len_data = 1000)

iwl_test_dataset = Sequence(K=args.K, D=args.D_sum, len_context=args.len_context, len_data = 1000, skip_generating_betas = True)
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


# In[106]:


def validate_gradient_descent(epoch, val_loader, model, args, criterion, device, coarse_graining="standard"):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
   
    test_losses = utils.AverageMeter('Loss', ':.4e') 
    
    model.eval() # switch to eval mode
    
    with torch.no_grad():
        for i, (seq, target, _true_beta) in enumerate(val_loader):
            seq, target, _true_beta = seq.to(device), target.to(device), _true_beta.to(device)
            if coarse_graining == "absbot":
                # true_beta: shape (B, D)
                true_beta = _true_beta.squeeze(2)
                argsort_beta_visible = torch.argsort(torch.abs(true_beta), dim=-1)[:, :args.D_visible] # sort each row of true_beta by absolute value, shape (B, D_visible)
                test_beta_visible = torch.gather(true_beta, dim=1, index=argsort_beta_visible) # shape (B, D_visible)
                sigma_test_xi = torch.pow(args.sigma_xi ** 2 + torch.matmul(true_beta.unsqueeze(1), true_beta.unsqueeze(2)).squeeze(2).squeeze(1) \
                                        - torch.matmul(test_beta_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)).squeeze(2).squeeze(1), 0.5)
                x_test_visible = torch.gather(seq[:, -1, :].squeeze(1), dim=1, index=argsort_beta_visible) # shape (B, D_visible)
                new_target = torch.matmul(x_test_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)).squeeze(2) 
                new_target = new_target.squeeze(1)
                new_target += torch.randn(new_target.size(0), device=device) * sigma_test_xi # shape (B, 1) 
                target[:, -1, 0] = new_target
                
            elif coarse_graining == "abstop":
                true_beta = _true_beta.squeeze(2) # shape (B, D)
                # print ("true_beta", true_beta.shape)
                argsort_beta_visible = torch.argsort(torch.abs(true_beta), dim=-1)[:, -args.D_visible:] # sort each row of true_beta by absolute value, shape (B, D_visible)
                # test_beta_visible = true_beta[argsort_beta_visible] # take top D_visible betas, shape (B, D_visible) 
                test_beta_visible = torch.gather(true_beta, dim=1, index=argsort_beta_visible) # shape (B, D_visible)
                # print  ("-args.D_visible", -args.D_visible, "argsort_beta_visible", argsort_beta_visible.shape, "test_beta_visible", test_beta_visible.shape)
                sigma_test_xi = torch.pow(args.sigma_xi ** 2 + torch.matmul(true_beta.unsqueeze(1), true_beta.unsqueeze(2)) \
                                        - torch.matmul(test_beta_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)), 0.5).squeeze(2).squeeze(1) # shape (B)
                # x_test_visible = seq[:, -1, :-1].squeeze(1)[argsort_beta_visible] # shape (B, D_visible)
                x_test_visible = torch.gather(seq[:, -1, :].squeeze(1), dim=1, index=argsort_beta_visible) # shape (B, D_visible) 
                
                # target = x_test_visible  @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
                new_target = torch.matmul(x_test_visible.unsqueeze(1), test_beta_visible.unsqueeze(2)).squeeze(2) 
                new_target = new_target.squeeze(1)
                new_target += torch.randn(new_target.size(0), device=device) * sigma_test_xi # shape (B, 1) 
                # print ("new_target", new_target, "sigma_test_xi", sigma_test_xi )
                target[:, -1, 0] = new_target

                
            elif coarse_graining == "standard":
                pass
            seq, target = seq.to(device), target.to(device)
            # print ("seq", seq.shape, "target", target.shape)
            output = model(seq, target) 
            # print ("seq", seq.shape, "target", target.shape, "output", output.shape )
            preds = output[:, 0::2, :]
            # print ("preds", preds.shape, "target", target[:, :, :].shape)
            loss = criterion(preds, target)
            test_losses.update(loss.item(), target.size(0))
            # acc1 = utils.accuracy(output, seq_target, topk=[1])
            # test_top1[seq_len].update(acc1[0], target.size(0))
            # acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
            # test_top1[seq_len].update(acc1, target.size(0))

    return test_losses 


# In[107]:


import pickle
# import matplotlib.pyplot as plt
exp_name = f"./cache/{args.wandb_group_name}_{args.fileprefix}_K_{args.K}_D_{args.D_sum}_L_{args.len_context}_hidden_{args.num_hidden_features}_coarse_{args.coarse_graining}_{time.time()}.pkl"
for epoch in range(args.epochs):
    icl_indistribution_losses = validate_gradient_descent(epoch, icl_test_loader, model, args, criterion, device, coarse_graining="standard")
    icl_outdistribution_losses = validate_gradient_descent(epoch, icl_test_loader, model, args, criterion, device, coarse_graining=args.coarse_graining)
    iwl_indistribution_losses = validate_gradient_descent(epoch, iwl_test_loader, model, args, criterion, device, coarse_graining="standard")
    iwl_outdistribution_losses = validate_gradient_descent(epoch, iwl_test_loader, model, args, criterion, device, coarse_graining=args.coarse_graining)
    
    model.train() # switch to train mode
    losses = utils.AverageMeter('Loss', ':.4e')
    ridge_losses = utils.AverageMeter('Ridge Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

 
    for i, (seq, target, _) in enumerate(train_loader):
        optimizer.zero_grad()
        seq, target = seq.to(device), target.to(device)
        # print ("seq", seq.shape, "target", target.shape)
        output = model(seq, target) 
        # print ("seq", seq.shape, "target", target.shape, "output", output.shape )
        preds = output[:, 0::2, :]
        # print ("preds", preds.shape, "target", target[:, :, :].shape)
        loss = criterion(preds, target)
        
        # batch_first_seq, batch_first_target = seq[0, :-1, :], target[0, :-1, 0]
        # print ("batch_first_seq", batch_first_seq.shape, "batch_first_target", batch_first_target.shape)
        # ridge = utils.Ridge(alpha=1e-9,fit_intercept=True) 
        # ridge.fit(batch_first_seq, batch_first_target)
        # val_loss = criterion(ridge.predict(seq[0, [-1], :]), target[0, -1, 0])
        # print ("loss", loss, "ridge loss", val_loss, "pred", ridge.predict(seq[0, [-1], :]).shape)
        # ridge_losses.update(val_loss.item(), 1) 
        # compute ridge loss on first sequence
        
        
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), target.size(0)) 
        # acc1 = utils.accuracy(output, (seq_target), topk=[1])
        # print ("output", output.shape, output[0], seq_target[0], loss, acc1, model.temperature)
        # top1.update(acc1[0], target.size(0))
        # acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
        # top1.update(acc1, target.size(0))
    print ("loss" , loss)
    # step scheduler
    # scheduler.step()

    # save metrics
    # print("output",  torch.argsort(output, dim=-1), "target", target )
    # print("Current average loss", losses.avg, top1.avg, "epoch", epoch) 
    # seen_val_losses, seen_val_top1 = validate_gradient_descent(icl_loader, seen_projs_permutations_loader, model, args, criterion, device)
    
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

    if epoch % 10 == 0 and args.wandb_log != True:
        with open(exp_name, "wb") as f:
            pickle.dump(record, f)
  
if args.wandb_log != True:
    with open(exp_name, "wb") as f:
        pickle.dump(record, f)


# In[ ]:





# In[ ]:




