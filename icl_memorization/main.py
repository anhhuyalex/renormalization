#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store') 
    

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
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

if args.wandb_group_name == "memo_feb26_uniformdist_modelsize":
    num_heads = [1] + list(range(2, 17, 2)) # len: 9
    num_layers = [1] + list(range(2, 17, 2)) # len: 9 
    args.num_heads = num_heads[args.SLURM_ARRAY_TASK_ID % len(num_heads)] 
    args.num_layers = num_layers[args.SLURM_ARRAY_TASK_ID // len(num_heads)]
    print("SLURM_ARRAY_TASK_ID",args.SLURM_ARRAY_TASK_ID, "num_heads", args.num_heads, "num_layers", args.num_layers)
    args.sequence_sampling_distribution = "uniform"
elif args.wandb_group_name == "memo_feb26_zipf_num_heads_8_num_layers_12":
    args.num_heads = 8
    args.num_layers = 12
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
elif args.wandb_group_name == "memo_feb26_zipf_num_heads_16_num_layers_24":
    args.num_heads = 16
    args.num_layers = 24
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
elif args.wandb_group_name == "memo_feb26_zipf_num_heads_24_num_layers_36":
    args.num_heads = 24
    args.num_layers = 36
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
elif args.wandb_group_name == "memo_feb26_zipf_num_heads_24_num_layers_36_rope_embedding":
    args.num_heads = 24
    args.num_layers = 36
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
    args.rope_embedding = True
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


# In[11]:


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
            self.sequences = torch.randint(0, 2, (self.K, self.len_context)) 
        if args.sequence_sampling_distribution == "zipf":
            self.p = 1.0 / np.arange(1, self.K + 1)
            self.p /= np.sum(self.p)
        else:
            assert self.sequence_sampling_distribution == "uniform", f"sequence_sampling_distribution must be uniform or zipf, got {self.sequence_sampling_distribution}"
        self.skip = skip
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


# In[ ]:


# importlib.reload(gpt)
import gpt
criterion = nn.CrossEntropyLoss()
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


# In[13]:


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
                       len_data = args.K,
                       sequence_sampling_distribution = "uniform")
iwl_dataset.sequences = train_dataset.sequences
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


# In[14]:


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


# In[18]:


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
                # test_metrics["length"].append([i_seq+1] * logsoftmaxloss.shape[0])
                # test_metrics["logsoftmaxloss"].append(logsoftmaxloss.detach().cpu().numpy())
                
                # test_metrics["accuracy"].append(is_correct.detach().cpu().numpy())
                # else: 

                # print ("i_seq", i_seq, "logsoftmaxloss", logsoftmaxloss.shape )
                
    # for c in test_metrics:
    #     # test_metrics[c] = np.concatenate(test_metrics[c]) 
    #     try: 
    #         print("test", c,  [_.shape for _ in test_metrics[c]])
    #     except:
    #         pass
    # print ("sequence_ranks", sequence_ranks.shape)
    # print ("lengths", lengths.shape)
    # print ("logsoftmaxlosses", logsoftmaxlosses.shape)
    # print ("accuracys", accuracys.shape)
    if args.K < 10000: # for small K, we can save the whole tensor 
        test_metrics["sequence_rank"] = sequence_ranks.detach().cpu().numpy()
        test_metrics["length"] = lengths.detach().cpu().numpy()
        test_metrics["logsoftmaxloss"] = logsoftmaxlosses.detach().cpu().numpy()
        test_metrics["accuracy"] = accuracys.detach().cpu().numpy()
    else:
        test_metrics["sequence_rank"] = sequence_ranks[:,0].detach().cpu().numpy() # average over positions
        test_metrics["length"] = 50
        test_metrics["logsoftmaxloss"] = logsoftmaxlosses.mean(dim=1).detach().cpu().numpy()
        test_metrics["accuracy"] = accuracys.mean(dim=1).detach().cpu().numpy()
    
    # test_metrics = pd.DataFrame(test_metrics)
    # df = test_metrics.groupby(["sequence_rank", "length"]).mean()
    # pivot = test_metrics.pivot_table(index="length", columns="sequence_rank", values="logsoftmaxloss")
    # import pandas as pd 
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(pivot, annot=False)
    # plt.show()
    return test_metrics
 


# In[ ]:


import json
import pickle
# import matplotlib.pyplot as plt
if os.path.exists(f"./cache/{args.wandb_group_name}") == False:
    os.makedirs(f"./cache/{args.wandb_group_name}", exist_ok=True)
exp_name = f"./cache/{args.wandb_group_name}/{args.wandb_group_name}_{args.fileprefix}_K_{args.K}_L_{args.len_context}_hidden_{args.num_hidden_features}_nheads_{args.num_heads}_nlayers_{args.num_layers}_{time.time()}.pkl"
print("Saving to", exp_name)
num_iters_per_epoch = 1000
num_apppearances = np.zeros(args.K)
test_every = np.log10(args.K).astype(int) * 2
for iter in range(args.num_iters // num_iters_per_epoch):
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
        loss = criterion(preds[:,:-1,:].reshape(B * (N-1), D), seq[:,1:].reshape(B * (N-1)))
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), seq.size(0)) 
        
    # scheduler.step()
    # print ("loss" , loss, "preds", preds.shape, "target", seq.shape)
    
    logs.update({
            "train_loss": losses.avg,
            "epoch": iter,
            "lr": optimizer.param_groups[0]['lr'],  
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
        with open(exp_name, "wb") as f:
            pickle.dump(record, f)
        # use json
        # with open(exp_name, "w") as f:
            # json.dump(record, f)
  
if args.wandb_log != True:
    with open(exp_name, "wb") as f:
        pickle.dump(record, f)
sys.exit(0)


# In[ ]:


import pickle
import traceback
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
 
wandb_group_name = "memo_dec20_uniformdist_modelsize"
# Filter runs by group name
# Initialize lists to store data for the heatmap
n_heads_list = []
n_layers_list = []
accuracy_list = []
loss_list = []
item=0
runs = glob(f"./cache/{wandb_group_name}/{wandb_group_name}_*.pkl")

import json
from collections import defaultdict
for run in runs :
    # Extract model configurations
    print(run)
    try: 
        with open(run, "rb") as f:
            record = pickle.load(f)
    except Exception as e:
        print(traceback.format_exc()) 
        continue 
    # record = json.loads(run)
    print(record.keys(), len(record["logs"]), record["logs"][0].keys())
    n_heads_list.append(record["args"]["num_heads"]) 
    n_layers_list.append(record["args"]["num_layers"])
    loss_across_positions = [v for k, v in record["logs"][-1].items() if "test_loss_" in k] 
    loss_list.append(np.mean(loss_across_positions))
    for k, v in record["logs"][-1].items():
        if "test_top1_" in k:
            try: 
                accuracy_across_positions += [v.item()] 
            except: 
                accuracy_across_positions += [v]
    accuracy_list.append(np.mean(accuracy_across_positions)) 
    print ("loss_across_positions", loss_across_positions)


# In[ ]:


import json
for run in runs[0:1]:
    # Extract model configurations
    print(run)
    # try: 
    with open(run, "rb") as f:
        record = pickle.load(f)
    # record = json.loads(run)
    print(record.keys(), len(record["logs"]), record["logs"][0].keys())
    # except Exception as e: 
    #     print (traceback.format_exc())
    #     continue
    
    # Extract model configurations
    for i, l in enumerate(record["logs"]):
        test = l.get("test_metrics")
        if len(test) > 0:
            print(test.keys())
            test = pd.DataFrame(test)
            #display(test)
            pivot = test.pivot_table(index="length", columns="sequence_rank", values="logsoftmaxloss")
            #display(pivot)
            # plt.figure(figsize=(12,8))
            # sns.heatmap(pivot, annot=False, vmin=0, vmax=1)
            # plt.show()
            pivot = test.pivot_table(index="length", columns="sequence_rank", values="accuracy")
            #display(pivot)
            # plt.figure(figsize=(12,8))
            # sns.heatmap(pivot, annot=False)
            # plt.show()
            K = record["args"]["K"]
            p = np.array([1/(i+1) for i in range(K)])
            p /= np.sum(p)
            expected_number_of_presentations = p * record["args"]["num_iter_per_epoch"] * len(record["logs"])
            # print ("expected_number_of_presentations", expected_number_of_presentations)
            loss_at_20 = test[test["length"] == 20]["logsoftmaxloss"] 
            display (loss_at_20)
        else:
            print ("epoch", i, "no test_metrics")
            


# In[ ]:


#     n_heads = record["args"].get("n_heads")
#     n_layers = record["args"].get("n_layers")
    

#     # Check if both n_heads and n_layers exist, skip run if any of them are missing.
#     if n_heads is None or n_layers is None:
#         print(f"Skipping run {run} because n_heads or n_layers is missing.")
#         continue
    
    
#     # Determine the plotting epoch
#     plot_epoch = min(10, len(record["logs"])) 
#     item = record["args"].get("len_context") - 2 # position 98 predicts position 99 (last token)

#     # Skip run if it doesn't have enough history
#     if plot_epoch == 0:
#         print(f"Skipping run {run} because it has no training history.")
#         continue

#     # Extract accuracy at the plot epoch
#     try:
#         # accuracy = run.history().iloc[plot_epoch-1][f"test_top1_{item}"] # index is 0-based, epoch is 1-based
#         accuracy = record["logs"][plot_epoch-1][f"test_top1_{item}"].cpu().numpy()
#         print(accuracy)
#     except KeyError:
#         print(f"Skipping run {run} because test_top1_98 not found in history.")
#         continue

#     # Append data to the lists
#     n_heads_list.append(n_heads)
#     n_layers_list.append(n_layers)
#     accuracy_list.append(accuracy)
#     print('accuracy_list',len(accuracy_list),len(accuracy))
    

# # Create a DataFrame from the collected data
# df = pd.DataFrame({
#     "n_heads": n_heads_list,
#     "n_layers": n_layers_list,
#     "accuracy": accuracy_list
# })
# display(df)
# # Create a pivot table for the heatmap
# pivot_df = df.pivot_table(index="n_layers", columns="n_heads", values="accuracy", aggfunc='mean')

# # Create the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': f'Test Top-1 Accuracy @ {item}th item'})
# plt.title("Test Top-1 Accuracy @ 98th Sequence vs Model Size")
# plt.xlabel("Number of Heads")
# plt.ylabel("Number of Layers")
# plt.tight_layout() # Adjust layout to prevent labels from overlapping
# plt.show()

