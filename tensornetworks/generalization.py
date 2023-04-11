import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import importlib
import utils
import attention
import glob
from collections import defaultdict

import seaborn as sns
import pandas as pd

importlib.reload(utils)
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def show_plt_if_in_notebook(title=None):
    if in_notebook():
        plt.show()
    else:
        plt.savefig(title)
def get_record(order, is_online, extra = "", title = None, num_inputs_list = [], 
               num_inputs_kept_list = [],
               sample_strategy_list = [],
               input_strategy_list=[], 
               l1_list = [],
               order_list = [],
               outdir = "/scratch/gpfs/qanguyen/poly1/",
              palette = sns.color_palette("Set3", 10)

):
     
    warnings.filterwarnings(action='once')
    pars = defaultdict(list)
    #record_names = ['resnet18_rep_1673625625.387933.pkl', 'resnet18_rep_1673618285.140812.pkl', 
    #               'resnet18_rep_1673618285.124871.pkl', 'resnet18_rep_1673618315.142802.pkl',
    #               'resnet18_rep_1673618403.386129.pkl', 'resnet18_rep_1673618403.386123.pkl',
    #               'resnet18_rep_1673618381.887423.pkl', 'resnet18_rep_1673618474.561327.pkl',
    #               'resnet18_rep_1673625654.48548.pkl', 'resnet18_rep_1673614260.528039.pkl']
    
#     record_names = ['resnet18_rep_1673873160.304721.pkl', 'resnet18_rep_1673873238.133468.pkl',
#                   'resnet18_rep_1673873238.273423.pkl', 'resnet18_rep_1673873237.417957.pkl',
#                   'resnet18_rep_1673873267.69003.pkl', 'resnet18_rep_1673873267.029088.pkl',
#                   'resnet18_rep_1673873267.064156.pkl', 'resnet18_rep_1673873329.711897.pkl',
#                   'resnet18_rep_1673873329.711921.pkl', 'resnet18_rep_1673873376.545421.pkl']
    record_names = ['resnet18_rep_1674056611.865267.pkl', 'resnet18_rep_1674056612.392694.pkl',
                  'resnet18_rep_1674056625.227239.pkl', 'resnet18_rep_1674057094.9487.pkl',
                  'resnet18_rep_1674058958.82337.pkl', 'resnet18_rep_1674059096.368824.pkl',
                  'resnet18_rep_1674059129.166538.pkl']#, 
#                     'resnet18_rep_1673873329.711897.pkl',
#                   'resnet18_rep_1673873329.711921.pkl', 'resnet18_rep_1673873376.545421.pkl']
    record_names = glob.glob(f"{outdir}/*pth.tar")
    
    record_included = [int(r.split("_")[-1].split(".")[0]) for r in record_names]
    record_included = [((r > 1674600000) and (r < 1674800000)) for r in record_included]
    record_names = [ r for i,r in enumerate(record_names) if record_included[i] == True]
    print(record_names, len(record_names))
    for f in record_names :
        print(f)
        try:
            
            record = torch.load(f) 
        except Exception as e: 
            print(e)
      
        for epoch in range(record.curr_epoch + 1):
                
            pars["data_rescale"].append(record.data_rescale)
            pars["epoch"].append(epoch)
            pars["train_loss"].append(record.metrics.train_losses[epoch])
            pars["test_loss"].append(record.metrics.val_losses[epoch])
            pars["train_top5"].append(record.metrics.train_acc5[epoch])
            pars["test_top5"].append(record.metrics.val_acc5[epoch])
            pars["train_top1"].append(record.metrics.train_acc1[epoch])
            pars["test_top1"].append(record.metrics.val_acc1[epoch])
        #print(f, "noise", record["data_params"]["noise"])
    fig, ax = plt.subplots(figsize=(10, 6))
    pars = pd.DataFrame.from_dict(pars) 
    sns.lineplot(x = "epoch", y="train_loss", 
                  hue="data_rescale",
                 palette = palette, data=pars  )
 
    #ax.set(yscale="log")
    plt.legend()
    #plt.ylim(1e-4, 5.5)
    plt.title("Train loss vs. epochs")
    show_plt_if_in_notebook("train_loss_vs_epochs.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pars = pd.DataFrame.from_dict(pars) 
    sns.lineplot(x = "epoch", y="test_loss", 
                  hue="data_rescale",
                 palette = palette, data=pars  )
 
    #ax.set(yscale="log")
    plt.legend()
    #plt.ylim(1e-4, 5.5)
    plt.title("Test loss vs. epochs")
    show_plt_if_in_notebook("test_loss_vs_epochs.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pars = pd.DataFrame.from_dict(pars) 
    sns.lineplot(x = "epoch", y="train_top5", 
                  hue="data_rescale",
                 palette = palette, data=pars  )
 
    #ax.set(yscale="log")
    plt.legend()
    #plt.ylim(1e-4, 5.5)
    plt.title("Train top5 accuracy vs. epochs")
    show_plt_if_in_notebook("train_top5acc_vs_epochs.png") 
   
    fig, ax = plt.subplots(figsize=(10, 6))
    pars = pd.DataFrame.from_dict(pars) 
    sns.lineplot(x = "epoch", y="test_top5", 
                  hue="data_rescale",
                 palette = palette, data=pars  )
 
    #ax.set(yscale="log")
    plt.legend()
    #plt.ylim(1e-4, 5.5)
    plt.title("Test top5 accuracy vs. epochs")
    show_plt_if_in_notebook("test_top5acc_vs_epochs.png")
 
    fig, ax = plt.subplots(figsize=(10, 6))
    pars = pd.DataFrame.from_dict(pars) 
    sns.lineplot(x = "epoch", y="train_top1", 
                  hue="data_rescale",
                 palette = palette, data=pars  )
 
    #ax.set(yscale="log")
    plt.legend()
    #plt.ylim(1e-4, 5.5)
    plt.title("Train top1 accuracy vs. epochs")
    show_plt_if_in_notebook("train_top1acc_vs_epochs.png")
 
    fig, ax = plt.subplots(figsize=(10, 6))
    pars = pd.DataFrame.from_dict(pars) 
    sns.lineplot(x = "epoch", y="test_top1", 
                  hue="data_rescale",
                 palette = palette, data=pars  )
 
    #ax.set(yscale="log")
    plt.legend()
    #plt.ylim(1e-4, 5.5)
    plt.title("Test top1 accuracy vs. epochs")
    show_plt_if_in_notebook("test_top1acc_vs_epochs.png")
 
    
import warnings

# get_record(model_name = "mlp_small", order = 1, is_online=False, title = "Polynomial regression loss, order=1", order_list=[2,4,8,12,80])
# get_record(model_name = "mlp_small", order = "3", input_strategy_list=["random"], 
#            is_online=False, title = f"Polynomial regression loss, random inputs, final loss vs. num_inputs", 
#            num_inputs_list=np.arange(2, 32, 2), 
#            sample_strategy_list=["coefs"],
#            l1_listhttp://localhost:8072/notebooks/tensornetworks/generalization.ipynb# = None,
#            outdir = "/scratch/gpfs/qanguyen/poly_l1")

# get_record(model_name = "mlp_large", order = "*", input_strategy_list=["random"], 
#            is_online=False, title = f"Polynomial regression loss, random inputs, final loss vs. num_inputs", 
#            num_inputs_list=np.arange(2, 62, 2), 
#            sample_strategy_list=["roots"],
#            l1_list = [0.0],
#            order_list = [3,4],
#            outdir = "/scratch/gpfs/qanguyen/poly_roots")
workdir = "/scratch/gpfs/qanguyen"
# workdir = "."
get_record( order = "*", input_strategy_list=["repeat"], 
           is_online=False, title = f"Imagenet loss, loss vs. data_rescale", 
           num_inputs_list=np.arange(2, 62, 2), 
           sample_strategy_list=["roots"],
           l1_list = [0.0],
           order_list = [3,4,5],
           outdir = f"{workdir}/imagenet_info") 
