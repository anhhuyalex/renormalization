import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import importlib
import argparse
import datetime

import utils
import configs

from ax.service.managed_loop import optimize

def get_parser():
    """
    Possible save_dir
        "/gpfs/scratch60/turk-browne/an633/aha"
        "/gpfs/milgram/scratch60/turk-browne/an633/aha"
    """
    parser = argparse.ArgumentParser(
            description='Pytorch training framework for general dist training')
    parser.add_argument(
            '--save_dir', 
            default="/gpfs/milgram/scratch60/turk-browne/an633/renorm", type=str, 
            action='store')
    parser.add_argument(
            '--model_name', 
            default="mlp", type=str, 
            action='store')
    parser.add_argument(
            '--lr', 
            default=1e-3, type=float, 
            action='store')
    parser.add_argument(
            '--pixel_shuffled', 
            action='store_true')
    return parser

def get_runner(args):
    
    def runner(parameter):
        print("parameters", parameter)
        cfg = configs.get_configs(args)
        
        # set params
        cfg["train_params"]["batch_size"] = parameter["batch_size"]
        for g in cfg["model_optim_params"]["optimizer"].param_groups:
            g['lr'] = parameter["lr"]
            
        # set save fname
        exp_name = f"{args.model_name}" \
                + f"_shuffled_{args.pixel_shuffled}" \
                + f"_lr_{parameter['lr']}" \
                + f"_rep_{datetime.datetime.now().timestamp()}"
        cfg["save_params"]["exp_name"] = exp_name
        print("exp_name", exp_name)
        
        trainer = utils.CIFAR_trainer(**cfg)
        trainer.train()
        test_loss = trainer.record["test_loss"]
        if np.isnan(test_loss):
            return 1e100
        else:
            return test_loss
        
    return runner

def main():
    parser = get_parser()
    args = parser.parse_args()
    

        
        
    hyperparameters = [
                    {"name": "batch_size", "type": "range", "bounds": [1, 256], "value_type": "int"},
                    {"name": "lr", "type": "range", "bounds": [1e-5, 1e-1], "value_type": "float", "log_scale": True}
                ]
    best_parameters, values, experiment, model = optimize(
            parameters=hyperparameters,
            evaluation_function=get_runner(args = args),
            minimize = True,
            total_trials=100
        )
    
if __name__ == '__main__':
    main()