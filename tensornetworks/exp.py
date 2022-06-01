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
    parser.add_argument(
            '--fix_permutation', 
            action='store_true')
    parser.add_argument(
            '--freeze_epoch', 
            default=-1, type=int, 
            action='store')

    parser.add_argument('-d', '--debug', help="in debug mode or not", 
                        action='store_true')

    return parser

def get_runner(args):
    
    def runner(parameter):
        print("parameters", parameter)
        args.pixel_shuffled = parameter["pixel_shuffled"]
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
        test_loss = trainer.record["metrics"]["test_loss"]
        if np.isnan(test_loss):
            return 1e100
        else:
            return test_loss
        
    return runner

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    for _ in range(50):
        # shuffled run
        runner = get_runner(args = args)
        runner(dict(batch_size = 64, lr = 1e-3, pixel_shuffled = True))

        # unshuffled run
        runner = get_runner(args = args)
        runner(dict(batch_size = 64, lr = 1e-3, pixel_shuffled = False))
#     from ax.service.managed_loop import optimize
#     hyperparameters = [
#                     #{"name": "batch_size", "type": "range", "bounds": [256, 257], "value_type": "int"},
#                     {"name": "lr", "type": "choice", "values": [1e-2], "value_type": "float"},
#                     {"name": "batch_size", "type": "choice", "values": [64], "value_type": "int"}
#                 ]
#     best_parameters, values, experiment, model = optimize(
#             parameters=hyperparameters,
#             evaluation_function=get_runner(args = args),
#             minimize = True,
#             total_trials=100
#         )
    
if __name__ == '__main__':
    main()