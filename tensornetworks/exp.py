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
import configs

def get_parser():
    """
    Possible save_dir
        "/gpfs/scratch60/turk-browne/an633/aha"
        "/gpfs/milgram/scratch60/turk-browne/an633/aha"
    """
    parser.add_argument(
            '--save_dir', 
            default="/gpfs/milgram/scratch60/turk-browne/an633/renorm", type=str, 
            action='store')
    parser.add_argument(
            '--model_name', 
            default="mlp", type=str, 
            action='store')
    parser.add_argument(
            '--pixel_shuffled', type=bool,
            action='store')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    cfg = configs.get_configs(args)

    for _ in range(20):
        trainer = utils.CIFAR_trainer(**cfg),
                    model_optim_params = dict(model = net, criterion = criterion, optimizer = optimizer))
        trainer.train()
    
if __name__ == '__main__':
    main()