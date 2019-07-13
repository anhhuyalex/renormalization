import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import math
import numpy as np
import sys

class SupervisedConvNet(nn.Module):
    def __init__(self, filter_size, square_size):
        super(SupervisedConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)  
        self.linear1 = nn.Linear(filter_size ** 2, 1)
        # self.linear2 = nn.Linear(100, 1)
        

    def forward(self, x):
        # add hidden layers with relu activation function
        layer1 = torch.tanh(self.conv2d(x))
        reshape = layer1.view(-1, 1, self.square_size**2)
        layer2 = torch.tanh(self.linear1(reshape))
        # layer3 = torch.tanh(self.linear2(layer2))
        # layer3 = torch.clamp(layer3, 0, 1)
        # for row in x:
        #     print("row", row)
        #     for el in row[0]:
        #         print("el", el)
        # x = torch.tanh(self.decoder(x))
        return reshape, layer2#, layer3

class IsingDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.y = label
        
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 200))
    print ("learning rate", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr