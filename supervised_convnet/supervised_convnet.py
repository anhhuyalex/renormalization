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
    def __init__(self, filter_size, square_size, hidden_size):
        """
        Arguments:
        filter_size ~ size of the convolution kernel (3 x 3)
        square size ~ how many strides of convolution in the input
        """
        super(SupervisedConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)
        self.linear_hidden = nn.Linear(square_size ** 2, hidden_size)
        self.linear_output = nn.Linear(hidden_size, 1)


    def forward(self, x):
        # add hidden layers with relu activation function
        convolution = torch.tanh(self.conv2d(x)).view(-1, 1, self.square_size**2)

        hidden_output = torch.sigmoid(self.linear_hidden(convolution))

        output = torch.sigmoid(self.linear_output(hidden_output))

        # print("input", x)
        # print("convolution", convolution)
        # print("hidden_output", hidden_output)
        # print("output", output)

        return convolution, hidden_output, output

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

def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)


def print_model_gradient(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, "grad", param.grad)

def get_param_histogram(model):
    param_histogram = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_histogram.extend(param.data.reshape(-1))
    return np.array(param_histogram)

def get_param_grad_histogram(model):
    param_grad_histogram = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_grad_histogram.extend(param.data.reshape(-1))
    return np.array(param_grad_histogram)
