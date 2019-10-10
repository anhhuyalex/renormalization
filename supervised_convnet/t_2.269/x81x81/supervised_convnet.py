import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import math
import numpy as np
import sys
import time

class SupervisedConvNet(nn.Module):
    def __init__(self, filter_size, square_size, hidden_size, num_hidden_layers,
                first_activation = "tanh", activation_func = "sigmoid",
                out_channels = 1, seed = time.time()):
        """
        Arguments:
        filter_size ~ size of the convolution kernel (3 x 3)
        square size ~ how many strides of convolution in the input
        """
        torch.manual_seed(seed)
        super(SupervisedConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        if first_activation == "tanh":
            self.first_activation = torch.tanh
        elif first_activation == "relu":
            self.first_activation = torch.nn.LeakyReLU(0.1)
        if activation_func == "sigmoid":
            self.activation_func = torch.sigmoid
        elif activation_func == "relu":
            self.activation_func = torch.tanh
        self.conv1 = nn.Conv2d(1, out_channels, filter_size, padding=0, stride = filter_size)
        self.first_linear = nn.Linear(self.out_channels * square_size ** 2, hidden_size)
        hidden_layer = [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        self.linear_hidden = nn.ModuleList(hidden_layer)
        self.linear_output = nn.Linear(hidden_size, 1)

        self.trace = []



    def forward(self, x):
        # add hidden layers with relu activation function

        x = self.first_activation(self.conv1(x)).view(-1, 1, self.out_channels * self.square_size**2)
        x = self.activation_func(self.first_linear(x))
        for linear in self.linear_hidden:
            x = self.activation_func(linear(x))
        x = torch.sigmoid(self.linear_output(x))
        x = x.squeeze(1)
        # print("input", x)
        # print("convolution", convolution)
        # print("hidden_output", hidden_output)
        # print("output", output)

        return x
        
class MultiLayerConvNet(nn.Module):
    def __init__(self, filter_size, square_size, hidden_size, num_hidden_layers, 
                num_conv_layers,
                first_activation = "tanh", activation_func = "sigmoid",
                out_channels = 1, seed = time.time()):
        """
        Arguments:
        filter_size ~ size of the convolution kernel (3 x 3)
        square size ~ how many strides of convolution in the input
        """
        torch.manual_seed(seed)
        super(MultiLayerConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        if first_activation == "tanh":
            self.first_activation = torch.tanh
        elif first_activation == "relu":
            self.first_activation = torch.nn.LeakyReLU(0.1)
        if activation_func == "sigmoid":
            self.activation_func = torch.sigmoid
        elif activation_func == "relu":
            self.activation_func = torch.tanh
        conv_layers = [nn.Conv2d(1, out_channels, filter_size, padding=0, stride = filter_size) 
                        for _ in range(num_conv_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.first_linear = nn.Linear(self.out_channels * square_size ** 2, hidden_size)
        hidden_layer = [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        self.linear_hidden = nn.ModuleList(hidden_layer)
        self.linear_output = nn.Linear(hidden_size, 1)

        self.trace = []



    def forward(self, x):
        # add hidden layers with relu activation function
        for conv_layer in self.conv_layers:
            x = self.first_activation(conv_layer(x))
            # print("x shape", x.shape)
        x = x.view(-1, 1, self.out_channels * self.square_size**2)
        x = self.activation_func(self.first_linear(x))
        for linear in self.linear_hidden:
            x = self.activation_func(linear(x))
        x = torch.sigmoid(self.linear_output(x))
        x = x.squeeze(1)
        # print("input", x)
        # print("convolution", convolution)
        # print("hidden_output", hidden_output)
        # print("output", output)

        return x

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

if __name__ == "__main__":
    hidden_size = 10
    out_channels = 1
    num_hidden_layers = 3
    model = SupervisedConvNet(filter_size = 3, square_size = 3, \
            hidden_size = hidden_size, out_channels = out_channels, num_hidden_layers = num_hidden_layers)
    print_model_parameters(model)
