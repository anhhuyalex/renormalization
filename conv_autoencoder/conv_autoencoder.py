import torch
simport torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import sys

class ConvAutoencoder(nn.Module):
    def __init__(self, filter_size, square_size):
        super(ConvAutoencoder, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.conv1d = nn.Conv1d(1, filter_size ** 2, 1, stride = 1) # up-sampling layer
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)  
        self.decoder = nn.Linear(1, filter_size ** 2)
        

        
    def forward(self, x):
        # add hidden layers with relu activation function
        x = F.relu(self.conv2d(x))
        x = x.view(-1, 1, self.square_size)
        
        x = torch.tanh(self.conv1d(x))
        # for row in x:
        #     print("row", row)
        #     for el in row[0]:
        #         print("el", el)
        # x = torch.tanh(self.decoder(x))
        return x.unsqueeze(1)

