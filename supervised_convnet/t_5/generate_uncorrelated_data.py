import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys

# Batch size, channels, height, width

# train on 3 x 3

data = np.load("ising81x81_temp5.npy")
# print("data", data[:10])
# Create uncorrelated samples
uncorrelated_data = []

for _ in range(10000):
    sample = np.random.randint(0, 10000, (27, 27))
    horizontal, vertical = np.random.randint(0, 79, (2, 27, 27))
    uncorrelated = []
    for i in range(27):
        tile = []
        for j in range(27):
            tile.append(data[sample[i, j], horizontal[i, j]:(horizontal[i, j] + 3), \
                    vertical[i, j]:(vertical[i, j] + 3)])
        uncorrelated.append(np.hstack(tile))
    uncorrelated_data.append(np.vstack(uncorrelated))

uncorrelated_data = np.array(uncorrelated_data)
# print("uncorrelated_data", uncorrelated_data[0, :20, :20])
np.save("81x81/ising81x81_temp5_uncorrelated81x81.npy", uncorrelated_data)
# # print(sample, vertical, horizontal)
