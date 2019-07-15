import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys

# Batch size, channels, height, width

# train on 3 x 3

data = np.load("ising81x81_temp2.269.npy")[:, :9, :9]
print("data", data[:10])
# Create uncorrelated samples
uncorrelated_data = []
for _ in range(10000):
    sample = np.random.randint(0, 10000, (3, 3))
    horizontal, vertical = np.random.randint(0, 3, (2, 3, 3))
    uncorrelated = []
    for i in range(3):
        tile = []
        for j in range(3):
            tile.append(data[sample[i, j], 3*horizontal[i, j]:(3*horizontal[i, j] + 3), \
                    3*vertical[i, j]:(3*vertical[i, j] + 3)])
        uncorrelated.append(np.hstack(tile))
    uncorrelated_data.append(np.vstack(uncorrelated))

uncorrelated_data = np.array(uncorrelated_data)
print("uncorrelated_data", uncorrelated_data[:10])
np.save("ising81x81_temp2.269_uncorrelated9x9.npy", uncorrelated_data)
# # print(sample, vertical, horizontal)