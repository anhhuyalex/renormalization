import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import supervised_convnet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.model_selection import train_test_split

w1_9x9_to_3x3 = (torch.load("9x9->3x3_from81x81.pt"))

class RenormalizerConvNet(nn.Module):
    def __init__(self, filter_size):
        super(RenormalizerConvNet, self).__init__()
        self.filter_size = filter_size
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)  
        self.conv2d.weight = torch.nn.Parameter(w1_9x9_to_3x3['conv2d.weight'])
        self.conv2d.bias = torch.nn.Parameter(w1_9x9_to_3x3['conv2d.bias'])
        

    def forward(self, x):
        # add hidden layers with relu activation function
        layer1 = torch.tanh(self.conv2d(x))
        return layer1

correlated_data = np.load("../ising81x81_temp1.npy")[:10000,:,:]
# data = np.vstack((uncorrelated_data, correlated_data))
data = (correlated_data)
# label = np.hstack((-np.ones(10000), np.ones(10000)))
label = np.hstack((np.ones(10000)))
print(data.shape)
# X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0, random_state=42)
# isingdataset = supervised_convnet.IsingDataset(X_train[:200], y_train[:200])
isingdataset = supervised_convnet.IsingDataset(data, label)
print(isingdataset.y)

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 10000
# number of epochs to train the model
n_epochs = 1
# learning rate
lr = 0.0001
# adjust learning rate?
adjust_learning_rate = False

# specify loss function
criterion = nn.MSELoss()

# build model
model = RenormalizerConvNet(3)

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    accuracy = 0.0

    # adjust learning rate
    if adjust_learning_rate == True:
        supervised_convnet.adjust_learning_rate(optimizer, epoch, lr)

    ###################
    # train the model #
    ###################
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1).type('torch.FloatTensor')
        target = target.type('torch.FloatTensor')
        optimizer.zero_grad()
        output = model(data).squeeze(1)
        # loss = criterion(output, target) 
        print("data", data[0])
        print("output", (output.detach().numpy())[0])
        print("target", (target)[:10])


np.save("../ising81x81->27x27_using_w1_temp1_correlated.npy", output.detach().numpy())