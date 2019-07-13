import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import supervised_convnet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.model_selection import train_test_split


class SupervisedConvNet(nn.Module):
    def __init__(self, filter_size, square_size):
        super(SupervisedConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)  
        self.linear1 = nn.Linear(self.square_size ** 2, 1)
        # self.linear2 = nn.Linear(100, 1)
        

    def forward(self, x):
        # add hidden layers with relu activation function
        layer1 = torch.tanh(self.conv2d(x))
        reshape = layer1.view(-1, 1, self.square_size**2)
        # print("reshape", reshape)
        layer2 = torch.tanh(self.linear1(reshape))
        return layer1, reshape, layer2

# load already generated uncorrelated (in generate_uncorrelated_data.py)
uncorrelated_data = np.load("../ising81x81->9x9_using_w1_temp1_uncorrelated.npy")
# correlated data is 27x27 so take only 9x9 first element
correlated_data = np.load("../ising81x81->27x27_using_w1_temp1_correlated.npy")[:10000,:9,:9] 
data = np.vstack((uncorrelated_data, correlated_data))
# data = (correlated_data)
label = np.hstack((-np.ones(10000), np.ones(10000)))
# label = np.hstack((np.ones(10000)))
# print(data.shape)
print(uncorrelated_data.shape)
print(correlated_data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
isingdataset = supervised_convnet.IsingDataset(X_train[:200], y_train[:200])
# isingdataset = supervised_convnet.IsingDataset(data, label)
# print(isingdataset.X[:10])
# print(isingdataset.y[:10])

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 200
# number of epochs to train the model
n_epochs = 1000
# learning rate
lr = 0.01
# adjust learning rate?
adjust_learning_rate = False

# specify loss function
criterion = nn.MSELoss()

# build model
model = SupervisedConvNet(3, 3)

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
        output = model(data)[-1].view(-1)
        loss = criterion(output, target) 
        # add regularization
        # for param in model.parameters():
        #     loss += ((param)**2).sum()/200
        loss.backward()
        optimizer.step()

        # update running training loss
        accuracy += (torch.sign(output) == target).sum().item()
    
    # print avg training statistics 
    # train_loss = train_loss/len(train_loader)
    if epoch % 10 == 0:
        print('Epoch: {} \tTraining Loss: {}'.format(
            epoch, 
            accuracy
            ))
        # print("data", data)
        # print("output", (output))
        # print("target", (target))

# patience = 0
# for batch_idx, (data, target) in enumerate(train_loader):
#     data = data.unsqueeze(1).type('torch.FloatTensor')#[0].unsqueeze(1)
#     # print("data", data)
#     target = target.type('torch.FloatTensor')
#     optimizer.zero_grad()
#     output = [i.view(-1) for i in model(data)]
#     print("data", data[:10])
#     print("output", (output[:10]))
#     print("target", target[:10])
#     v = torch.tensor([[[[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#         [-1., -1., -1., -1., -1., -1., -1., -1., -1.]]]])
#     print("correlated model(v)", model(v))
#     v = torch.tensor([[[[ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
#         [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
#         [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
#         [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
#         [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
#         [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
#         [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.],
#         [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.],
#         [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.]]]])
#     print("uncorrelated model(v)", model(v))
#     # loss = criterion(output, target[0])
#     # print("loss.data", loss.data)
#     # loss.backward()
#     patience += 1
#     if patience > 100:
#         break
    

v = torch.tensor([[[[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1.,  1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.]]]])
print("negative", model(v))
print("positive", model(-v))

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)

torch.save(model.state_dict(), "9x9->3x3_from27x27.pt")