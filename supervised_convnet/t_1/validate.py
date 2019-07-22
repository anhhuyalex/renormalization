import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys;
sys.path.insert(0, "../")
import supervised_convnet
import numpy as np
import sys
from sklearn.model_selection import train_test_split


uncorrelated_data = np.load("ising81x81_temp1_uncorrelated9x9.npy")
correlated_data = np.load("ising81x81_temp1.npy")[:10000,:9,:9]
print("Uncorrelated", uncorrelated_data[0])
print("Correlated", correlated_data[0])

data = np.vstack((uncorrelated_data, correlated_data))
label = np.hstack((-np.ones(10000), np.ones(10000)))
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
# raise ValueError

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 100
# number of epochs to train the model
n_epochs = 200
# learning rate
lr = 1
# adjust learning rate?
adjust_learning_rate = False

# specify loss function
criterion = nn.MSELoss()

# parameters
param = {}
param['conv2d.weight'] = torch.tensor([[[[-7.5402, -7.5463, -7.5373],
          [-7.5487, -7.5419, -7.5335],
          [-7.5399, -7.5374, -7.5378]]]])
param['conv2d.bias'] = torch.tensor([-8.4569])
param['linear.weight'] = torch.tensor([[-0.3792, -0.4312, -0.4115, -0.3076, -0.3455, -0.3570, -0.3672, -0.3415,
         -0.3874]])
param['linear.bias'] = torch.tensor([-2.2739])

# Load parameters
class FrozenSupervisedConvNet(supervised_convnet.SupervisedConvNet):
    def __init__(self, filter_size, square_size):
        super(FrozenSupervisedConvNet, self).__init__(filter_size, square_size)
        self.conv2d.weight = torch.nn.Parameter(param['conv2d.weight'])
        self.conv2d.bias = torch.nn.Parameter(param['conv2d.bias'])
        self.linear1.weight = torch.nn.Parameter(param['linear.weight'])
        self.linear1.bias = torch.nn.Parameter(param['linear.bias'])


# build model
model = FrozenSupervisedConvNet(filter_size = 3, square_size = 3)

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 0.1)


# prepare data loaders
train_isingdataset = supervised_convnet.IsingDataset(X_train[:2000], y_train[:2000])
train_loader = torch.utils.data.DataLoader(train_isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

validate_isingdataset = supervised_convnet.IsingDataset(X_train[-2000:], y_train[-2000:])
validate_loader = torch.utils.data.DataLoader(validate_isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
supervised_convnet.print_model_parameters(model)

# monitor training loss
accuracy = 0.0
train_loss = 0.0
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
    # update running training loss
    accuracy += (torch.sign(output) == target).sum().item() / batch_size
    train_loss += loss.item() * batch_size




# print avg training statistics
# train_loss = train_loss/len(train_loader)
# if epoch % 1 == 0:
validate_accuracy = 0
for batch_idx, (data, target) in enumerate(validate_loader):
    data = data.unsqueeze(1).type('torch.FloatTensor')
    for i in range(10,100):
        print("data", data[i])
        target = target.type('torch.FloatTensor')
        layer1, output = model(data)
        print ("layer1", layer1[i])
        print ("output", output.view(-1)[i])
        print ("label", target[i].item())
    validate_accuracy += (torch.sign(output) == target).sum().item() / batch_size
    break
print('Accuracy: {} \t Validate_Accuracy: {}'.format(
    accuracy/len(train_loader),
    validate_accuracy/len(validate_loader),
    ))


print("model parameters! \n")
supervised_convnet.print_model_parameters(model)
