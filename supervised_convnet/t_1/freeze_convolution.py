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

writer = SummaryWriter(comment="--freeze conv, batch size 2000, training set 2000, epoch 200, lr 1, weight decay 0.1")


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
batch_size = 2000
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
param['linear.weight'] = torch.tensor([[0.4481, 0.4216, 0.4671, 0.3929, 0.4573, 0.4340, 0.3999, 0.5145, 0.4143]])
param['linear.bias'] = torch.tensor([-1.2442])

# Load parameters
class FrozenSupervisedConvNet(supervised_convnet.SupervisedConvNet):
    def __init__(self, filter_size, square_size):
        super(FrozenSupervisedConvNet, self).__init__(filter_size, square_size)
        self.conv2d.weight = torch.nn.Parameter(param['conv2d.weight'], requires_grad = False)
        self.conv2d.bias = torch.nn.Parameter(param['conv2d.bias'], requires_grad = False)
        # self.linear1.weight = torch.nn.Parameter(param['linear.weight'])
        # self.linear1.bias = torch.nn.Parameter(param['linear.bias'])


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

global_step = 0
for epoch in range(1, n_epochs+1):
    print("epoch", epoch)
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
        loss.backward()
        optimizer.step()
        global_step += 1
        # update running training loss
        accuracy += (torch.sign(output) == target).sum().item() / batch_size
        train_loss += loss.item() * batch_size




    # print avg training statistics
    # train_loss = train_loss/len(train_loader)
    if epoch % 1 == 0:
        validate_accuracy = 0
        for batch_idx, (data, target) in enumerate(validate_loader):
            data = data.unsqueeze(1).type('torch.FloatTensor')
            target = target.type('torch.FloatTensor')
            output = model(data)[-1].view(-1)
            validate_accuracy += (torch.sign(output) == target).sum().item() / batch_size
        print('Epoch: {} \t Accuracy: {} \t Validate_Accuracy: {}'.format(
            epoch,
            accuracy/len(train_loader),
            validate_accuracy/len(validate_loader),
            ))
        # supervised_convnet.print_model_gradient(model)
    # writer.add_scalar("validation_accuracy", validate_accuracy/len(train_loader))
    # print("accuracy", accuracy/len(train_loader))
    supervised_convnet.print_model_gradient(model)
    model_params = supervised_convnet.get_param_histogram(model)
    model_grad = supervised_convnet.get_param_grad_histogram(model)
    writer.add_scalar("training_accuracy", accuracy/len(train_loader), global_step)
    writer.add_scalar("validate_accuracy", validate_accuracy/len(validate_loader), global_step)
    writer.add_scalar("parameter_mean", np.mean(model_params), global_step)
    writer.add_scalar("parameter_grad_mean", np.mean(model_grad), global_step)
    writer.add_scalar("parameter_std", np.std(model_params), global_step)
    writer.add_scalar("parameter_grad_std", np.std(model_grad), global_step)
    writer.add_histogram("parameter_histogram", model_params, global_step)
    writer.add_histogram("parameter_grad_histogram", model_grad, global_step)

print("model parameters! \n")
supervised_convnet.print_model_parameters(model)

# writer.add_graph(model, data)
writer.close()
