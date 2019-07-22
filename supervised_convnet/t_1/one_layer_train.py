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




correlated_data = np.vstack((-np.ones(shape=(500, 1, 9)), np.ones(shape=(500, 1, 9))))
uncorrelated_data = np.random.choice([-1, 1], size=(1000, 1, 9))
data = np.vstack((correlated_data, uncorrelated_data))
label = np.hstack((np.zeros(1000), np.ones(1000)))
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
# raise ValueError

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 200
# number of epochs to train the model
n_epochs = 100
# learning rate
lr = 1
# adjust learning rate?
adjust_learning_rate = False

# specify loss function
criterion = nn.BCELoss()

class SupervisedConvNet(nn.Module):
    def __init__(self, filter_size):
        """
        Arguments:
        filter_size ~ size of the convolution kernel (3 x 3)
        square size ~ how many strides of convolution in the input
        """
        super(SupervisedConvNet, self).__init__()
        self.linear1 = nn.Linear(filter_size ** 2, 1)


    def forward(self, x):
        # add hidden layers with relu activation function
        linear = self.linear1(x)
        layer2 = torch.sigmoid(linear)

        return linear, layer2#, layer3

# build model
model = SupervisedConvNet(filter_size = 3)

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 0.0)


# prepare data loaders
train_isingdataset = supervised_convnet.IsingDataset(X_train[:2000], y_train[:2000])
train_loader = torch.utils.data.DataLoader(train_isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

validate_isingdataset = supervised_convnet.IsingDataset(X_train[-2000:], y_train[-2000:])
validate_loader = torch.utils.data.DataLoader(validate_isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
supervised_convnet.print_model_parameters(model)

global_step = 0
for epoch in range(1, n_epochs+1):
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
        train_loss += loss.item()




    # print avg training statistics
    # train_loss = train_loss/len(train_loader)
    if epoch % 100 == 0:
        validate_accuracy = 0
        for batch_idx, (data, target) in enumerate(validate_loader):
            data = data.unsqueeze(1).type('torch.FloatTensor')
            target = target.type('torch.FloatTensor')
            output = model(data)[-1].view(-1)
            print(data[0])
            print(data[0])
            validate_accuracy += (torch.sign(output) == target).sum().item() / batch_size
        print('Epoch: {} \t train_loss: {} \t Validate_Accuracy: {}'.format(
            epoch,
            train_loss/len(train_loader),
            validate_accuracy/len(validate_loader),
            ))
        supervised_convnet.print_model_gradient(model)
    # writer.add_scalar("validation_accuracy", validate_accuracy/len(train_loader))
    # print("accuracy", accuracy/len(train_loader))
    # supervised_convnet.print_model_gradient(model)
    # model_params = supervised_convnet.get_param_histogram(model)
    # model_grad = supervised_convnet.get_param_grad_histogram(model)
    # writer.add_scalar("training_accuracy", accuracy/len(train_loader), global_step)
    # writer.add_scalar("validate_accuracy", validate_accuracy/len(validate_loader), global_step)
    # writer.add_scalar("parameter_mean", np.mean(model_params), global_step)
    # writer.add_scalar("parameter_grad_mean", np.mean(model_grad), global_step)
    # writer.add_scalar("parameter_std", np.std(model_params), global_step)
    # writer.add_scalar("parameter_grad_std", np.std(model_grad), global_step)
    # writer.add_histogram("parameter_histogram", model_params, global_step)
    # writer.add_histogram("parameter_grad_histogram", model_grad, global_step)

print("model parameters! \n")
supervised_convnet.print_model_parameters(model)

# writer.add_graph(model, data)
# writer.close()
for batch_idx, (data, target) in enumerate(validate_loader):
    data = data.unsqueeze(1).type('torch.FloatTensor')
    for i in range(0,10):
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
