import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys;
sys.path.insert(0, "../../")
import supervised_convnet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.model_selection import train_test_split


uncorrelated_data = np.load("ising81x81_temp2.269_uncorrelated81x81.npy")
correlated_data = np.load("../ising81x81_temp2.269.npy")
print("Uncorrelated", uncorrelated_data[10])
print("Correlated", correlated_data[0])


data = np.vstack((uncorrelated_data, correlated_data))
label = np.hstack((np.zeros(10000), np.ones(10000)))
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
# raise ValueError

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 100
# how many samples of training set to train on
train_size = 8000
# number of epochs to train the model
n_epochs = 300
# learning rate
lr = 0.03
# weight decay
weight_decay = 0.0001
# hidden layer size
hidden_size = 10
# adjust learning rate?
adjust_learning_rate = False

# specify loss function
criterion = nn.BCELoss()

# tensorboard for tracking
writer = SummaryWriter(comment="--batch size {}, training set {}, epoch {}, lr {}, \
                        weight decay {}, hidden_size {}".format(
                        batch_size, train_size, n_epochs, lr, weight_decay, hidden_size
))

# parameters
param = {}
param['conv2d.weight'] = torch.tensor([[[[1.0, 1.0, 1.0],
          [1.0, 0.0, 1.0],
          [1.0, 1.0, 1.0]]]])/8
param['conv2d.bias'] = torch.tensor([0.0])

# Load parameters
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
        self.conv2d.weight = torch.nn.Parameter(param['conv2d.weight'], requires_grad=False)
        self.conv2d.bias = torch.nn.Parameter(param['conv2d.bias'], requires_grad=False)
        self.linear_hidden = nn.Linear(square_size ** 2, hidden_size)
        self.linear_output = nn.Linear(hidden_size, 1)


    def forward(self, x):
        # add hidden layers with relu activation function
        convolution = (self.conv2d(x)).view(-1, 1, self.square_size**2)

        hidden_output = torch.sigmoid(self.linear_hidden(convolution))

        output = torch.sigmoid(self.linear_output(hidden_output))

        # print("input", x)
        # print("convolution", convolution)
        # print("hidden_output", hidden_output)
        # print("output", output)

        return convolution, hidden_output, output


# build model
model = SupervisedConvNet(filter_size = 3, square_size = 27, hidden_size = hidden_size)

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)


# prepare data loaders
train_isingdataset = supervised_convnet.IsingDataset(X_train[:train_size], y_train[:train_size])
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
        accuracy += (torch.abs(target - output) < 0.5).sum().item() / batch_size
        train_loss += loss.item() * batch_size




    # print avg training statistics
    # train_loss = train_loss/len(train_loader)
    if epoch % 1 == 0:
        validate_accuracy = 0
        for batch_idx, (data, target) in enumerate(validate_loader):
            data = data.unsqueeze(1).type('torch.FloatTensor')
            target = target.type('torch.FloatTensor')
            output = model(data)[-1].view(-1)
            validate_accuracy += (torch.abs(target - output) < 0.5).sum().item() / batch_size
        print('Epoch: {} \t Train Loss: {} \t Validate_Accuracy: {}'.format(
            epoch,
            train_loss/len(train_loader),
            validate_accuracy/len(validate_loader),
            ))
        # supervised_convnet.print_model_gradient(model)

    # writer.add_scalar("validation_accuracy", validate_accuracy/len(train_loader))
    # print("trainLoss", train_loss/len(train_loader))
    # print("accuracy", accuracy/len(train_loader))
    model_params = supervised_convnet.get_param_histogram(model)
    model_grad = supervised_convnet.get_param_grad_histogram(model)
    writer.add_scalar("training_accuracy", accuracy/len(train_loader), global_step)
    # writer.add_scalar("validate_accuracy", validate_accuracy/len(validate_loader), global_step)
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
# patience = 0
# for batch_idx, (data, target) in enumerate(validate_loader):
#     data = data.unsqueeze(1).type('torch.FloatTensor')#[0].unsqueeze(1)
#     # print("data", data)
#     target = target.type('torch.FloatTensor')
#     optimizer.zero_grad()
#     output = model(data)[-1].view(-1)
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
#
#
# torch.save(model.state_dict(), "9x9->3x3.pt")
    # optimizer.step()
import numpy as np
s = np.random.multivariate_normal([0,0],[[1,3/5],[3/5,2]],size=1000000)
np.mean(s[:,0])
np.mean(s[:,1])
np.mean(np.prod(s, axis=1))
