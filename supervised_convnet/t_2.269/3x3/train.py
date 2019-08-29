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


uncorrelated_data = np.load("ising81x81_temp2.269_uncorrelated9x9.npy")
correlated_data = np.load("../ising81x81_temp2.269.npy")[:10000,:9,:9]
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
# how many samples of training set to validate on
validate_size = 2000
# number of epochs to train the model
n_epochs = 100
# learning rate
lr = 0.01
# weight decay
weight_decay = 0.000
# amsgrad use
amsgrad = False
# betas0
betas0 = 0.9
# betas1
betas1 = 0.99
# hidden layer size
hidden_size = 10
# number of hidden layers
num_hidden_layers = 3
# number of convolutional filters SupervisedConvNet
out_channels = 1
# adjust learning rate?ss
adjust_learning_rate = False

# specify loss function
criterion = nn.BCELoss()

# tensorboard for tracking
writer = SummaryWriter(comment="--batch size {}, training set {}, epoch {}, lr {}, \
                        weight decay {}, hidden_size {}, out_channels {}, num_hidden_layers {}".format(
                        batch_size, train_size, n_epochs, lr, weight_decay, hidden_size, out_channels, num_hidden_layers
))
model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 3, \
        hidden_size = hidden_size, out_channels = out_channels, num_hidden_layers = num_hidden_layers)

def trainer(model = model, batch_size = batch_size, train_size = train_size, n_epochs = n_epochs, lr = lr,
            weight_decay = weight_decay, adjust_learning_rate = adjust_learning_rate, amsgrad = amsgrad,
            betas0 = betas0, betas1 = betas1):
    print("Testing out: ")
    print("batch_size: ", batch_size)
    print("train_size: ", train_size)
    print("n_epochs: ", n_epochs)
    print("lr: ", lr)
    print("weight_decay: ", weight_decay)
    print("betas0: ", betas0)
    print("betas1: ", betas1)
    print("hidden_size: ", model.hidden_size)

    best_validate_accuracy = 0

    # build model


    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay, amsgrad = amsgrad)


    # prepare data loaders
    train_isingdataset = supervised_convnet.IsingDataset(X_train[:train_size], y_train[:train_size])
    train_loader = torch.utils.data.DataLoader(train_isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    validate_isingdataset = supervised_convnet.IsingDataset(X_train[-validate_size:], y_train[-validate_size:])
    validate_loader = torch.utils.data.DataLoader(validate_isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    supervised_convnet.print_model_parameters(model)

    global_step = 0
    first_epoch_validate_accuracy = 0
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
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            global_step += 1
            # update running training loss
            accuracy += (torch.abs(target - output) < 0.5).sum().item() / batch_size
            train_loss += loss.item() #* batch_size





        # print avg training statistics
        # train_loss = train_loss/len(train_loader)
        if epoch % 1 == 0:
            validate_accuracy = 0
            for batch_idx, (data, target) in enumerate(validate_loader):
                data = data.unsqueeze(1).type('torch.FloatTensor')
                target = target.type('torch.FloatTensor')
                output = model(data)
                output = output.squeeze(1)
                validate_accuracy += (torch.abs(target - output) < 0.5).sum().item()
            print('Epoch: {} \t Train Loss: {} \t Validate_Accuracy: {}'.format(
                epoch,
                train_loss/len(train_loader),
                validate_accuracy/validate_size,
                ))

            if validate_accuracy/validate_size > best_validate_accuracy:
                best_validate_accuracy = validate_accuracy/validate_size

            if epoch == 1:
                first_epoch_validate_accuracy = validate_accuracy/validate_size
            # supervised_convnet.print_model_gradient(model)

        # writer.add_scalar("validation_accuracy", validate_accuracy/len(train_loader))
        # print("trainLoss", train_loss/len(train_loader))
        # print("accuracy", accuracy/len(train_loader))
        model_params = supervised_convnet.get_param_histogram(model)
        model_grad = supervised_convnet.get_param_grad_histogram(model)
        # writer.add_scalar("training_accuracy", accuracy/len(train_loader), global_step)
        # # writer.add_scalar("validate_accuracy", validate_accuracy/len(validate_loader), global_step)
        # writer.add_scalar("parameter_mean", np.mean(model_params), global_step)
        # writer.add_scalar("parameter_grad_mean", np.mean(model_grad), global_step)
        # writer.add_scalar("parameter_std", np.std(model_params), global_step)
        # writer.add_scalar("parameter_grad_std", np.std(model_grad), global_step)
        # writer.add_histogram("parameter_histogram", model_params, global_step)
        # writer.add_histogram("parameter_grad_histogram", model_grad, global_step)

    print("model parameters! \n")
    supervised_convnet.print_model_parameters(model)

    # return last accuracy
    return validate_accuracy/validate_size, model.state_dict(), first_epoch_validate_accuracy
# writer.add_graph(model, data)
# writer.close()
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
