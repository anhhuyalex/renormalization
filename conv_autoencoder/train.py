import torch
import torch.nn as nn
import torch.utils.data
import conv_autoencoder
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys

# Batch size, channels, height, width

# train on 3 x 3
if sys.argv[1] == "3x3":
    data = np.load("ising81x81_temp1.npy")[:, :3, :3]

    v = conv_autoencoder.ConvAutoencoder(3, 1)

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(v.parameters(), lr=0.001)


    # Create training and test dataloaders
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(data[:8000], batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(data[:2000], batch_size=batch_size, num_workers=num_workers)

    # number of epochs to train the model
    n_epochs = 500
    l1_crit = nn.L1Loss(size_average=False)

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for d in train_loader:
            # no need to flatten images
            d = (d.unsqueeze(1)).type(torch.FloatTensor)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = v(d).view(-1, 1, 3, 3)
            # calculate the loss
            loss = criterion(outputs, d)
    #         for param in v.parameters():
    #             loss += (torch.abs(param)).mean()/200
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * batch_size
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))


    # output = v(x)
    # print("output", output.shape)
    # w = SummaryWriter(log_dir="./logs") 

    # w.add_graph(v, x)
 elif sys.argv[1] == "9x9":
    data = np.load("ising81x81_temp1.npy")[:, :9, :9]

    v = conv_autoencoder.ConvAutoencoder(3, 9)

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(v.parameters(), lr=0.001)


    # Create training and test dataloaders
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(data[:8000], batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(data[:2000], batch_size=batch_size, num_workers=num_workers)

    # number of epochs to train the model
    n_epochs = 500
    l1_crit = nn.L1Loss(size_average=False)

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for d in train_loader:
            # no need to flatten images
            d = (d.unsqueeze(1)).type(torch.FloatTensor)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = v(d).view(-1, 1, 3, 3)
            # calculate the loss
            loss = criterion(outputs, d)
    #         for param in v.parameters():
    #             loss += (torch.abs(param)).mean()/200
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * batch_size
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))


    # output = v(x)
    # print("output", output.shape)
    # w = SummaryWriter(log_dir="./logs") 

    # w.add_graph(v, x)

