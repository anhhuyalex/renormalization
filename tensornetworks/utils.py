import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.models as models

import numpy as np
import pickle

IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation, keep_rate=1.0, N_CLASSES = 10):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(self.input_size,
                                   self.hidden_sizes[0])
        self.dropout = torch.nn.Dropout(1 - keep_rate)
        self.hidden = nn.ModuleList()
        for k in range(len(self.hidden_sizes) - 1):
            self.hidden.append(nn.Linear(self.hidden_sizes[k], self.hidden_sizes[k+1]))
            
            
        self.out = torch.nn.Linear(self.hidden_sizes[-1], N_CLASSES)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        if self.activation == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        for lay in self.hidden:
            if self.activation == "sigmoid":
                x = sigmoid(lay(x))
            elif self.activation == "relu":
                x = torch.nn.functional.relu(lay(x))
            x = self.dropout(x)
        return (self.out(x))
    
class CNN(nn.Module):
    def __init__(self, conv1_chans = 6, conv2_chans = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_chans, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_chans, conv2_chans, 5)
        self.fc1 = nn.Linear(conv2_chans * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, n_classes):
        """ Alexnet Classifier
        Borrowed from https://github.com/ayshrv/cs7641-img-classification/blob/master/models/alexnet.py
        Arguments:
            n_classes (int): Number of classes to score
        """
        super(AlexNet, self).__init__()

        self.alexnet = models.alexnet(pretrained=False, num_classes=n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        """
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width
        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        unnormalised_scores = self.alexnet(images)
        #logits = self.softmax(unnormalised_scores)
        return unnormalised_scores

class ShuffledCIFAR10:
    def __init__(self, *, 
                 pixel_shuffled = None, image_width = IMAGE_WIDTH, train = True, download=False, transform = None):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=download, transform=transform)
        self.pixel_shuffled = pixel_shuffled
        if pixel_shuffled == True:
            self.image_width = image_width
            self.perm = torch.randperm(image_width * image_width)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        inputs, labels = self.dataset[idx]
        if self.pixel_shuffled == True:
            inputs = inputs.view(IMAGE_CHANNELS, self.image_width * self.image_width)
            inputs = inputs[:, self.perm].view(IMAGE_CHANNELS, self.image_width, self.image_width)
        return inputs, labels
            

class CIFAR_trainer:
    def __init__(self, data_params = dict(),
                       train_params = dict(batch_size = 4,
                                          num_epochs = 20),
                       model_optim_params = dict(),
                       save_params = dict()):
        
        self.data_params = data_params
        self.train_params = train_params
        self.model_optim_params = model_optim_params
        self.save_params = save_params
        self.setup()
        
    def setup(self):
        self.build_data_loader()
        self.build_model_optimizer()
        self.initialize_record()
        
    def build_data_loader(self):
        batch_size = self.train_params['batch_size']
        
        trainset = ShuffledCIFAR10(train=True, **self.data_params) 
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

        testset = ShuffledCIFAR10(train=False, **self.data_params) 
        if self.data_params["pixel_shuffled"] == True:
            testset.perm = trainset.perm
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    def build_model_optimizer(self):
        self.model = self.model_optim_params["model"]
        self.criterion = self.model_optim_params["criterion"]
        self.optimizer = self.model_optim_params["optimizer"]
        
    def initialize_record(self):
        self.record = dict(
            metrics = dict(
                train_loss_prog = [],
                train_acc_prog = [],
                test_loss_prog = [],
                test_acc_prog = [],
                test_loss = 0.0,
                test_accuracy = 0.0
            ),    
            print_interval = 5000 // self.train_params['batch_size'],
            data_params = self.data_params,
            train_params = self.train_params,
            model_optim_params = self.model_optim_params,
            save_params = self.save_params,
            model = None
        )
    def after_train_iter(self, running_loss, running_accuracy, outputs, labels):
        self.loss.backward()
        self.optimizer.step()
        
        # get top-1 accuracy
        pred_class = torch.argmax(outputs, dim=1)
        running_accuracy += torch.mean((pred_class == labels).float()).item()

        # print statistics
        running_loss += self.loss.item()
        if self.iter % self.record["print_interval"] == self.record["print_interval"] - 1:    # print every self.record["print_interval"] mini-batches
            running_loss /= self.record["print_interval"]
            running_accuracy /= self.record["print_interval"]
            self.record["metrics"]["train_loss_prog"].append(running_loss)
            self.record["metrics"]["train_acc_prog"].append(running_accuracy)
            print(f'[{self.epoch + 1}, {self.iter + 1:5d}] loss: {running_loss:.3f}, accuracy:  {running_accuracy:.3f}')
            running_loss = 0.0
            running_accuracy = 0.0
        return running_loss, running_accuracy
    
    def after_train_epoch(self):
        # get test loss and accuracy
        test_loss, test_accuracy = self.test()
        self.record["metrics"]["test_loss_prog"].append(test_loss)
        self.record["metrics"]["test_acc_prog"].append(test_accuracy)
        
        # put model back into train mode
        self.model.train()
        
    def train(self):
        num_epochs = self.train_params['num_epochs']
        for self.epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss, running_accuracy = 0.0, 0.0
            for self.iter, data in enumerate(self.trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                self.loss = self.criterion(outputs, labels)
                
                running_loss, running_accuracy = self.after_train_iter(running_loss, running_accuracy, outputs, labels)

            self.after_train_epoch()

        print('Finished Training')
        self.after_run()
        
    def test(self):
        running_loss, running_accuracy = 0.0, 0.0
        self.model.eval()
        for i, data in enumerate(self.testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            
            # get top-1 accuracy
            pred_class = torch.argmax(outputs, dim=1)
            running_accuracy += torch.mean((pred_class == labels).float()).item()

        # print loss
        running_loss /= len(self.testloader)
        running_accuracy /= len(self.testloader)
        print(f'Test loss: {running_loss:.3f}, accuracy: {running_accuracy:.3f}')
        return running_loss, running_accuracy
    
    def after_run(self):
        # Get final test loss and accuracy
        test_loss, test_accuracy = self.test()
        self.record["metrics"]["test_loss"] = test_loss
        self.record["metrics"]["test_accuracy"] = test_accuracy
        
        # Save model
        self.record["model"] = list(self.model.named_parameters())
        
        # Save record
        fname = f"{self.save_params['save_dir']}/{self.save_params['exp_name']}"
        save_file_pickle(fname, self.record)
        
    
def save_file_pickle(fname, file):
    with open(f"{fname}.pkl", 'wb') as f:
        pickle.dump(file, f)
        
def load_file_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)