import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np

import importlib
import argparse
import datetime

import utils
import configs



def get_parser():
    """
    Possible save_dir
        "/gpfs/scratch60/turk-browne/an633/aha"
        "/gpfs/milgram/scratch60/turk-browne/an633/aha"
    """
    parser = argparse.ArgumentParser(
            description='Pytorch training framework for general dist training')
    parser.add_argument(
            '--save_dir', 
            default="/gpfs/milgram/scratch60/turk-browne/an633/renorm", type=str, 
            action='store')
    parser.add_argument(
            '--model_name', 
            default="mlp", type=str, 
            action='store')
    parser.add_argument(
            '--num_inputs', 
            default=80, type=int, 
            action='store')
    parser.add_argument(
            '--order', 
            default=1, type=int, 
            action='store')
    parser.add_argument(
            '--random_coefs', 
            action='store_true')
    

    parser.add_argument('-d', '--debug', help="in debug mode or not", 
                        action='store_true')

    return parser

class PolynomialData(Dataset):
    def __init__(self, num_examples, num_inputs, order, random_coefs):
        self.num_examples = num_examples
        self.num_inputs = num_inputs
        self.order = order
        self.random_coefs = random_coefs
        if random_coefs == False:
            self.coefs = torch.rand(self.order+1) * 2 - 1 # coefficients from [-1,1]
            
    def __len__(self):
        return self.num_examples 
    
    def __getitem__(self, idx):
        inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
        if self.random_coefs == True:
            self.coefs = torch.rand(self.order+1) * 2 - 1 # random coefficients from [-1,1]

        outputs = [self.coefs[i] * torch.pow(inputs, i) for i in range(self.order + 1)]
        outputs = torch.sum(torch.stack(outputs), dim=0)
        points = torch.cat([inputs, outputs], dim=-1)
        return points, self.coefs[0]

class PolynomialTrainer(utils.BaseTrainer):
    def build_data_loader(self):
        batch_size = self.train_params['batch_size']
        
        trainset = PolynomialData(**self.data_params) 
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

        testset = PolynomialData(**self.data_params) 
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
            data = dict(
                inputs = None,
                labels = None
            )
            print_interval = 5000 // self.train_params['batch_size'],
            data_params = self.data_params,
            train_params = self.train_params,
            model_optim_params = self.model_optim_params,
            save_params = self.save_params,
            model = None
        )
        
    def after_train_iter(self, running_loss, inputs, labels):
        self.loss.backward()
        self.optimizer.step()
        
        # update running loss
        running_loss += self.loss.item()
        
        # save train data
        self.record["data"]["inputs"] = inputs
        self.record["data"]["labels"] = labels
        return running_loss
    
    def after_train_epoch(self, running_loss):
        self.record["metrics"]["train_loss_prog"].append(running_loss  / (self.iter + 1))
        
    def after_run(self):
        # Save model
        self.record["model"] = list(self.model.named_parameters())
        
        # Save record
        self.save_record()
        
    def train(self):
        num_epochs = self.train_params['num_epochs']
        for self.epoch in range(num_epochs):  # loop over the dataset multiple times
            #self.before_train_epoch()
            running_loss = 0.0
            for self.iter, data in enumerate(self.trainloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)#.squeeze(1)
                self.loss = self.criterion(outputs, labels)
                
                running_loss = self.after_train_iter(running_loss, inputs, labels)

            self.after_train_epoch(running_loss)
            print("running_loss", running_loss / (self.iter + 1))
        print('Finished Training')
        self.after_run()
        
def get_polynomial_configs(args):
    # Get model optim params
    if args.model_name == "mlp":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[1000,1000,1000,1000,1000],
                   activation = "relu", N_CLASSES = 1).cuda()
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)#, momentum=0.9)
    model_optim_params = dict(model = net, criterion = criterion, optimizer = optimizer, model_name = args.model_name)
    
    # Get data params
    data_params = dict(num_examples = 64*64, num_inputs = args.num_inputs, order = args.order, random_coefs = args.random_coefs)

    # Get train parameters
    train_params = dict(batch_size = 256, 
                        num_epochs = 500)

    # Get save params
    save_params = dict(save_dir = args.save_dir)
    
    cfg = dict(
        model_optim_params = model_optim_params,
        data_params = data_params,
        train_params = train_params,
        save_params = save_params
    ) 
    return cfg

def get_runner(args):
    
    def runner(parameter):
        print("parameters", parameter)
        cfg = get_polynomial_configs(args)
        
        
        # set save fname
        exp_name = f"{args.model_name}" \
                + f"_order_{args.order}" \
                + f"_numinputs_{args.num_inputs}" \
                + f"_random_coefs_{args.random_coefs}" \
                + f"_rep_{datetime.datetime.now().timestamp()}"
        cfg["save_params"]["exp_name"] = exp_name
        print("exp_name", exp_name)
        
        trainer = PolynomialTrainer(**cfg)
        trainer.train()
        
    return runner

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    for _ in range(1):
        runner = get_runner(args = args)
        runner(dict())

    
if __name__ == '__main__':
    main()