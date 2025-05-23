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
import attention
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
            '--num_examples', 
            default=50000, type=int, 
            action='store')
    parser.add_argument(
            '--random_coefs', 
            default=False, 
            type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument(
            '--input_strategy', 
            default="random", 
            type=str)
    parser.add_argument(
            '--output_strategy', 
            default="evaluate_at_0", 
            type=str)
    parser.add_argument(
            '--is_online', 
            default=False, 
            type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument(
            '--noise', 
            default=0.0, type=float, 
            action='store')
    parser.add_argument(
            '--weight_decay', 
            default=0.0, type=float, 
            action='store')
    parser.add_argument(
            '--num_inputs_kept', 
            default=0, type=int, 
            action='store')
    parser.add_argument(
            '--attn_pe_weight', 
            default=0, type=float, 
            action='store')
    parser.add_argument(
            '--lr', 
            default=0.5e-3, type=float, 
            action='store')

    parser.add_argument('-d', '--debug', help="in debug mode or not", 
                        action='store_true')

    return parser

class PolynomialData(Dataset):
    def __init__(self, num_examples, num_inputs, order, random_coefs, input_strategy, is_online, noise, output_strategy):
        self.num_examples = num_examples
        self.num_inputs = num_inputs
        self.order = order
        self.random_coefs = random_coefs
        self.input_strategy = input_strategy
        self.is_online = is_online
        self.noise = noise
        self.output_strategy = output_strategy
        if random_coefs == False:
            self.coefs = torch.rand(self.order+1) * 20 - 10 # coefficients from [-10,10]
            
        # how to sample inputs
        if input_strategy == "linspace":
            self.inputs = torch.linspace(-1, 1, num_inputs) # inputs from [-1,1]
        elif input_strategy == "quenched":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
        elif input_strategy == "random" or input_strategy == "random_sort":
            pass # handle input creation when sampling
        else:
            raise ValueError(f"input_strategy {input_strategy} not supported")
        if self.is_online == False:
            self.data = [self.sample_point() for _ in range(self.num_examples)]
            utils.save_file_pickle("random_sorted_data", self.data)
            print(wa)
    def __len__(self):
        return self.num_examples 
    
    def sample_point(self):
        if self.input_strategy == "random":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
        elif self.input_strategy == "random_sort":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
            self.inputs, _ = torch.sort(self.inputs)

        if self.random_coefs == True:
            coefs = torch.rand(self.order+1) * 20 - 10 # random coefficients from [-1,1]
        
        
        outputs = [coefs[i] * torch.pow(self.inputs, i) for i in range(self.order + 1)]
        outputs = torch.sum(torch.stack(outputs), dim=0)
        outputs += self.noise * torch.randn_like(outputs) # add noise
        
        points = torch.cat([self.inputs, outputs], dim=-1)
        
        if self.output_strategy == "evaluate_at_2":
            target = [coefs[i] * torch.pow(torch.tensor([2]), i) for i in range(self.order + 1)]
            target = torch.sum(torch.stack(target), dim=0)
            target += self.noise * torch.randn_like(target) # add noise
            target = target.squeeze(0)
        else:
            target = coefs[0] 
        return points, target
    
    def __getitem__(self, idx):
        if self.is_online == True:
            return self.sample_point()
        else:
            points, intercept = self.data[idx]
            return points, intercept

class PolynomialTrainer(utils.BaseTrainer):
    def build_data_loader(self):
        batch_size = self.train_params['batch_size']
        
        trainset = PolynomialData(**self.data_params) 
        testset = PolynomialData(**self.data_params) 
        # if input_strategy is quenched, train and test must have the same inputs
        if self.data_params["input_strategy"] == "quenched":
            testset.inputs = trainset.inputs
            testset.data = [testset.sample_point() for _ in range(testset.num_examples)]
            print(testset.inputs, trainset.inputs)
        
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
        
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
        
        
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
            ),
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
        
        return running_loss
    
    def after_train_epoch(self, train_running_loss):
        self.record["metrics"]["train_loss_prog"].append(train_running_loss  / (self.iter + 1))
        
        # validate
        with torch.no_grad():
            val_running_loss = 0.0
            for self.val_iter, data in enumerate(self.testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs).squeeze(1)
                val_running_loss += self.criterion(outputs, labels).item()
            self.record["metrics"]["test_loss_prog"].append(val_running_loss  / (self.val_iter + 1))
            print("val", inputs[:5, :], labels[:5])
        return train_running_loss, val_running_loss
    
    def after_run(self):
        # Save model
        self.record["model"] = self.model
        
        # Save record
        self.save_record()
        
    def train(self):
        num_epochs = self.train_params['num_epochs']
        for self.epoch in range(num_epochs):  # loop over the dataset multiple times
            #self.before_train_epoch()
            train_running_loss = 0.0
            for self.iter, data in enumerate(self.trainloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                #print("inputs", inputs.shape)
                #plt.plot(inputs[0,:40].detach().cpu().numpy(), inputs[0,40:].detach().cpu().numpy(), marker="+" , linestyle='None')
                #plt.plot(inputs[1,:40].detach().cpu().numpy(), inputs[1,40:].detach().cpu().numpy(), marker="+" , linestyle='None')
                #plt.plot(inputs[2,:40].detach().cpu().numpy(), inputs[2,40:].detach().cpu().numpy(), marker="+",  linestyle='None')
                #plt.savefig('myfig')
                #print(labels[:3])
                #print(wa)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs).squeeze(1)
                self.loss = self.criterion(outputs, labels)
                
                train_running_loss = self.after_train_iter(train_running_loss, inputs, labels)

            train_running_loss, val_running_loss = self.after_train_epoch(train_running_loss)
            print(self.epoch, "train_running_loss", train_running_loss / (self.iter + 1), "val_running_loss", val_running_loss / (self.val_iter + 1))
            print(inputs[:5, :], labels[:5])
            
        print('Finished Training')
        self.after_run()
        
def get_polynomial_configs(args):
    # Get model optim params
    if args.model_name == "mlp_large":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[1000,1000,1000,1000,1000],
                   activation = "relu", N_CLASSES = 1).cuda()
    elif args.model_name == "mlp_small":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[100,100,100],
                   activation = "relu", N_CLASSES = 1).cuda()
    elif args.model_name == "resnet_small":
        net = utils.ResNet(input_size=args.num_inputs * 2, hidden_sizes=[100,100,100],
                   activation = "relu", N_CLASSES = 1).cuda()
    elif args.model_name == "mlp_small_batchnorm":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[100,100,100],
                   activation = "relu", N_CLASSES = 1, batch_norm = True).cuda()
    elif args.model_name == "mlp_large_batchnorm":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[1000,1000,1000,1000,1000],
                   activation = "relu", N_CLASSES = 1, batch_norm = True).cuda()
    elif args.model_name == "mlp_small_silence":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[100,100,100],
                   activation = "relu", N_CLASSES = 1).cuda()
        net.silence_fc1_weights(num_inputs_kept = args.num_inputs_kept)
    elif args.model_name == "attention_small":
        net = attention.SimpleViT(image_size=(args.num_inputs * 2, 1), patch_size = (1, 1),
                                  num_classes = 1,
                                  dim = 100,
                                  depth = 3, heads = 1,
                                  channels = 1,
                                  dim_head = 100,
                                  pe_weight = args.attn_pe_weight,
                                  mlp_dim=100).cuda()
    else:
        raise ValueError("network not supported")
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)#, momentum=0.9)
    model_optim_params = dict(model = net, criterion = criterion, optimizer = optimizer, model_name = args.model_name)
    
    # Get data params
    data_params = dict(num_examples = args.num_examples, num_inputs = args.num_inputs, order = args.order, random_coefs = args.random_coefs, input_strategy = args.input_strategy, is_online = args.is_online, noise=args.noise, output_strategy=args.output_strategy)

    # Get train parameters
    train_params = dict(batch_size = 256, 
                        num_epochs = 5000)

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
                + f"_numexamples_{args.num_examples}" \
                + f"_is_online_{args.is_online}" \
                + f"_num_inputs_kept_{args.num_inputs_kept}" \
                + f"_rep_{datetime.datetime.now().timestamp()}"
        cfg["save_params"]["exp_name"] = exp_name
        print("exp_name", exp_name)
        
        trainer = PolynomialTrainer(**cfg)
        trainer.train()
        
    return runner

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    for _ in range(3):
        runner = get_runner(args = args)
        runner(dict())

    
if __name__ == '__main__':
    main()