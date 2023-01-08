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
import os
import wandb

os.environ["WANDB_API_KEY"] = "74369089a72bb385ac20560974425f1e30fd2f94"
os.environ["WANDB_MODE"] = "offline"

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
            '--full_inputs', 
            default=60, type=int, 
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
            '--num_inputs_kept', 
            default=0, type=int, 
            action='store')
    parser.add_argument(
            '--sample_strategy', 
            default="coefs", type=str, 
            action='store')
    
    parser.add_argument(
            '--attn_pe_weight', 
            default=0, type=float, 
            action='store')
    parser.add_argument(
            '--lr', 
            default=1e-3, type=float, 
            action='store')
    parser.add_argument(
            '--weight_decay', 
            default=0.0, type=float, 
            action='store')
    parser.add_argument(
            '--first_layer_l1_regularize', 
            default=0.0, type=float, 
            action='store')
    parser.add_argument(
            '--num_train_epochs', 
            default=2000, type=int, 
            action='store')
    parser.add_argument(
            '--tags', 
            default="debug", type=str, 
            action='store')
    
    parser.add_argument('-d', '--debug', help="in debug mode or not", 
                        action='store_true')

    return parser

class PolynomialData(Dataset):
    def __init__(self, num_examples, num_inputs, order, random_coefs, 
                 input_strategy, is_online, noise, output_strategy,
                 sample_strategy, full_inputs = None):
        self.num_examples = num_examples
        self.num_inputs = num_inputs
        self.order = order
        self.random_coefs = random_coefs
        self.input_strategy = input_strategy
        self.is_online = is_online
        self.noise = noise
        self.output_strategy = output_strategy
        self.full_inputs = full_inputs
        if random_coefs == False:
            self.coefs = torch.rand(self.order+1) * 20 - 10 # coefficients from [-10,10]
            
        # how to sample inputs
        if input_strategy == "linspace":
            self.inputs = torch.linspace(-1, 1, num_inputs) # inputs from [-1,1]
        elif input_strategy == "quenched":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
        elif input_strategy in ["random" , "random_sort", "repeat"]:
            pass # handle input creation when sampling
        else:
            raise ValueError(f"input_strategy {input_strategy} not supported")
            
        if sample_strategy == "coefs":
            self.sample_point = self.sample_point_using_coefs
        elif sample_strategy == "roots":
            self.sample_point = self.sample_point_using_roots
            
        if self.is_online == False:
            self.data = [self.sample_point() for _ in range(self.num_examples)]
            
    def __len__(self):
        return self.num_examples 
        
    def sample_point_using_coefs(self):
        if self.input_strategy == "random":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
        elif self.input_strategy == "random_sort":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
            self.inputs, _ = torch.sort(self.inputs)
        elif self.input_strategy == "repeat":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1
             
            tiles = self.full_inputs // self.num_inputs
        if self.random_coefs == True:
            coefs = torch.rand(self.order+1) * 20 - 10 # random coefficients from [-1,1]
        
        
        outputs = [coefs[i] * torch.pow(self.inputs, i) for i in range(self.order + 1)]
        outputs = torch.sum(torch.stack(outputs), dim=0)
        outputs += self.noise * torch.randn_like(outputs) # add noise
        
        points = torch.cat([self.inputs, outputs], dim=-1).to(torch.float64) 
        
        if self.output_strategy == "evaluate_at_2":
            target = [coefs[i] * torch.pow(torch.tensor([2]), i) for i in range(self.order + 1)]
            target = torch.sum(torch.stack(target), dim=0)
            target += self.noise * torch.randn_like(target) # add noise
            target = target.squeeze(0)
        elif self.output_strategy == "evaluate_at_0":
            target = coefs[0].to(torch.float64) 
        else:
            raise ValueError(f"output_strategy {self.output_strategy} not supported")
            
        return points, target.to(torch.float64) 
    
    def sample_point_using_roots(self):
        # Sample root of polynomial and multiplicities
        e = torch.distributions.exponential.Exponential(rate=27.0/self.order) # guarantees about 90% of rates below 3
        multiplicity = torch.ceil(e.rsample([self.order]))
        sums = torch.cumsum(multiplicity, dim=0)
        sums = sums[sums <= self.order]
        multiplicity = multiplicity[:len(sums)]
        if sums[-1] < self.order:
            multiplicity = torch.cat([multiplicity, torch.tensor([self.order-sums[-1]])])
        roots = torch.cat([torch.normal(-1+2.2*i/(len(multiplicity)), 0.1, (1,)) for i in range(len(multiplicity))])
        #roots = torch.rand(len(multiplicity)) * 2 - 1 # roots from [-1,1]
        roots = torch.cat([torch.full([int(multiplicity[i])], (roots[i])) for i in range(len(multiplicity))])
#         print("roots", "multiplicity", roots, multiplicity)
        
        if self.input_strategy == "random":
            diff = torch.max(roots) - torch.min(roots) - 0.5
            self.inputs = torch.rand(self.num_inputs) * (diff) - diff/2 # inputs from [-1,1]
        elif self.input_strategy == "random_sort":
            self.inputs = torch.rand(self.num_inputs) * 2 - 1 # inputs from [-1,1]
            self.inputs, _ = torch.sort(self.inputs)
        elif self.input_strategy == "repeat":
            diff = torch.max(roots) - torch.min(roots) - 0.5
            self.inputs = torch.rand(self.num_inputs) * (diff) - diff/2 # inputs from [-1,1]
             
            tiles = self.full_inputs // self.num_inputs
            remainder = self.full_inputs % self.num_inputs
            self.inputs = torch.tile(self.inputs, dims=(tiles,))
            self.inputs = torch.cat([self.inputs, self.inputs[:remainder]])
            
            
        outputs = torch.prod(self.inputs.reshape(-1, 1) - roots.reshape(1, -1), dim=1)
        scalar = torch.max(torch.abs(outputs)).clone() 
        translate = torch.rand(1) * 2 - 1 # translate from [-1,1]
        outputs = outputs / scalar + translate

        outputs += self.noise * torch.randn_like(outputs) # add noise
        
        points = torch.cat([self.inputs, outputs], dim=-1).to(torch.float64) 
        
        if self.output_strategy == "evaluate_at_2":
            target = torch.prod(2-roots).to(torch.float64)  / scalar + translate
        elif self.output_strategy == "evaluate_at_0":
            target = torch.prod(0-roots).to(torch.float64)  / scalar + translate
        else:
            raise ValueError(f"output_strategy {self.output_strategy} not supported")
        return points, target.item()
    
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
        self.model = self.model_optim_params["model"].to(torch.float64)
        self.optimizer = self.model_optim_params["optimizer"]
        
        if self.model_optim_params["criterion"] == "mse_loss":
            self.criterion = self.mse_loss
        elif self.model_optim_params["criterion"] == "mse_loss_l1_first_layer":
            self.criterion = self.mse_loss_l1_first_layer
        
    def mse_loss(self, outputs, labels):
        return torch.nn.functional.mse_loss(outputs, labels) 
    
    def mse_loss_l1_first_layer(self, outputs, labels):
        """
        Compute loss + weight * l1_of_first_layer
        """
        loss = torch.nn.functional.mse_loss(outputs, labels) 
        l1 = self.model_optim_params["first_layer_l1_regularize"] * sum([torch.linalg.norm(p, 1) for p in self.model.fc1.parameters()]) 
        
        return loss + l1
        
        
    def initialize_record(self):
        self.record = dict(
            metrics = dict(
                train_loss_prog = [],
                train_mse_prog = [],
                test_loss_prog = [],
                test_mse_prog = [],
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
            model = None,
            success = False
        )
        
    def after_train_iter(self, running_loss, mse_loss, inputs, outputs, labels):
        self.loss.backward()
        self.optimizer.step()
        
        # update running loss
        running_loss += self.loss.item()
        mse_loss += self.mse_loss(outputs, labels).item()
        
        return running_loss, mse_loss
    
    def after_train_epoch(self, train_running_loss, train_mse_loss):
        self.record["metrics"]["train_loss_prog"].append(train_running_loss  / (self.iter + 1))
        self.record["metrics"]["train_mse_prog"].append(train_mse_loss  / (self.iter + 1))
        
        # validate
        with torch.no_grad():
            val_running_loss = 0.0
            val_mse_loss = 0.0
            for self.val_iter, data in enumerate(self.testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs).squeeze(1)
                val_running_loss += self.criterion(outputs, labels).item()
                val_mse_loss += self.mse_loss(outputs, labels).item()
            self.record["metrics"]["test_loss_prog"].append(val_running_loss  / (self.val_iter + 1))
            self.record["metrics"]["test_mse_prog"].append(val_mse_loss  / (self.val_iter + 1))
            #print("val", inputs[:5, :], labels[:5])
            
        # Log to wandb
        wandb.log({"train_running_loss": train_running_loss / (self.iter + 1), 
                   "train_mse_loss": train_mse_loss / (self.iter + 1), 
                  "val_running_loss": val_running_loss / (self.val_iter + 1),
                  "val_mse_loss": val_mse_loss / (self.val_iter + 1)})

        return train_running_loss, val_running_loss
    def after_run(self):
        # Save model
        self.record["model"] = self.model
        
        # Success !
        self.record["success"] = True
        
        # Save record
        self.save_record()
        
        # Send to wandb
        wandb.config = self.record
        
        
    def train(self):
        num_epochs = self.train_params['num_train_epochs']
        for self.epoch in range(num_epochs):  # loop over the dataset multiple times
            #self.before_train_epoch()
            train_running_loss, train_mse_loss = 0.0, 0.0
            for self.iter, data in enumerate(self.trainloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs).squeeze(1)
                self.loss = self.criterion(outputs, labels)
                
                train_running_loss, train_mse_loss = self.after_train_iter(train_running_loss, train_mse_loss, inputs, outputs, labels)

            train_running_loss, val_running_loss = self.after_train_epoch(train_running_loss, train_mse_loss)
            print(self.epoch, "train_running_loss", train_running_loss / (self.iter + 1), "val_running_loss", val_running_loss / (self.val_iter + 1))
            
        print('Finished Training')
        self.after_run()
        
def get_polynomial_configs(args):
    # Get model optim params
    if args.model_name == "mlp_large":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[1000,1000,1000],
                   activation = "relu", N_CLASSES = 1).cuda()
    elif args.model_name == "mlp_small":
        net = utils.MLP(input_size=args.num_inputs * 2, hidden_sizes=[100,100,100],
                   activation = "relu", N_CLASSES = 1).cuda()
    elif args.model_name == "mlp_small_repeat":
        net = utils.MLP(input_size=args.full_inputs * 2, hidden_sizes=[100,100,100],
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
        
    criterion = "mse_loss_l1_first_layer"
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)#, momentum=0.9)
    model_optim_params = dict(model = net, criterion = criterion, optimizer = optimizer, model_name = args.model_name,
                              lr = args.lr, first_layer_l1_regularize = args.first_layer_l1_regularize)
    
    # Get data params
    data_params = dict(num_examples = args.num_examples, num_inputs = args.num_inputs, order = args.order, random_coefs = args.random_coefs, input_strategy = args.input_strategy, is_online = args.is_online, noise=args.noise, output_strategy=args.output_strategy, sample_strategy=args.sample_strategy, full_inputs = args.full_inputs)

    # Get train parameters
    train_params = dict(batch_size = 256, 
                        num_train_epochs = args.num_train_epochs,
                        )

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
    
    l1_weights = [0.0, 0.00001, 0.00005, 0.0001]
    #l1_weights = [0.0005, 0.001, 0.005, 0.01]
    for l1 in l1_weights:
        for _ in range(2):
            args.first_layer_l1_regularize = l1
            print("Arguments", args)
            wandb_run = wandb.init(project="polynomial", entity="anhhuyalex", tags = [args.tags], reinit=True)
            runner = get_runner(args = args)
            runner(dict())
            wandb_run.finish()
    
if __name__ == '__main__':
    main()