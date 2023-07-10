
import argparse
import os
import random
import copy
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

import datetime
import utils
import numpy as np
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--epochs', default=90, type=int,  
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument(
            '--coarsegrain_blocksize', 
            default=1, type=int,
            help = "if coarse graining, what is the block size of the coarse graining")
parser.add_argument('--train_method', default=90, type=str, 
                    help='train method (either ridge regression or gradient descent)')
parser.add_argument('--num_train_samples', default=1, type=int,  
                    help='fraction of training set to use')

parser.add_argument('--num_hidden_features', default=0, type=int,  
                    help='number of hidden fourier features')
parser.add_argument('--nonlinearity', default="tanh", type=str,  
                    help='type of nonlinearity used')
parser.add_argument('--gamma_regularize', default=1e-6, type=float,  
                    help='fraction of training set to use')
parser.add_argument('--SNR', default=1e-6, type=float,  
                    help='fraction of training set to use')
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store')
parser.add_argument(
            '--save_dir', 
            default="/scratch/gpfs/qanguyen/imagenet_info",
            type=str, 
            action='store')

class RandomFeaturesImagenet(datasets.ImageFolder):
    def __init__(self, root = "./data",  
                 target_size = 224,
                 phase = "train",
                 normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                 block_size = 1, 
                 width_after_pool = 224,
                 seed = None,
                 SNR = 1,
                 noise_seed = 1
                ):
        
        assert phase in ["train", "test"], f"phase must be either 'train' or 'test', got {phase} instead"
        self.target_size = target_size
        
        self.block_size = block_size        
        self.avg_kernel = nn.AvgPool2d(block_size, stride=block_size)
        self.rem_idx = target_size % block_size
        
        if phase == "train":
            transform =  transforms.Compose([
                    #transforms.RandomResizedCrop(self.target_size),
                    #transforms.RandomHorizontalFlip(),
                    transforms.Resize(256),
                    transforms.CenterCrop(self.target_size),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                normalize,
            ])
        rng = np.random.default_rng(42)
        
        self.teacher = torch.tensor(rng.standard_normal(size=3*(width_after_pool)*(width_after_pool))).float()
        print("teacher", self.teacher[:100])
        self.sqrtD = np.sqrt(3*(width_after_pool)*(width_after_pool))
        super(RandomFeaturesImagenet, self).__init__(root, transform)
        noise_rng = np.random.default_rng(noise_seed)
        self.noise = noise_rng.standard_normal(size=self.__len__()) / np.sqrt(SNR)
        print("noise", self.noise)
        
            
    #def __len__(self):
    #    return 256*10
    
    def __getitem__(self, index: int):
        sample, target = super(RandomFeaturesImagenet, self).__getitem__(index)
        sample = self.avg_kernel(sample).flatten()
        return sample, torch.dot(sample, self.teacher) / self.sqrtD + self.noise[index]
    
class RandomFeaturesNormal:
    def __init__(self, 
                 phase = "train",
                 width_after_pool = 224,
                 seed = None,
                 SNR = 1,
                 noise_seed = 1,
                 len = 1
                ):
        
       
        self.len = len
        rng = np.random.default_rng(42)
        self.teacher = torch.tensor(rng.standard_normal(size=3*(width_after_pool)*(width_after_pool))).float()
        #print("teacher", self.teacher[:100])
        
        self.D = (3*(width_after_pool)*(width_after_pool))
        noise_rng = np.random.default_rng(noise_seed)
        self.noise = torch.tensor(noise_rng.standard_normal(size=self.__len__()) / np.sqrt(SNR)).float()
        self.sqrtD = np.sqrt(self.D)
        self.X = torch.tensor(noise_rng.standard_normal(size=(len, self.D) )).float()
        self.Y = torch.matmul(self.X, self.teacher) / self.sqrtD + self.noise
        
        print("X", self.X.shape, self.teacher.shape, self.Y.shape)
        #print("Y", self.Y[:100])
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]
    
def get_record(args):
    record = utils.dotdict(
                args = args,
                metrics = utils.dotdict(
                    train_mse = -1,
                    test_mse = -1
                )
                
        )  
    return record

def get_model(args):
    width_after_pool = math.floor((224 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1)
    random_features_model = torch.randn(3*(width_after_pool)*(width_after_pool), 
                                     args.num_hidden_features)
    return random_features_model
    
def get_dataset(args):
    if "ilsvrc_2012_classification_localization" in args.data:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')


        train_dataset = RandomFeaturesImagenet(root = traindir,  
                phase = "train",   
                block_size = args.coarsegrain_blocksize,
                width_after_pool = math.floor((224 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1),
                SNR = args.SNR,
                noise_seed = 2023
                )

        val_dataset = RandomFeaturesImagenet(root = valdir,  
                phase = "test",   
                block_size = args.coarsegrain_blocksize,
                width_after_pool = math.floor((224 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1),
                SNR = args.SNR,
                noise_seed = 2022
                )
        val_dataset.teacher = train_dataset.teacher.detach().clone()
        print("train teacher",train_dataset.teacher )
        print("val teacher",val_dataset.teacher )
        rng = np.random.default_rng(42)
        num_train = len(train_dataset)
        train_idx = rng.integers(low=0, high=num_train, size=args.num_train_samples)
        print(train_idx[:100])
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = None
    elif "random_normal_data" in args.data:
        train_dataset = RandomFeaturesNormal( 
                width_after_pool = math.floor((224 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1),
                SNR = args.SNR,
                noise_seed = 2023,
                len = args.num_train_samples
                )

        val_dataset = RandomFeaturesNormal( 
                width_after_pool = math.floor((224 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1),
                SNR = args.SNR,
                noise_seed = 2022,
                len = 20000
                )
        val_dataset.teacher = train_dataset.teacher.detach().clone()
        print("train teacher",train_dataset.teacher )
        print("val teacher",val_dataset.teacher )
        train_sampler = None
        val_sampler = None
   
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader, train_dataset.sqrtD

def get_nonlinearity(args):
    if args.nonlinearity == "relu":
        nonlinearity = nn.ReLU()
    elif args.nonlinearity == "tanh":
        nonlinearity = nn.Tanh()
    elif args.nonlinearity == "line":
        nonlinearity = nn.Identity()
    else:
        raise ValueError
        
    return nonlinearity

def train_ridge_regression(train_loader, device, random_features_model, nonlinearity, sqrtD, args):#, model, criterion, optimizer, scheduler, , args, record, record_weights):
    data_time = utils.AverageMeter('Data', ':6.3f')

    # switch to train mode
    # model.train()

    end = time.time()
    Zs = []
    Ys = []
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        print("images, target", images.shape, target.shape)
        
        random_features = nonlinearity(torch.matmul(images, random_features_model) / sqrtD )
        Zs.append(random_features)
        Ys.append(target)
        print("random_features ", random_features.shape )
        
    P = args.num_hidden_features
    D = math.floor((224 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1)
    N = args.num_train_samples
    Zs = torch.cat(Zs) # shape: N x P
    Ys = torch.cat(Ys) # shape: N x 1
    Cov_plus_diag = 1 / args.num_train_samples * torch.matmul(Zs.T, Zs) + P * args.gamma_regularize / D * torch.eye(Zs.shape[1],
                                                                                                                   device = device)
    Inv = torch.linalg.solve(Cov_plus_diag, Zs, left=False)
    alpha = 1 / N * torch.matmul(Ys.view(1, -1), Inv) 
    Y_pred = torch.matmul(Zs, alpha.T).squeeze(1)
    train_mse = torch.nn.functional.mse_loss(Y_pred, Ys)
    print("Zs", Zs.shape, "Inv", Inv.shape, "alpha", alpha.shape, "Y_pred", Y_pred.shape, train_mse)

    return train_mse, alpha

def validate_ridge_regression(val_loader, alpha, random_features_model, args, nonlinearity, sqrtD, device):

    with torch.no_grad():
        end = time.time()
        Ys = []
        Y_pred = []
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True)
            random_features = nonlinearity(torch.matmul(images, random_features_model) / sqrtD )
            Ys.append(target)
            pred = torch.matmul(random_features, alpha.T)
            Y_pred.append( pred)
            #print("pred",pred.shape)
            #if i > 5:break
        Ys = torch.cat(Ys) # shape: N x 1
        Y_pred = torch.cat(Y_pred).squeeze(1)
        test_mse = torch.nn.functional.mse_loss(Y_pred, Ys)
    return test_mse

def train_gradient_descent(train_loader, device, random_features_model, nonlinearity, sqrtD, args, record):#, model, criterion, optimizer, scheduler, , args, record, record_weights):
    losses = utils.AverageMeter('Loss', ':.4e')
    data_time = utils.AverageMeter('Data', ':6.3f')
    output_layer = torch.nn.Linear(in_features = args.num_hidden_features, out_features = 1).to(device).float()
    optimizer = torch.optim.SGD(output_layer.parameters(),  
                                    args.lr / (np.sqrt(args.num_hidden_features)),
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay
                                     )
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)
    
    # switch to train mode
    output_layer.train()
    for epoch in range(args.epochs):
        losses.reset()
        end = time.time()

        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            print("images, target", images.shape, target.shape)

            random_features = nonlinearity(torch.matmul(images, random_features_model) / sqrtD )
            optimizer.zero_grad()
            output = output_layer(random_features)
            #print("output", output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), images.size(0))
    
        print("Current average loss", losses.avg )
             
        record.metrics.train_mse[epoch] = losses.avg
    return losses.avg, output_layer

def validate_gradient_descent(val_loader, output_layer, random_features_model, args, nonlinearity, sqrtD, device):
    losses = utils.AverageMeter('Loss', ':.4e')
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)
    
    with torch.no_grad():
        end = time.time()
         
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True)
            random_features = nonlinearity(torch.matmul(images, random_features_model) / sqrtD )
            output = output_layer(random_features)
           
            loss = criterion(output, target)
            #print(loss.item())
            losses.update(loss.item(), images.size(0))
             
    return losses.avg

def main():
    args = parser.parse_args()
    print("Args", args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_features_model = get_model(args).to(device)
    train_loader, val_loader, sqrtD = get_dataset(args)
    record = get_record(args)
    nonlinearity = get_nonlinearity(args)
    # Save parameters
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}", )
   

    # train
    if args.train_method == "ridge_regression":
        train_mse, alpha = train_ridge_regression(train_loader, device, random_features_model, nonlinearity, sqrtD, args)
        test_mse = validate_ridge_regression(val_loader, alpha, random_features_model, args, nonlinearity, sqrtD, device)
        record.metrics.train_mse = train_mse
        record.metrics.test_mse = test_mse
    elif args.train_method == "gradient_descent":
        record.metrics.train_mse = {}
        train_mse, output_layer = train_gradient_descent(train_loader, device, random_features_model, nonlinearity, sqrtD, args, record)
        test_mse = validate_gradient_descent(val_loader, output_layer, random_features_model, args, nonlinearity, sqrtD, device)
        record.metrics.test_mse = test_mse
    # evaluate on validation set
    



    

    print("metrics", record.metrics)    
    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)
    #utils.save_checkpoint(record_weights, save_dir = args.save_dir, filename = f"weights_{args.exp_name}")

if __name__ == '__main__':
    main()
