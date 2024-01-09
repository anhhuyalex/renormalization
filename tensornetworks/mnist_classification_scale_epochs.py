import argparse
import os
import copy
import time
from enum import Enum

import torch
import torch.nn.functional as F
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
parser.add_argument('data', metavar='DIR', nargs='?', default='./data',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--epochs', default=90, type=int,  
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
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
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument(
            '--coarsegrain_blocksize', 
            default=None, type=int,
            help = "if coarse graining, what is the block size of the coarse graining")
parser.add_argument(
            '--target_size', 
            default=None, type=int,
            help = "target size of the coarse graining, using Wave's fractional coarse-graining scheme")
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--num_train_samples', default=1, type=int,  
                    help='fraction of training set to use')
parser.add_argument('--num_hidden_features', default=0, type=int,  
                    help='number of hidden fourier features')
parser.add_argument('--no_transform', action='store_true', 
                    help='no normalization')
parser.add_argument('--multiclass_lr', action='store_true', 
                    help='multi-class logistic regression')
parser.add_argument('--nonlinearity', default="tanh", type=str,  
                    help='type of nonlinearity used')
parser.add_argument('--upsample', action='store_true', default=False,
                        help='whether to upsample the downsampled data to original size')
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
    
def get_record(args):
    record = utils.dotdict(
                args = args,
                metrics = utils.dotdict(
                    train_mse = utils.dotdict(),
                    train_top1 = utils.dotdict(),
                    train_top5 = utils.dotdict(),
                    test_mse = utils.dotdict(),
                    test_top1 = utils.dotdict(),
                    test_top5 = utils.dotdict(),
                    weight_norm = utils.dotdict(), 
                    )
        )  
    return record

 
class RandomFeaturesMNIST(datasets.MNIST):
    def __init__(self, root = "./data",  
                 train = True,
                 transform = None,
                 block_size = 1, 
                 upsample = False,
                 target_size = None,
                ):
        
        
        self.block_size = block_size        
        self.avg_kernel = nn.AvgPool2d(block_size, stride=block_size)
        # self.avg_kernel no grad 
        for param in self.avg_kernel.parameters():
            param.requires_grad = False

        if target_size is not None:
            self.target_size = target_size
            self.transform_matrix = self.get_transformation_matrix(target_size, 28)
            self.retransform_matrix = self.get_transformation_matrix(28, target_size)
            
        self.upsample = upsample
         
        super(RandomFeaturesMNIST, self).__init__(root, train=train, transform=transform, download=True)
         
    def get_transformation_matrix(self, target_size, full_shape):
        # List of coarse-grained coordinates
        x1,y1=torch.meshgrid(torch.arange(target_size),torch.arange(target_size))
        x2,y2 = x1+1,y1+1
        x1,y1,x2,y2 = x1/target_size,y1/target_size,x2/target_size,y2/target_size
        r_prime = torch.vstack([x1.flatten(),x2.flatten(),y1.flatten(),y2.flatten()]).T 

        # List of fine-grained coordinates
        m1,n1=torch.meshgrid(torch.arange(full_shape),torch.arange(full_shape))
        m2,n2 = m1+1,n1+1
        m1,n1,m2,n2 = m1 / full_shape,n1 /full_shape,m2 / full_shape,n2 /full_shape 
        r = torch.vstack([m1.flatten(),m2.flatten(),n1.flatten(),n2.flatten()]).T 

        minrprimex1x2 = torch.minimum(r_prime[:,0], r_prime[:,1])
        minrx1x2 = torch.minimum(r[:,0], r[:,1])
        maxrprimex1x2 = torch.maximum(r_prime[:,0], r_prime[:,1])
        maxrx1x2 = torch.maximum(r[:,0], r[:,1])

        minrprimey1y2 = torch.minimum(r_prime[:,2], r_prime[:,3])
        minry1y2 = torch.minimum(r[:,2], r[:,3])
        maxrprimey1y2 = torch.maximum(r_prime[:,2], r_prime[:,3])
        maxry1y2 = torch.maximum(r[:,2], r[:,3])

        x1 = torch.maximum(minrprimex1x2.unsqueeze(0),minrx1x2.unsqueeze(1))
        x2 = torch.minimum(maxrprimex1x2.unsqueeze(0),maxrx1x2.unsqueeze(1)) 
        delta_x = torch.clamp(x2-x1,min=0)
        y1 = torch.maximum(minrprimey1y2.unsqueeze(0),minry1y2.unsqueeze(1))
        y2 = torch.minimum(maxrprimey1y2.unsqueeze(0),maxry1y2.unsqueeze(1)) 
        delta_y = torch.clamp(y2-y1,min=0)
        return delta_x * delta_y
        

    def __getitem__(self, index: int):
        sample, target = super(RandomFeaturesMNIST, self).__getitem__(index)
        if self.target_size is not None:
            sample = torch.matmul(sample.view(1, -1), self.transform_matrix)
            
            sample = sample.view(1, self.target_size, self.target_size) * self.target_size * self.target_size
            if self.upsample:
                sample = torch.matmul (sample.view(1, -1), self.retransform_matrix)
                sample = sample.view(1, 28, 28) * 28 * 28
        else:
            sample = self.avg_kernel(sample)
            if self.upsample:
                sample = torch.repeat_interleave(sample,   self.block_size, dim=1)
                sample = torch.repeat_interleave(sample,   self.block_size, dim=2)
                sample_size = sample.shape[-1]
                if sample_size != 28: # if not the original size, pad the last few pixels
                    remainder = 28 - sample_size # size of imagenet - current sample size
                    sample = torch.cat([sample, sample[:, :, -remainder:]], dim=-1) # pad the last few pixels
                    sample = torch.cat([sample, sample[:, -remainder:, :]], dim=-2) # pad the last few pixels
                

        return sample, target


def get_model(args, nonlinearity):
    
    if args.upsample == False:
        if args.target_size is not None:
            width_after_pool = args.target_size
        else:
            width_after_pool = math.floor((28 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1)
    else:
        width_after_pool = 28

    if args.multiclass_lr:
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width_after_pool**2, 10),
        ) 
    else:
        # 2 layer MLP
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width_after_pool**2, args.num_hidden_features),
            nonlinearity,
            nn.Linear(args.num_hidden_features, 10),
        )
    return model
    
def get_dataset(args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {'num_workers': args.workers,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
   
    if args.no_transform == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]) 
    
    train_dataset = RandomFeaturesMNIST(root = "./data", 
                                        train = True,
                                        transform = transform,
                                        block_size = args.coarsegrain_blocksize,
                                        target_size = args.target_size,
                                        upsample = args.upsample
                                         ) 
    
    
    val_dataset = RandomFeaturesMNIST(root = "./data",
                                      train = False,
                                      transform = transform,
                                      block_size = args.coarsegrain_blocksize,
                                      target_size = args.target_size,
                                      upsample = args.upsample
                                      )

    # Get a fixed subset of the training set
    rng = np.random.default_rng(42)
    num_train = len(train_dataset)
    train_idx = rng.integers(low=0, high=num_train, size=args.num_train_samples)
    print("train_idx", train_idx[:100])
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx) 
    val_sampler = None 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, shuffle=(train_sampler is None),
        **train_kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler, shuffle=(train_sampler is None), **test_kwargs)
    
    return train_loader, val_loader 

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
 
def train_gradient_descent(train_loader, val_loader, device, model, nonlinearity, args, record): 
    """ 
    Train a linear model using gradient descent

    :param device: the device to train on
    :param model: the model being trained
    :param nonlinearity: the nonlinearity used
    :param args: the arguments
    :param record: the record to save the metrics

    :return: the trained model
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    data_time = utils.AverageMeter('Data', ':6.3f')
     
    optimizer = torch.optim.SGD(model.parameters(),  
                                 lr=   args.lr , 
                                    weight_decay=args.weight_decay
                                     )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    
    
    args.epochs = int (args.epochs * (60000 / args.num_train_samples)) + 1
    
    for epoch in range(args.epochs):
        losses.reset() 
        top1.reset()
        top5.reset()
        model.train() # switch to train mode
        end = time.time()


        for i, (images, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(device, non_blocking=True).view(images.shape[0], -1) # flatten
            target = target.to(device, non_blocking=True)
                      
            output = model(images)
            print (model.state_dict()['3.weight'])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), images.size(0)) 
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            print("loss", loss.item(), "acc1", acc1[0], "acc5", acc5[0])
            
        # step scheduler
        scheduler.step()

        # save metrics
        # print("output",  torch.argsort(output, dim=-1), "target", target )
        print("Current average loss", losses.avg, top1.avg, top5.avg, "epoch", epoch, "norm of weights") 
        record.metrics.train_mse[epoch] = losses.avg
        record.metrics.train_top1[epoch] = top1.avg
        record.metrics.train_top5[epoch] = top5.avg
        

        val_losses, val_top1, val_top5 = validate_gradient_descent(val_loader, model, args, nonlinearity, criterion, device)
        record.metrics.test_mse[epoch] = val_losses
        record.metrics.test_top1[epoch] = val_top1
        record.metrics.test_top5[epoch] = val_top5
       
        print("val_losses, val_top1, val_top5", val_losses, val_top1, val_top5)

    # record.metrics.weight_norm = model.state_dict()
    return losses.avg, model, criterion

def validate_gradient_descent(val_loader, model, args, nonlinearity, criterion, device):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    with torch.no_grad():
          
        
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True).view(images.shape[0], -1) # flatten
            target = target.to(device, non_blocking=True)
                
             
            output = model(images)
             
            
            loss = criterion(output, target)
            #print(loss.item())
            losses.update(loss.item(), images.size(0))
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
 
    return losses.avg, top1.avg, top5.avg 

def main():
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
 
    print("Args", args)
    nonlinearity = get_nonlinearity(args)
    model = get_model(args, nonlinearity).to(device)
    train_loader, val_loader = get_dataset(args)
    
    record = get_record(args)
    
    # Save parameters
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}" )
    

    
    record.metrics.train_mse = {}
    record.metrics.distance_to_true = {}
    train_mse, model, criterion = train_gradient_descent(train_loader, val_loader, device, model, nonlinearity, args, record)
    
    val_losses, val_top1, val_top5 = validate_gradient_descent(val_loader, model, args, nonlinearity, criterion, device)
    record.metrics.test_mse[args.epochs] = val_losses
    record.metrics.test_top1[args.epochs] = val_top1
    record.metrics.test_top5[args.epochs] = val_top5
    print("val_losses, val_top1, val_top5", val_losses, val_top1, val_top5)
   
    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)

if __name__ == '__main__':
    main()
