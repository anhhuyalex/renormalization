from torchmetrics.functional.classification import multiclass_calibration_error
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
            default=1, type=int,
            help = "if coarse graining, what is the block size of the coarse graining")
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--train_method', default=90, type=str, 
                    help='train method (either ridge regression or gradient descent)')
parser.add_argument('--num_train_samples', default=1, type=int,  
                    help='fraction of training set to use')
parser.add_argument('--upsample', action='store_true', default=False,
                        help='whether to upsample the downsampled data to original size')
parser.add_argument('--num_hidden_features', default=0, type=int,  
                    help='number of hidden fourier features')
parser.add_argument('--nonlinearity', default="tanh", type=str,  
                    help='type of nonlinearity used')
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
                    l1_calibration = utils.dotdict(),
                    l2_calibration = utils.dotdict(),
                    lmax_calibration = utils.dotdict(),
                    )
        )  
    return record

 
class RandomFeaturesMNIST(datasets.MNIST):
    def __init__(self, root = "./data",  
                 train = True,
                 transform = None,
                 block_size = 1, 
                 upsample = False
                ):
        
        
        self.block_size = block_size        
        self.avg_kernel = nn.AvgPool2d(block_size, stride=block_size)
        self.upsample = upsample
         
         
        super(RandomFeaturesMNIST, self).__init__(root, train=train, transform=transform, download=True)
         
    
    def __getitem__(self, index: int):
        sample, target = super(RandomFeaturesMNIST, self).__getitem__(index)

        
        sample = self.avg_kernel(sample)
        if self.upsample:
            sample = torch.repeat_interleave(sample,   self.block_size, dim=1)
            sample = torch.repeat_interleave(sample,   self.block_size, dim=2)
            sample_size = sample.shape[-1]
            remainder = 28 - sample_size # size of imagenet - current sample size
            sample = torch.cat([sample, sample[:, :, -remainder:]], dim=-1) # pad the last few pixels
            sample = torch.cat([sample, sample[:, -remainder:, :]], dim=-2) # pad the last few pixels
            
         
        return sample, target


def get_model(args):
    if args.upsample == False:
        width_after_pool = math.floor((28 - args.coarsegrain_blocksize) / args.coarsegrain_blocksize + 1)
    else:
        width_after_pool = 28

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate numbers from a normal distribution
    mean = 0  # Mean of the normal distribution
    std_dev = 1  # Standard deviation of the normal distribution
    size = [1*(width_after_pool)*(width_after_pool), args.num_hidden_features]  # Number of samples to generate

    # Generate the random features
    random_features_model = torch.tensor(np.random.normal(mean, std_dev, size)).float()
    print("random_features_model", random_features_model[0])
    return random_features_model
    
def get_dataset(args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {'num_workers': args.workers,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
   
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]) 
    train_dataset = RandomFeaturesMNIST(root = "./data", 
                                        train = True,
                                        transform = transform,
                                        block_size = args.coarsegrain_blocksize,
                                        upsample = args.upsample
                                         ) 
    
    
    val_dataset = RandomFeaturesMNIST(root = "./data",
                                      train = False,
                                      transform = transform,
                                      block_size = args.coarsegrain_blocksize,
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
 
def train_gradient_descent(train_loader, val_loader, device, random_features_model, nonlinearity, args, record): 
    """ 
    Train a linear model using gradient descent

    :param device: the device to train on
    :param random_features_model: the random features model
    :param nonlinearity: the nonlinearity used
    :param args: the arguments
    :param record: the record to save the metrics

    :return: the trained model
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    output_layer = torch.nn.Linear(in_features = args.num_hidden_features,  
                                   out_features = 10, 
                                   bias = True
                                   ).to(device).float()
    optimizer = torch.optim.Adam(output_layer.parameters(),  
                                 lr=   args.lr , 
                                    weight_decay=args.weight_decay
                                     )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    
    # switch to train mode
    output_layer.train()
    for epoch in range(args.epochs):
        losses.reset()
        end = time.time()

        for i, (images, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(device, non_blocking=True).view(images.shape[0], -1) # flatten
            target = target.to(device, non_blocking=True)
            sqrtD = float(math.sqrt(images.shape[1])) # sqrt of dimension
            
            random_features = nonlinearity(torch.matmul(images, random_features_model) / sqrtD )
            
            output = output_layer(random_features)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), images.size(0)) 
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            #print("loss", loss.item())
        # step scheduler
        scheduler.step()

        # save metrics
        # print("output",  torch.argsort(output, dim=-1), "target", target )
        print("Current average loss", losses.avg, top1.avg, top5.avg, "epoch", epoch, "norm of weights", torch.norm(output_layer.weight.data))
        record.metrics.train_mse[epoch] = losses.avg
        record.metrics.train_top1[epoch] = top1.avg
        record.metrics.train_top5[epoch] = top5.avg
        record.metrics.weight_norm[epoch] = torch.norm(output_layer.weight.data)

        val_losses, val_top1, val_top5, l1_calibration, l2_calibration, lmax_calibration = validate_gradient_descent(val_loader, output_layer, random_features_model, args, nonlinearity, criterion, device)
        record.metrics.test_mse[epoch] = val_losses
        record.metrics.test_top1[epoch] = val_top1
        record.metrics.test_top5[epoch] = val_top5
        record.metrics.l1_calibration[epoch] = l1_calibration
        record.metrics.l2_calibration[epoch] = l2_calibration
        record.metrics.lmax_calibration[epoch] = lmax_calibration
        print("val_losses, val_top1, val_top5", val_losses, val_top1, val_top5)
    return losses.avg, output_layer, criterion

def validate_gradient_descent(val_loader, output_layer, random_features_model, args, nonlinearity, criterion, device):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    
    output_layer.eval()
    with torch.no_grad():
        end = time.time()
        outputs_for_calibration = []
        targets_for_calibration = []
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True).view(images.shape[0], -1) # flatten
            target = target.to(device, non_blocking=True)
            sqrtD = float(math.sqrt(images.shape[1])) # sqrt of dimension
             
            random_features = nonlinearity(torch.matmul(images, random_features_model) / sqrtD )
            output = output_layer(random_features)
            # save outputs and targets for calibration
            outputs_for_calibration.append(output)
            targets_for_calibration.append(target)
            loss = criterion(output, target)
            #print(loss.item())
            losses.update(loss.item(), images.size(0))
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # if i > 10: break # only do 100 batches for debug
             
    outputs_for_calibration = torch.cat(outputs_for_calibration, dim=0)
    targets_for_calibration = torch.cat(targets_for_calibration, dim=0)
    l1_calibration = multiclass_calibration_error(outputs_for_calibration, targets_for_calibration, num_classes=10, n_bins=15, norm='l1')
    l2_calibration = multiclass_calibration_error(outputs_for_calibration, targets_for_calibration, num_classes=10, n_bins=15, norm='l2')
    lmax_calibration = multiclass_calibration_error(outputs_for_calibration, targets_for_calibration, num_classes=10, n_bins=15, norm='max')
    print("l1_calibration", l1_calibration, "l2_calibration", l2_calibration, "lmax_calibration", lmax_calibration)
    return losses.avg, top1.avg, top5.avg, l1_calibration, l2_calibration, lmax_calibration

def main():
    args = parser.parse_args()
    print("Args", args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 

    random_features_model = get_model(args).to(device)
    train_loader, val_loader = get_dataset(args)
    
    record = get_record(args)
    nonlinearity = get_nonlinearity(args)
    # Save parameters
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}" )
    

    
    record.metrics.train_mse = {}
    record.metrics.distance_to_true = {}
    train_mse, output_layer, criterion = train_gradient_descent(train_loader, val_loader, device, random_features_model, nonlinearity, args, record)
    
    val_losses, val_top1, val_top5, l1_calibration, l2_calibration, lmax_calibration  = validate_gradient_descent(val_loader, output_layer, random_features_model, args, nonlinearity, criterion, device)
    record.metrics.test_mse[args.epochs+1] = val_losses
    record.metrics.test_top1[args.epochs+1] = val_top1
    record.metrics.test_top5[args.epochs+1] = val_top5
    record.metrics.l1_calibration[args.epochs+1] = l1_calibration
    record.metrics.l2_calibration[args.epochs+1] = l2_calibration
    record.metrics.lmax_calibration[args.epochs+1] = lmax_calibration
    print("val_losses, val_top1, val_top5", val_losses, val_top1, val_top5)
     
    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)
    #utils.save_checkpoint(record_weights, save_dir = args.save_dir, filename = f"weights_{args.exp_name}")

if __name__ == '__main__':
    main()
