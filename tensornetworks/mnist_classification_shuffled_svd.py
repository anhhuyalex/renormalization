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
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type = str, default = "mnist",
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
parser.add_argument('--optimizer_type', default="sgd", type=str,
                    help='optimizer type')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
 
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--lr_scheduler', default="linear", type=str,  
                    help='learning rate scheduler')
parser.add_argument('--num_train_samples', default=1, type=int,  
                    help='fraction of training set to use')
parser.add_argument('--num_hidden_features', default=0, type=int,  
                    help='number of hidden fourier features') 
parser.add_argument('--multiclass_lr', action='store_true', 
                    help='multi-class logistic regression')
parser.add_argument('--nonlinearity', default="tanh", type=str,  
                    help='type of nonlinearity used')
parser.add_argument('--randomfeatures', action='store_true', default=False,
                        help='whether to do random features')
parser.add_argument('--is_high_signal_to_noise', default="False", type=str,
                    help='whether to keep high signal to noise ratio')
parser.add_argument('--is_shuffle_signal', default="True", type=str,
                    help='whether to shuffle signal, if false, set signal to zero')
parser.add_argument('--highsignal_pca_components_kept', 
                    default=1.0, type=float,
                    help='fraction of pca components to keep')
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
 

def get_model(args, nonlinearity):
     
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
        # random features
        # make first layer untrainable 
        if args.randomfeatures == True:
            model[1].requires_grad_(False)

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
   
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    if args.data == "mnist":
        train_dataset = datasets.MNIST(root = "./data",
                                        train = True,
                                        transform = transform,
                                        download = True
                                        )
        

        val_dataset = datasets.MNIST(root = "./data",
                                            train = False,
                                            transform = transform,
                                            download = True
                                            )
    elif args.data == "fashionmnist":
        train_dataset = datasets.FashionMNIST(root = "./data",
                                        train = True,
                                        transform = transform,
                                        download = True
                                        )
        

        val_dataset = datasets.FashionMNIST(root = "./data",
                                            train = False,
                                            transform = transform,
                                            download = True
                                            )
    else:
        raise ValueError(f"Invalid dataset: got {args.data}")
    
    # get X, y
    X, y = zip(*[(dat,targ) for dat, targ in train_dataset]) # get all the training samples
    print ("X", len(X))
    X = np.stack([x.flatten() for x in X], axis=0)
    y = np.concatenate( [np.expand_dims(yi, axis=0) for yi in y], axis=0)
    print ("X", X.shape, "y", y.shape, type (X), type(y))
    pca = PCA(n_components=None)
    pca.fit(X)
    Xpca = pca.transform(X)  
    highsignal_pca_components_kept = int(args.highsignal_pca_components_kept * Xpca.shape[1]) 
    high_signal, low_signal = copy.deepcopy(Xpca[:,:highsignal_pca_components_kept]), copy.deepcopy(Xpca[:,highsignal_pca_components_kept:]) 
    if args.is_shuffle_signal == "True": # shuffling signal 
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), shuffle the high signal
            np.random.shuffle(high_signal.T)
        else: # if using high signal (i.e. good coarse graining), shuffle the low signal
            np.random.shuffle(low_signal.T) 
    elif args.is_shuffle_signal == "False": # if not shuffling signal, set signal to zero
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), set the high signal to zero
            high_signal = np.zeros_like(high_signal)
        else: # if using high signal (i.e. good coarse graining), set the low signal to zero
            low_signal = np.zeros_like(low_signal) 
    Xpca = np.hstack([high_signal, low_signal]) 
    X = pca.inverse_transform(Xpca)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
    
    Xtest, ytest = zip(*[(dat,targ) for dat, targ in val_dataset]) # get all the training samples
    Xtest = np.stack ([x.flatten() for x in Xtest], axis=0)
    ytest = np.concatenate( [np.expand_dims(yi, axis=0) for yi in ytest], axis=0)
    Xtestpca = pca.transform(Xtest)
    high_test_signal, low_test_signal = copy.deepcopy(Xtestpca[:,:highsignal_pca_components_kept]), copy.deepcopy(Xtestpca[:,highsignal_pca_components_kept:])
    if args.is_shuffle_signal == "True": # shuffling signal 
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), shuffle the high signal
            np.random.shuffle(high_test_signal.T)
        else: # if using high signal (i.e. good coarse graining), shuffle the low signal
            np.random.shuffle(low_test_signal.T)
    elif args.is_shuffle_signal == "False":
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), set the high signal to zero
            high_test_signal = np.zeros_like(high_test_signal)
        else: # if using high signal (i.e. good coarse graining), set the low signal to zero
            low_test_signal = np.zeros_like(low_test_signal)
            
    Xtestpca = np.hstack([high_test_signal, low_test_signal])
    Xtest = pca.inverse_transform(Xtestpca)
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(Xtest).float(), torch.tensor(ytest).long())

    # Get a fixed subset of the training set
    rng = np.random.default_rng()
    num_train = len(X)
    # train_idx = rng.integers(low=0, high=num_train, size=args.num_train_samples)
    train_idx = rng.permutation(num_train)[:args.num_train_samples]
    print("train_idx", train_idx[:100])
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx) 
    val_sampler = None 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, shuffle=(train_sampler is None),
        **train_kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler, shuffle=(val_sampler is None), **test_kwargs)
    
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
    model = model.float().to(device)
    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),  
                                    lr=args.lr , 
                                    weight_decay=args.weight_decay
                                        )
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(),  
                                    lr=args.lr , 
                                    weight_decay=args.weight_decay
                                    )
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    if args.lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, 
                                                  total_iters=args.epochs * len(train_loader), last_epoch=-1)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01, last_epoch=-1)
    elif args.lr_scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps = args.epochs * len(train_loader),
                                                        pct_start = 0.01, final_div_factor = 1e4, last_epoch=-1, verbose=False)
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    
    
    # args.epochs = min(int (args.epochs * (60000 / args.num_train_samples)) + 1, 1000)
    
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
            images = images.to(device, non_blocking=True)  # flatten
            target = target.to(device, non_blocking=True)
            print ("images", images.shape, "target", target.shape)
                      
            output = model(images)
            # print (model.state_dict()['3.weight'])
            # print ("target", target)
            loss = criterion(output, target)
            loss.backward()
            if i == 0: weight_norm = 0 # torch.norm(model[3].weight.grad.flatten()).cpu().detach().clone().numpy()
            optimizer.step()
            losses.update(loss.item(), images.size(0)) 
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            print("loss", loss.item(), "acc1", acc1[0], "acc5", acc5[0])#, model.state_dict()['1.weight'])
            if args.lr_scheduler == "OneCycleLR":
                scheduler.step()
            
        # step scheduler
        if args.lr_scheduler != "OneCycleLR":
            scheduler.step()

        # save metrics
        # print("output",  torch.argsortaaaaaSAaSAsaa       (output, dim=-1), "target", target )
        print("Current average loss", losses.avg, top1.avg, top5.avg, "epoch", epoch, "norm of weights") 
        record.metrics.train_mse[epoch] = losses.avg
        record.metrics.train_top1[epoch] = top1.avg
        record.metrics.train_top5[epoch] = top5.avg
        record.metrics.weight_norm[epoch] = weight_norm
        

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
    # save model
    # record.model = model.state_dict()
    print("val_losses, val_top1, val_top5", val_losses, val_top1, val_top5)
   
    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)

if __name__ == '__main__':
    main()
