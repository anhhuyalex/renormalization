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
parser.add_argument('--randomfeatures_target_size', default=28, type=int,
                    help='target size for random features')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer_type', default="sgd", type=str,
                    help='optimizer type')
parser.add_argument('--is_random_init', default="False", type=str,
                        help='whether to use random initialization')
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
parser.add_argument('--is_task_binary', default="False", type=str,
                    help='whether to use binary task')
parser.add_argument('--is_high_signal_to_noise', default="False", type=str,
                    help='whether to keep high signal to noise ratio')
parser.add_argument('--is_shuffle_signal', default="True", type=str,
                    help='whether to shuffle signal, if false, set signal to zero')
parser.add_argument('--gaussian_shuffle_std', default=1, type=float,
                    help='std of Gaussian shuffling signal')
parser.add_argument('--is_inverse_transform', default="True", type=str,
                    help='whether to inverse transform signal for training')
parser.add_argument('--highsignal_pca_components_kept', 
                    default=1.0, type=float,
                    help='fraction of pca components to keep')
parser.add_argument('--save_iter', default=100, type=int,  
                    help='save every save_iter epochs') 
parser.add_argument('--seed', default=100, type=int,  
                    help='seed') 
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
            nn.Linear(args.num_hidden_features, 10 if args.is_task_binary != "True" else 2),
        )
        # random features
        # make first layer untrainable 
        if args.randomfeatures == True:
            model[1].requires_grad_(False)

    return model
    
class RandomFeaturesMNIST(datasets.MNIST):
    def __init__(self, root = "./data",  
                 train = True,
                 transform = None,
                 block_size = 1, 
                 upsample = False,
                 target_size = None,
                ):
         
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
        # if self.target_size is not None:
        sample = torch.matmul(sample.view(1, -1), self.transform_matrix)
        
        sample = sample.view(1, self.target_size, self.target_size) * self.target_size * self.target_size
        sample = torch.matmul (sample.view(1, -1), self.retransform_matrix)
        sample = sample.view(1, 28, 28) * 28 * 28 
        return sample, target
    
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
    elif args.data == "randomfeaturesmnist":
        train_dataset = RandomFeaturesMNIST(root = "./data",
                                        train = True,
                                        transform = transform,
                                        target_size = args.randomfeatures_target_size,
                                        upsample = True,
                                        download = True
                                        )
        

        val_dataset = RandomFeaturesMNIST(root = "./data",
                                            train = False,
                                            transform = transform,
                                            target_size = args.randomfeatures_target_size,
                                            upsample = True,
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
    if args.is_task_binary == "True":
        X, y = zip(*[(dat,targ) for dat, targ in train_dataset if targ == 0 or targ == 1])
    else:
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
        rng = np.random.default_rng(args.seed * 2)
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), shuffle the high signal
            rng.shuffle(high_signal.T)
        else: # if using high signal (i.e. good coarse graining), shuffle the low signal
            rng.shuffle(low_signal.T)
    elif args.is_shuffle_signal == "False": # if not shuffling signal, set signal to zero
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), set the high signal to zero
            high_signal = np.zeros_like(high_signal)
        else: # if using high signal (i.e. good coarse graining), set the low signal to zero
            low_signal = np.zeros_like(low_signal) 
    elif args.is_shuffle_signal == "Gaussian": # if Gaussian shuffling signal, set signal to Gaussian
        if args.is_high_signal_to_noise == "False": 
            # high_signal = np.random.normal(0, args.gaussian_shuffle_std, high_signal.shape) 
            means_high_signal = np.mean(high_signal, axis=0) # shape: high_signal.shape[1] i.e. # of high components
            std_high_signal = np.std(high_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            high_signal = [np.random.normal(means_high_signal[i], std_high_signal[i], high_signal.shape[0]) for i in range(high_signal.shape[1])]
            high_signal = np.stack(high_signal, axis=1)
        else:
            means_low_signal = np.mean(low_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            std_low_signal = np.std(low_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            low_signal = [np.random.normal(means_low_signal[i], std_low_signal[i], low_signal.shape[0]) for i in range(low_signal.shape[1])]
            low_signal = np.stack(low_signal, axis=1)
            print ("low_signal", low_signal.shape)
    Xpca = np.hstack([high_signal, low_signal]) 
    if args.is_inverse_transform == "True":
        X = pca.inverse_transform(Xpca)
    else:
        X = Xpca
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
    if args.is_task_binary == "True":
        Xtest, ytest = zip(*[(dat,targ) for dat, targ in val_dataset if targ == 0 or targ == 1])
    else:
        Xtest, ytest = zip(*[(dat,targ) for dat, targ in val_dataset]) # get all the training samples
    Xtest = np.stack ([x.flatten() for x in Xtest], axis=0)
    ytest = np.concatenate( [np.expand_dims(yi, axis=0) for yi in ytest], axis=0)
    Xtestpca = pca.transform(Xtest)
    high_test_signal, low_test_signal = copy.deepcopy(Xtestpca[:,:highsignal_pca_components_kept]), copy.deepcopy(Xtestpca[:,highsignal_pca_components_kept:])
    if args.is_shuffle_signal == "True": # shuffling signal 
        rng = np.random.default_rng(args.seed * 3)
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), shuffle the high signal
            rng.shuffle(high_test_signal.T)
        else: # if using high signal (i.e. good coarse graining), shuffle the low signal
            rng.shuffle(low_test_signal.T)
    elif args.is_shuffle_signal == "False":
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), set the high signal to zero
            high_test_signal = np.zeros_like(high_test_signal)
        else: # if using high signal (i.e. good coarse graining), set the low signal to zero
            low_test_signal = np.zeros_like(low_test_signal)
    elif args.is_shuffle_signal == "Gaussian": # if Gaussian shuffling signal, set signal to Gaussian
        if args.is_high_signal_to_noise == "False": 
            # high_signal = np.random.normal(0, args.gaussian_shuffle_std, high_signal.shape) 
            means_high_test_signal = np.mean(high_test_signal, axis=0) # shape: high_signal.shape[1] i.e. # of high components
            std_high_test_signal = np.std(high_test_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            high_test_signal = [np.random.normal(means_high_test_signal[i], std_high_test_signal[i], high_test_signal.shape[0]) for i in range(high_test_signal.shape[1])]
            high_test_signal = np.stack(high_test_signal, axis=1)
        else:
            means_low_test_signal = np.mean(low_test_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            std_low_test_signal = np.std(low_test_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            low_test_signal = [np.random.normal(means_low_test_signal[i], std_low_test_signal[i], low_test_signal.shape[0]) for i in range(low_test_signal.shape[1])]
            low_test_signal = np.stack(low_test_signal, axis=1)
    Xtestpca = np.hstack([high_test_signal, low_test_signal])
    if args.is_inverse_transform == "True":
        Xtest = pca.inverse_transform(Xtestpca)
    else:
        Xtest = Xtestpca
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(Xtest).float(), torch.tensor(ytest).long())
    save_data = {
        "X": X,
        "y": y,
        "Xtest": Xtest,
        "ytest": ytest
    }
    utils.save_checkpoint(save_data, save_dir = "./cache", filename = f"{args.exp_name}_data.pth.tar")

    # Get a fixed subset of the training set
    rng = np.random.default_rng(args.seed)
    num_train = len(X)
    # train_idx = rng.integers(low=0, high=num_train, size=args.num_train_samples)
    train_idx = rng.permutation(num_train)[:args.num_train_samples]
    args.num_train_samples = len(train_idx) # set the (max) number of training samples to the actual number of training samples
    print("train_idx", train_idx[:100])
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx) 
    val_sampler = None 

    if "lbfgs" in args.optimizer_type:
        train_kwargs["batch_size"] = len(train_idx)
        test_kwargs["batch_size"] = len(ytest)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, shuffle=(train_sampler is None),
        **train_kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler, shuffle=False, **test_kwargs)
    
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
        if args.save_iter > 0 and epoch % args.save_iter == 0:
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

def train_lbfgs(train_loader, val_loader, device, model, nonlinearity, args, record):
    """
    Train a linear model using L-BFGS

    :param device: the device to train on
    :param model: the model being trained
    :param nonlinearity: the nonlinearity used
    :param args: the arguments
    :param record: the record to save the metrics

    :return: the trained model
    """
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=1000, history_size=100)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    model = model.float().to(device)

    for epoch in range(args.epochs):
        model.train() # switch to train mode
        end = time.time()

        def closure():
            optimizer.zero_grad()
            # loss = 0
            for i, (images, target) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(images)
                loss = criterion(output, target)
                loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()

        # save metrics
        val_losses, val_top1, val_top5 = validate_lbfgs(val_loader, model, args, nonlinearity, criterion, device)
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        
        record.metrics.train_mse[epoch] = loss.item()
        record.metrics.train_top1[epoch] = acc1[0]
        record.metrics.train_top5[epoch] = acc5[0]


        record.metrics.test_mse[epoch] = val_losses
        record.metrics.test_top1[epoch] = val_top1
        record.metrics.test_top5[epoch] = val_top5
    print("train_mse", record.metrics.train_mse[epoch], "test_mse", record.metrics.test_mse[epoch], "test_top1", record.metrics.test_top1[epoch], "test_top5", record.metrics.test_top5[epoch])
    return model, criterion

def validate_lbfgs(val_loader, model, args, nonlinearity, criterion, device):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('xxx@x', ':6.2f')
    top5 = utils.AverageMeter('xxx@x', ':6.2f')

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return losses.avg, top1.avg, top5.avg

def train_sklearn_lbfgs(train_loader, val_loader, device, model, nonlinearity, args, record):
    """
    Train a linear model using L-BFGS using sklearn

    :param device: the device to train on
    :param model: the model being trained
    :param nonlinearity: the nonlinearity used
    :param args: the arguments
    :param record: the record to save the metrics

    :return: the trained model
    """
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1e12, 
                            maxfun=1e7,
                             C = 1/args.weight_decay if args.weight_decay > 0 else 1.0,
                             penalty="none" if args.weight_decay <= 0.0 else 'l2',
                             solver='lbfgs',verbose=1,
                             multi_class='multinomial' if args.is_task_binary != "True" else 'ovr',
                             random_init = True if args.is_random_init == "True" else False,
                             ) 
    from sklearn.metrics import log_loss

    for i, (images, target) in enumerate(train_loader):
        print ("images", images.shape, "target", target.shape)
        images = images.cpu().numpy()
        target = target.cpu().numpy()
        clf = clf.fit(images, target)
        record.metrics.train_mse[0] = log_loss(target, clf.predict_proba(images))
        record.metrics.train_top1[0] = clf.score(images, target)
         
    for i, (images, target) in enumerate(val_loader):
        images = images.cpu().numpy()
        target = target.cpu().numpy()
        record.metrics.test_mse[0] = log_loss(target, clf.predict_proba(images))
        record.metrics.test_top1[0] = clf.score(images, target) 
    print("train_mse", record.metrics.train_mse[0], "test_mse", record.metrics.test_mse[0], "train_top1", record.metrics.train_top1[0], "test_top1", record.metrics.test_top1[0])
    record["model"] = clf.coef_
    return model, criterion, record

def main():
    args = parser.parse_args()
    # Save parameters
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}" )

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
    
    
    

    
    record.metrics.train_mse = {}
    record.metrics.distance_to_true = {}

    if args.optimizer_type == "lbfgs":
        model, criterion = train_lbfgs(train_loader, val_loader, device, model, nonlinearity, args, record)
        val_losses, val_top1, val_top5 = validate_lbfgs(val_loader, model, args, nonlinearity, criterion, device)
    elif args.optimizer_type == "sklearn_lbfgs":
        model, criterion, record = train_sklearn_lbfgs(train_loader, val_loader, device, model, nonlinearity, args, record)
        utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)
        return 
    else:
        train_mse, model, criterion = train_gradient_descent(train_loader, val_loader, device, model, nonlinearity, args, record)
        val_losses, val_top1, val_top5 = validate_gradient_descent(val_loader, model, args, nonlinearity, criterion, device)

    record.metrics.test_mse[args.epochs] = val_losses
    record.metrics.test_top1[args.epochs] = val_top1
    record.metrics.test_top5[args.epochs] = val_top5
    # save model
    record.model = model.state_dict()
    print("val_losses, val_top1, val_top5", val_losses, val_top1, val_top5)
   
    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)

if __name__ == '__main__':
    main()
