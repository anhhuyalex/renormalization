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
parser.add_argument('--gaussian_shuffle_std', default=1, type=float,
                    help='std of Gaussian shuffling signal')
parser.add_argument('--is_inverse_transform', default="True", type=str,
                    help='whether to inverse transform signal for training')
parser.add_argument('--highsignal_pca_components_kept', 
                    default=1.0, type=float,
                    help='fraction of pca components to keep')
parser.add_argument('--true_noise', 
                    default=0.1, type=float,
                    help='noise of true mapping y = X @ w + noise')
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
            nn.Linear(width_after_pool**2, 1),
        ) 
    else:
        # 2 layer MLP
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width_after_pool**2, args.num_hidden_features),
            nonlinearity,
            nn.Linear(args.num_hidden_features, 1)
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
    elif args.data == "random_normalized_gaussian":
        rng = np.random.default_rng(args.seed)
        train_dataset = rng.normal(size = (1000, 1000))
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
    rng = np.random.default_rng(args.seed)
    beta_0 = rng.normal(size = X.shape[1]) / 10 # initialize true beta * (X.shape[1] ** -0.5) 
    # y = np.concatenate( [np.expand_dims(yi, axis=0) for yi in y], axis=0)
    y = X @ beta_0 + rng.normal(size = X.shape[0]) * args.true_noise # add noise
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
            # print ("low_signal", low_signal.shape)
            rng.shuffle(low_signal.T) 
            # print ("low_signal", low_signal.shape)
    elif args.is_shuffle_signal == "False": # if not shuffling signal, set signal to zero
        if args.is_high_signal_to_noise == "False": # if not using high signal (i.e. bad coarse graining), set the high signal to zero
            high_signal = np.zeros_like(high_signal)
        else: # if using high signal (i.e. good coarse graining), set the low signal to zero
            low_signal = np.zeros_like(low_signal) 
    elif args.is_shuffle_signal == "Gaussian": # if Gaussian shuffling signal, set signal to Gaussian
        rng = np.random.default_rng(args.seed * 2)
        if args.is_high_signal_to_noise == "False": 
            # high_signal = np.random.normal(0, args.gaussian_shuffle_std, high_signal.shape) 
            means_high_signal = np.mean(high_signal, axis=0) # shape: high_signal.shape[1] i.e. # of high components
            std_high_signal = np.std(high_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            high_signal = [rng.normal(means_high_signal[i], std_high_signal[i], high_signal.shape[0]) for i in range(high_signal.shape[1])]
            high_signal = np.stack(high_signal, axis=1)
        else:
            means_low_signal = np.mean(low_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            std_low_signal = np.std(low_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            low_signal = [rng.normal(means_low_signal[i], std_low_signal[i], low_signal.shape[0]) for i in range(low_signal.shape[1])]
            low_signal = np.stack(low_signal, axis=1)
            print ("low_signal", low_signal.shape)
    Xpca = np.hstack([high_signal, low_signal]) 
    if args.is_inverse_transform == "True":
        X = pca.inverse_transform(Xpca)
    else:
        X = Xpca
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
     
    Xtest, ytest = zip(*[(dat,targ) for dat, targ in val_dataset]) # get all the training samples
    Xtest = np.stack ([x.flatten() for x in Xtest], axis=0)
    # ytest = np.concatenate( [np.expand_dims(yi, axis=0) for yi in ytest], axis=0)
    rng = np.random.default_rng(args.seed * 2)
    ytest = Xtest @ beta_0 + rng.normal(size = Xtest.shape[0]) * args.true_noise # add noise
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
        rng = np.random.default_rng(args.seed * 3)
        if args.is_high_signal_to_noise == "False": 
            # high_signal = np.random.normal(0, args.gaussian_shuffle_std, high_signal.shape) 
            means_high_test_signal = np.mean(high_test_signal, axis=0) # shape: high_signal.shape[1] i.e. # of high components
            std_high_test_signal = np.std(high_test_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            high_test_signal = [rng.normal(means_high_test_signal[i], std_high_test_signal[i], high_test_signal.shape[0]) for i in range(high_test_signal.shape[1])]
            high_test_signal = np.stack(high_test_signal, axis=1)
        else:
            means_low_test_signal = np.mean(low_test_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            std_low_test_signal = np.std(low_test_signal, axis=0) # shape: low_signal.shape[1] i.e. # of low components
            low_test_signal = [rng.normal(means_low_test_signal[i], std_low_test_signal[i], low_test_signal.shape[0]) for i in range(low_test_signal.shape[1])]
            low_test_signal = np.stack(low_test_signal, axis=1)
    Xtestpca = np.hstack([high_test_signal, low_test_signal])
    if args.is_inverse_transform == "True":
        Xtest = pca.inverse_transform(Xtestpca)
    else:
        Xtest = Xtestpca 
    # save_data = {
    #     "X": X,
    #     "y": y,
    #     "Xtest": Xtest,
    #     "ytest": ytest
    # }
    # utils.save_checkpoint(save_data, save_dir = "./cache", filename = f"{args.exp_name}_data.pth.tar")

    val_dataset = torch.utils.data.TensorDataset(torch.tensor(Xtest).float(), torch.tensor(ytest).long())

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
        train_dataset, sampler=train_sampler, shuffle=False,
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

    from sklearn.linear_model import Ridge, LinearRegression
    if args.weight_decay == 0.0:
        clf = LinearRegression() 
    else:
        clf = Ridge(max_iter=1e12, 
                            alpha = args.weight_decay
                                ) 
    from sklearn.metrics import mean_squared_error

    for i, (images, target) in enumerate(train_loader):
        print ("images", images.shape, "target", target.shape)
        images = images.cpu().numpy()
        target = target.cpu().numpy()
        clf = clf.fit(images, target)
        record.metrics.train_mse[0] = mean_squared_error(target, clf.predict(images))
        record.metrics.train_top1[0] = clf.score(images, target)
         
    for i, (images, target) in enumerate(val_loader):
        images = images.cpu().numpy()
        target = target.cpu().numpy()
        record.metrics.test_mse[0] = mean_squared_error(target, clf.predict(images))
        record.metrics.test_top1[0] = clf.score(images, target) 
    print("train_mse", record.metrics.train_mse[0], "test_mse", record.metrics.test_mse[0], "train_top1", record.metrics.train_top1[0], "test_top1", record.metrics.test_top1[0])
    record["model"] = clf.coef_
    return model, criterion, record

def main():
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
 
    print("Args", args)
    # Save parameters
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}" )

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

