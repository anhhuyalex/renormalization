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

parser = argparse.ArgumentParser(description='Shuffled Regression experiment')
parser.add_argument('--N', default=1000, type=int, help='number of samples') 
parser.add_argument('--D_sum', default=1000, type=int, help='number of visible+ hidden features')
parser.add_argument('--D_hidden', default=5, type=int, help='number of hidden features') 
parser.add_argument('--sigma_xi', default=1.0, type=float, help='noise level')
parser.add_argument('--lambda_l2', default=0.0, type=float, help='l2 regularization')
parser.add_argument('--input_dist', default="isotropic", type=str, help='input distribution')
parser.add_argument('--shuffle_dim', default="False", type=str, help='shuffle dimensions')
parser.add_argument('--coarse_graining', default="random", type=str, help='coarse graining method')
parser.add_argument(
            '--fileprefix', default="", type=str, action='store')
parser.add_argument(
            '--save_dir', default="/scratch/gpfs/qanguyen/imagenet_info", type=str, action='store')
    
def get_dataset(args):
    # isotropic gaussian
    args.D_visible = args.D_sum - args.D_hidden
    if args.input_dist == "isotropic":
        x = np.random.randn(args.N, args.D_sum) 
        N_test = 5000
        x_test = np.random.randn(N_test, args.D_sum) 
    elif args.input_dist == "anisotropic":
        import powerlaw
        # generates random variates of power law distribution
        std = powerlaw.Power_Law(xmin=0.001, parameters=[2.0]).generate_random(args.D_sum)
        if args.shuffle_dim == "True":
            std[::-1].sort() # sort in descending order lol https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order
            x = np.random.randn(args.N, args.D_sum) * std
            high_signal, low_signal = copy.deepcopy(x[:, :args.D_visible]), copy.deepcopy(x[:, args.D_visible:])
            rng = np.random.default_rng(10)
            rng.shuffle(low_signal.T, axis=1)
            x_test = np.hstack([high_signal, low_signal])
        else:
            x = np.random.randn(args.N, args.D_sum) * std 
            N_test = 5000
            x_test = np.random.randn(N_test, args.D_sum) * std
    true_beta = np.random.randn(args.D_sum) / np.sqrt(args.D_sum)
    y = x @ true_beta + np.random.randn(args.N) * args.sigma_xi
    
    
    if args.coarse_graining == "random":
        test_beta_visible = true_beta[:args.D_visible]
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, :args.D_visible] @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
    elif args.coarse_graining == "top":
        argsort_beta = np.argsort(true_beta) 
        test_beta_visible = true_beta[argsort_beta[-args.D_visible:]] # take top D_visible 
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, argsort_beta[-args.D_visible:]] @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
    elif args.coarse_graining == "abstop":
        argsort_beta = np.argsort(np.abs(true_beta)) 
        test_beta_visible = true_beta[argsort_beta[-args.D_visible:]] # take top D_visible 
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, argsort_beta[-args.D_visible:]] @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
    elif args.coarse_graining == "absbot":
        argsort_beta = np.argsort(np.abs(true_beta)) 
        test_beta_visible = true_beta[argsort_beta[:args.D_visible]] # take bottom D_visible 
        # test_beta_visible = np.random.randn(args.D_sum) / np.sqrt(args.D_sum)
        # test_beta_visible[argsort_beta[:args.D_visible]] = true_beta[argsort_beta[:args.D_visible]] # take bottom D_visible
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, argsort_beta[:args.D_visible:]]  @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
        
    elif args.coarse_graining == "topstd":
        argsort_beta = np.argsort(true_beta) 
        test_beta_visible = np.random.randn(args.D_sum) / np.sqrt(args.D_sum)
        test_beta_visible[argsort_beta[-args.D_visible:]] = true_beta[argsort_beta[-args.D_visible:]] # take top D_visible 
        # sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test @ test_beta_visible + np.random.randn(N_test) * args.sigma_xi
    elif args.coarse_graining == "bottom":
        argsort_beta = np.argsort(true_beta) 
        test_beta_visible = true_beta[argsort_beta[:args.D_visible:]] # take bottom D_visible 
        # test_beta_visible = np.random.randn(args.D_sum) / np.sqrt(args.D_sum)
        # test_beta_visible[argsort_beta[:args.D_visible]] = true_beta[argsort_beta[:args.D_visible]] # take bottom D_visible
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, argsort_beta[:args.D_visible:]]  @ test_beta_visible + np.random.randn(N_test) * args.sigma_xi 
    
    elif args.coarse_graining == "anisotropic_topstd":
        assert args.input_dist == "anisotropic", "anisotropic input required"
        argsort_xstd = np.argsort(std) 
        test_beta_visible = true_beta[argsort_xstd[-args.D_visible:]] # take top D_visible
        # test_beta_visible[argsort_beta[-args.D_visible:]] = true_beta[argsort_beta[-args.D_visible:]] # take top D_visible 
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, argsort_xstd[-args.D_visible:]] @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
    elif args.coarse_graining == "anisotropic_botstd":
        assert args.input_dist == "anisotropic", "anisotropic input required"
        argsort_xstd = np.argsort(std) 
        test_beta_visible = true_beta[argsort_xstd[:args.D_visible:]] # take bottom D_visible
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible)) 
        y_test = x_test[:, argsort_xstd[:args.D_visible:]]  @ test_beta_visible + np.random.randn(N_test) * args.sigma_xi
    elif args.coarse_graining == "anisotropic_topprodstdbeta":
        assert args.input_dist == "anisotropic", "anisotropic input required"
        argsort_xstd = np.argsort(np.abs(std * true_beta) )
        test_beta_visible = true_beta[argsort_xstd[-args.D_visible:]] # take top D_visible
        # test_beta_visible[argsort_beta[-args.D_visible:]] = true_beta[argsort_beta[-args.D_visible:]] # take top D_visible 
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible))
        y_test = x_test[:, argsort_xstd[-args.D_visible:]] @ test_beta_visible + np.random.randn(N_test) * sigma_test_xi
    elif args.coarse_graining == "anisotropic_botprodstdbeta":
        assert args.input_dist == "anisotropic", "anisotropic input required"
        argsort_xstd = np.argsort(np.abs(std * true_beta) )
        test_beta_visible = true_beta[argsort_xstd[:args.D_visible:]] # take bottom D_visible
        sigma_test_xi = np.sqrt(args.sigma_xi ** 2 + np.dot(true_beta, true_beta) - np.dot(test_beta_visible, test_beta_visible)) 
        y_test = x_test[:, argsort_xstd[:args.D_visible:]]  @ test_beta_visible + np.random.randn(N_test) * args.sigma_xi
    return x, y, x_test, y_test, true_beta, test_beta_visible

def get_record(args):
    record = utils.dotdict(
                args = args,
                metrics = utils.dotdict(
                    train_mse = utils.dotdict(),
                    test_mse = utils.dotdict() 
                )
    )  
    return record

def fit_Ridge_estimator(x, y, x_test, y_test, args):
    covariance_matrix_with_diag = (x.T @ x  + args.lambda_l2 * np.eye(args.D_sum))
    X_top_Y = x.T @ y
    beta_hat = (np.linalg.solve(covariance_matrix_with_diag, X_top_Y))  
    y_pred = x @ beta_hat
    train_mse = np.mean((y - y_pred) ** 2)
    y_test_pred = x_test @ beta_hat
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    return train_mse, test_mse, beta_hat

def main():
    args = parser.parse_args()
     
    print("Args", args)
    # Save parameters
    args.exp_name = f"{args.fileprefix}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}" )

    x, y, x_test, y_test, true_beta, test_beta_visible = get_dataset(args)
    train_mse, test_mse, beta_hat = fit_Ridge_estimator(x, y, x_test, y_test, args)

    record = get_record(args)
    
    
    

    
    record.metrics.train_mse = train_mse
    record.metrics.test_mse = test_mse
    record.metrics.true_beta = true_beta
    record.metrics.test_beta_visible = test_beta_visible
    # record.dataset = utils.dotdict(
    #     x = x[:1000],
    #     y = y[:1000],
    #     x_test = x_test[:1000],
    #     y_test = y_test[:1000]
    # )
    record.model = beta_hat
    print (record.metrics)
   
    utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)

if __name__ == '__main__':
    main()

