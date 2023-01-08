import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import random

import importlib
import argparse
import datetime
import os
import warnings
import wandb

os.environ["WANDB_API_KEY"] = "74369089a72bb385ac20560974425f1e30fd2f94"
os.environ["WANDB_MODE"] = "offline"

import utils



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
            default="resnet18", type=str, 
            action='store')
    parser.add_argument(
            '--data_dir', 
            default="/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization", type=str, 
            action='store')
    
    parser.add_argument(
            '--random_coefs', 
            default=False, 
            type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    
    parser.add_argument(
            '--lr', 
            default=1e-1, type=float, 
            action='store')
    parser.add_argument(
            '--weight_decay', 
            default=1e-4, type=float, 
            action='store')
    parser.add_argument(
            '--momentum', 
            default=0.9, type=float, 
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

def get_imagenet_configs(args):
    # Get model optim params
    
        
    criterion = "mse_loss_l1_first_layer"
    model_optim_params = utils.dotdict(model_name = args.model_name, 
                                       criterion = criterion, 
                                       lr = args.lr, 
                                       weight_decay = args.weight_decay,
                                       momentum = args.momentum
                                       )
    
    # Get data params
    data_params = utils.dotdict()

    # Get train parameters
    train_params = utils.dotdict(
                        batch_size = 256, 
                        num_train_epochs = args.num_train_epochs,
                        seed = None,
                        gpu = None,
                        multiprocessing_distributed = True,
                        world_size = 1,
                        rank = 0,
                        dist_backend = "nccl",
                        dist_url = f'tcp://127.0.0.1:{utils.find_free_port()}',
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


class ImagenetTrainer(utils.BaseTrainer):
    
    def setup(self):
        #self.build_data_loader()
        #self.build_model_optimizer()
        #self.initialize_record()
        if self.train_params.seed is not None:
            random.seed(self.train_params.seed)
            torch.manual_seed(self.train_params.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
            
        if self.train_params.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')
            
        if torch.cuda.is_available():
            ngpus_per_node = torch.cuda.device_count()
        else:
            ngpus_per_node = 1

        if self.train_params.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.train_params.world_size = ngpus_per_node * self.train_params.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(self.setup_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))
        else:
            # Simply call main_worker function
            self.setup_worker(args.gpu, ngpus_per_node, args)

        print(wa)
        
    def setup_worker(self, gpu, ngpus_per_node):
        if self.train_params.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            self.train_params.rank = self.train_params.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=self.train_params.dist_backend, init_method=self.train_params.dist_url,
                                world_size=self.train_params.world_size, rank=self.train_params.rank)

        self.build_model_optimizer()
        self.build_data_loader()
        torch.distributed.destroy_process_group()
        
    def build_model_optimizer(self):
        self.model = torchvision.models.__dict__[self.model_optim_params.model_name]()
        if torch.cuda.is_available():
            self.model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        if torch.cuda.is_available():
            if self.train_params.gpu:
                self.device = torch.device('cuda:{}'.format(args.gpu))
            else:
                self.device = torch.device("cuda")
                
        # define loss function (criterion), optimizer, and learning rate scheduler
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.model_optim_params.lr,
                                    momentum=self.model_optim_params.momentum,
                                    weight_decay=self.model_optim_params.weight_decay)
        
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
#         if args.resume:
#             if os.path.isfile(args.resume):
#                 print("=> loading checkpoint '{}'".format(args.resume))
#                 if args.gpu is None:
#                     checkpoint = torch.load(args.resume)
#                 elif torch.cuda.is_available():
#                     # Map model to be loaded to specified single gpu.
#                     loc = 'cuda:{}'.format(args.gpu)
#                     checkpoint = torch.load(args.resume, map_location=loc)
#                 args.start_epoch = checkpoint['epoch']
#                 best_acc1 = checkpoint['best_acc1']
#                 if args.gpu is not None:
#                     # best_acc1 may be from a checkpoint from a different GPU
#                     best_acc1 = best_acc1.to(args.gpu)
#                 model.load_state_dict(checkpoint['state_dict'])
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#                 scheduler.load_state_dict(checkpoint['scheduler'])
#                 print("=> loaded checkpoint '{}' (epoch {})"
#                       .format(args.resume, checkpoint['epoch']))
#             else:
#                 print("=> no checkpoint found at '{}'".format(args.resume))
    def build_data_loader(self):
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        
        print(wa)

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
     
def get_runner(args):
    
    def runner(parameter = None):
        cfg = get_imagenet_configs(args)
        
        
        # set save fname
        exp_name = f"{args.model_name}" \
                + f"_rep_{datetime.datetime.now().timestamp()}"
        cfg["save_params"]["exp_name"] = exp_name
        print("exp_name", exp_name)
        
        trainer = ImagenetTrainer(**cfg)
        trainer.train()
        
    return runner

def main():
    parser = get_parser()
    args = parser.parse_args()
    

    runner = get_runner(args = args)
    runner()
    
if __name__ == '__main__':
    main()