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
import copy

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
            #default="/gpfs/milgram/scratch60/turk-browne/an633/renorm",
            default="/scratch/gpfs/qanguyen/imagenet_info",
            type=str, 
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
            '--data_rescale', 
            default=1.0, type=float, 
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
            default=90, type=int, 
            action='store')
    parser.add_argument(
            '--tags', 
            default="debug", type=str, 
            action='store')
    parser.add_argument(
            '--resume', 
            default=None, type=str, 
            action='store')
    
    parser.add_argument('-d', '--debug', help="in debug mode or not", 
                        action='store_true')

    return parser

def get_imagenet_configs(args):
    # Get model optim params
    
        
    #criterion = "mse_loss_l1_first_layer"
    model_optim_params = utils.dotdict(model_name = args.model_name, 
                                       #criterion = criterion, 
                                       lr = args.lr, 
                                       weight_decay = args.weight_decay,
                                       momentum = args.momentum,
                                       resume = args.resume
                                       )
    
    # Get data params
    data_params = utils.dotdict(
        data_dir = args.data_dir,
        data_rescale = args.data_rescale
    )

    # Get train parameters
    train_params = utils.dotdict(
                        batch_size = 256, 
                        num_train_epochs = args.num_train_epochs,
                        num_workers = 2,
                        seed = None,
                        gpu = None,
                        multiprocessing_distributed = True,
                        world_size = 1,
                        rank = 0,
                        dist_backend = "nccl",
                        dist_url = f'tcp://127.0.0.1:{utils.find_free_port()}',
                        )

    # Get save params
    save_params = utils.dotdict(save_dir = args.save_dir)
    
    cfg = dict(
        model_optim_params = model_optim_params,
        data_params = data_params,
        train_params = train_params,
        save_params = save_params
    ) 
    return cfg

class RescaleImagenet(torchvision.datasets.ImageFolder):
    def __init__(self, root = "./data", data_rescale = 1.0, 
                 target_size = 224,
                 phase = "train",
                 normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])):
        
        assert phase in ["train", "test"], f"phase must be either 'train' or 'test', got {phase} instead"
        self.target_size = target_size
        self.actual_size = int(target_size*data_rescale)
        if phase == "train":
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.actual_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.actual_size),
                transforms.ToTensor(),
                normalize,
            ])
        
        super(RescaleImagenet, self).__init__(root, transform)
        
    def __getitem__(self, index: int):
        sample, target = super(RescaleImagenet, self).__getitem__(index)
        
        tiles = self.target_size // self.actual_size        
        # remainder part
        rem_idx = self.target_size % self.actual_size
        remainder = sample[:, :, :rem_idx]
        
        
        # tile image horizontally
        sample = torch.tile(sample, (1, tiles, tiles))
        
        # fill in horizontal pieces
        sample = torch.cat((sample, torch.tile(remainder, (1, tiles, 1))), dim=2)
        
        # fill in vertical pieces
        remainder = sample[:, :rem_idx, :]
        sample = torch.cat((sample, remainder), dim=1)
        
        return sample, target
class ImagenetTrainer(utils.BaseTrainer):
    def initialize_record(self):
        if self.model_optim_params.resume:
            ckpt_path = f"{self.save_params.save_dir}/{self.model_optim_params.resume}"
            if os.path.isfile(ckpt_path):
                self.record = utils.load_file_pickle(ckpt_path)
            assert self.record.success == False, "success returns True, run has already completed !"
        else:
            self.record = utils.dotdict(
                current_epoch = -1,
                metrics = utils.dotdict(
                    train_losses = utils.dotdict(default=[]),
                    train_top1 = utils.dotdict(default=[]),
                    train_top5 = utils.dotdict(default=[]),
                    test_losses = utils.dotdict(default=[]),
                    test_top1 = utils.dotdict(default=[]),
                    test_top5 = utils.dotdict(default=[]),
                ),
                success = False,
                model_state = utils.dotdict(
                    epoch = None,
                    curr_model_state = None,
                    curr_optimizer_state = None,
                    curr_scheduler_state = None,
                    best_model_state = None,
                    best_optimizer_state = None,
                    best_scheduler_state = None,
                    best_model_acc = -1e10
                ),
                data_params = self.data_params,
                train_params = self.train_params,
                model_optim_params = self.model_optim_params,
                save_params = self.save_params,
            )
        print("data params",  self.data_params)
        print("train params",  self.train_params)
        print("model params",  self.model_optim_params)
        print("save params",  self.save_params)
            
    def setup(self):
        #self.build_data_loader()
        #self.build_model_optimizer()
        self.initialize_record()
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
            self.train_params.ngpus_per_node = torch.cuda.device_count()
            print(f"Training on {self.train_params.ngpus_per_node} GPUs")
        else:
            self.train_params.ngpus_per_node = 1
#         if self.train_params.multiprocessing_distributed:
#             # Since we have ngpus_per_node processes per node, the total world_size
#             # needs to be adjusted accordingly
#             self.train_params.world_size = self.train_params.ngpus_per_node * self.train_params.world_size
#             # Use torch.multiprocessing.spawn to launch distributed processes: the
#             # main_worker process function
#             mp.spawn(self.setup_worker, nprocs=self.train_params.ngpus_per_node, args=(self.train_params.ngpus_per_node,))
#             print(self.train_sampler)
#             print(wa)
#         else:
#             # Simply call main_worker function
#             self.setup_worker(args.gpu, self.train_params.ngpus_per_node, args)
        
        
#     def setup_worker(self, gpu, ngpus_per_node):
#         if self.train_params.multiprocessing_distributed:
#             # For multiprocessing distributed training, rank needs to be the
#             # global rank among all the processes
#             rank = self.train_params.rank * ngpus_per_node + gpu
#             print("My rank is", rank)
#         torch.distributed.init_process_group(backend=self.train_params.dist_backend, init_method=self.train_params.dist_url,
#                                 world_size=self.train_params.world_size, rank=rank)

#         self.build_model_optimizer(rank)
#         self.build_data_loader()
        
        
    def build_model_optimizer(self, rank):
        print("setting model")
        self.model = torchvision.models.__dict__[self.model_optim_params.model_name]()
        if torch.cuda.is_available():
            self.model.to(rank)
#             self.model.cuda()
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
        
        # Resume model from checkpoint if directed
        if self.model_optim_params.resume:
            ckpt_path = f"{self.save_params.save_dir}/{self.model_optim_params.resume}"
            print(ckpt_path, "Record should have already been loaded")
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(ckpt_path))
                #if args.gpu is None:
                #    checkpoint = torch.load(args.resume)
                #elif torch.cuda.is_available():
                #    # Map model to be loaded to specified single gpu.
                #   loc = 'cuda:{}'.format(args.gpu)
                #    checkpoint = torch.load(args.resume, map_location=loc)
                
                # start_epoch is final epoch in checkpoint + 1
                self.start_epoch = max([int(i) for i in self.record.metrics.test_top1.keys() if i != 'default']) + 1
                
                self.model.load_state_dict(self.record.model_state.curr_model_state)
                self.optimizer.load_state_dict(self.record.model_state.curr_optimizer_state)
                self.scheduler = self.record.model_state.curr_scheduler_state
                print("=> loaded checkpoint '{}', starting at epoch {})"
                      .format(ckpt_path, self.start_epoch))
            else:
                print("=> no checkpoint found at '{}'".format(ckpt_path))
        else:
            self.start_epoch = 0
                
    def build_data_loader(self):
        traindir = os.path.join(self.data_params.data_dir, 'train')
        valdir = os.path.join(self.data_params.data_dir, 'val')
        

        train_dataset = RescaleImagenet(
            root = traindir, data_rescale = self.data_params.data_rescale, phase = "train"
            )

        val_dataset = RescaleImagenet(
            root = valdir, data_rescale = self.data_params.data_rescale, phase = "test")
        print("loading train_sampler and val_sampler")
        
        if self.train_params.multiprocessing_distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            self.train_sampler = None
            self.val_sampler = None
        
        print("loading train_loader and val_loader")
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_params.batch_size, shuffle=(self.train_sampler is None),
            num_workers=self.train_params.num_workers, pin_memory=True, sampler=self.train_sampler)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.train_params.batch_size, shuffle=False,
            num_workers=self.train_params.num_workers, pin_memory=True, sampler=self.val_sampler)
        print("finished train_loader and val_loader")
        
    def train(self):
        mp.spawn(self.train_worker, nprocs=self.train_params.ngpus_per_node, args=(self.train_params.ngpus_per_node,))
        
    def after_iter(self, output, target, loss, epoch, train_batch_id):
        if train_batch_id % 10 == 1:
            print("Epoch", epoch, "Loss", loss, flush=True)
        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        self.record.metrics.train_losses[epoch].append(loss.item())
        self.record.metrics.train_top1[epoch].append(acc1.item())
        self.record.metrics.train_top5[epoch].append(acc5.item())
        
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def after_epoch(self, epoch):
        # compute train metric avgs
        self.record.metrics.train_losses[epoch] = np.mean(self.record.metrics.train_losses[epoch])
        self.record.metrics.train_top1[epoch] = np.mean(self.record.metrics.train_top1[epoch])
        self.record.metrics.train_top5[epoch] = np.mean(self.record.metrics.train_top5[epoch])
        
        # evaluate on validation set
        val_losses, val_top1, val_top5 = 0.0, 0.0, 0.0
        
        self.model.eval() # switch to evaluate mode
        with torch.no_grad():
            for i, (images, target) in enumerate(self.val_loader):
                # move data to the same device as model
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                

                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)
                
                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                self.record.metrics.test_losses[epoch].append(loss.item())
                self.record.metrics.test_top1[epoch].append(acc1.item())
                self.record.metrics.test_top5[epoch].append(acc5.item())

                #print("val", i)
                #if i > 5:
                #    break
                
        # compute test metric avgs
        #print("self.record.metrics.test_losses.get(epoch, [])", self.record.metrics.test_losses.get(epoch, []))
        #print("self.record.metrics.test_losses.get(epoch, [])", self.record.metrics.test_losses[].get(epoch, []))
        self.record.metrics.test_losses[epoch] = np.mean(self.record.metrics.test_losses[epoch])
        self.record.metrics.test_top1[epoch] = np.mean(self.record.metrics.test_top1[epoch])
        self.record.metrics.test_top5[epoch] = np.mean(self.record.metrics.test_top5[epoch])
        print("Epoch", epoch, "Average train loss", self.record.metrics.train_losses[epoch], "Top 1 train acc", self.record.metrics.train_top1[epoch], "Top 5 train acc", self.record.metrics.train_top5[epoch])
        print("Epoch", epoch, "Average val loss", self.record.metrics.test_losses[epoch], "Top 1 val acc", self.record.metrics.test_top1[epoch], "Top 5 val acc", self.record.metrics.test_top5[epoch])
        
        # step scheduler
        self.scheduler.step()
        
        # save state_dicts
        self.record.model_state.curr_model_state = copy.deepcopy(self.model.state_dict())
        self.record.model_state.curr_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.record.model_state.curr_scheduler_state = copy.deepcopy(self.scheduler)
        
        if self.record.model_state.best_model_acc < self.record.metrics.test_top1[epoch]:
            self.record.model_state.best_model_acc = self.record.metrics.test_top1[epoch]
            self.record.model_state.best_model_state = copy.deepcopy(self.model.state_dict())
            self.record.model_state.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            self.record.model_state.best_scheduler_state = copy.deepcopy(self.scheduler)
        
        # save run
        self.record.current_epoch = epoch
        self.save_record()
        
    def train_worker(self, gpu, ngpus_per_node):
        if self.train_params.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            rank = self.train_params.rank * ngpus_per_node + gpu
            print("My train rank is", rank)
            
        torch.distributed.init_process_group(backend=self.train_params.dist_backend, init_method=self.train_params.dist_url,
                              world_size=self.train_params.world_size,rank=rank)
        self.build_model_optimizer(rank)
        self.build_data_loader()
        
        for epoch in range(self.start_epoch, self.train_params.num_train_epochs):
            print("start epoch", epoch)
            if self.train_params.multiprocessing_distributed:
                self.train_sampler.set_epoch(epoch)
            print("finish train_sampler", epoch)
            # train for one epoch
            self.model.train() # switch to train mode
            print("put model in train mode epoch", epoch, torch.cuda.get_device_name(0))
            for train_batch_id, (images, target) in enumerate(self.train_loader):
                # move data to the same device as model
                
                images = images.cuda()#to(self.model.device, non_blocking=True)
                target = target.cuda()#.to(self.model.device, non_blocking=True)
                print("cuda images target")
                
                # compute output
                output = self.model(images)
                print("output", output.shape)
                loss = self.criterion(output, target)
                
                self.after_iter(output, target, loss, epoch, train_batch_id)
                #if self.stopped_training:
                #    break
                
                    
            # validation, scheduler, and other ops
            self.after_epoch(epoch)        
            
        self.after_run()
        
    def after_run(self):
        # Save record
        self.record.success = True
        self.save_record()
        print("Finished training successfully!")
        
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