# ResNet-18 ImageNet Categorization training
import argparse
import copy
import pdb
import os
import os.path as osp
import sys
import json
import numpy as np
import logging
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

from openselfsup.framework.epoch_based_runner import EpochBasedRunner
from openselfsup.framework.hooks.lr_updater import StepLrUpdaterHook
from openselfsup.framework.dist_utils import get_dist_info, init_dist
from openselfsup.framework.utils import mkdir_or_exist, get_root_logger
from openselfsup.framework.hooks.record_saver import MongoDBSaver
import torchvision.models as models


IMAGENET_FOLDER = os.environ.get(
        'IMAGENET_FOLDER',
        '/data5/chengxuz/Dataset/imagenet_raw/')
MODEL_SAVE_FOLDER = os.environ.get(
        'MODEL_SAVE_FOLDER',
        '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines')


def get_parser():
    parser = argparse.ArgumentParser(
            description='Pytorch training framework test')
    parser.add_argument(
            '--gpus', default='0', type=str, action='store')
    parser.add_argument(
            '--identifier', 
            default=None, type=str, 
            action='store', required=True)
    parser.add_argument(
            '--batch_size', type=int, 
            default=256,
            action='store')
    parser.add_argument(
            '--workers', type=int, 
            default=30,
            action='store')
    parser.add_argument(
            '--save_rec_to_file', 
            action='store_true')
    parser.add_argument(
            '--dist', 
            action='store', type=int, default=0,
            help='Number of processes in distributed training, 0 for no dist')
    parser.add_argument(
            '--local_rank', type=int, default=0,
            help='Used during distributed training')
    return parser


class ParamsBuilder(object):
    def __init__(self, args):
        self.args = args
        self.params = {'max_epochs': 100}

    def get_save_params(self):
        work_dir = os.path.join(MODEL_SAVE_FOLDER, self.args.identifier)
        save_params = {
                'ckpt_hook_kwargs': {
                    'interval': 10,
                    'out_dir': work_dir,
                    'cache_interval': 1,
                    },
                }
        if self.args.save_rec_to_file:
            save_params['record_saver_kwargs'] = {
                    'out_dir': work_dir}
        else:
            save_params['record_saver_kwargs'] = {
                    'port': 26001,
                    'database_name': 'ptutils',
                    'collection_name': 'debug',
                    'exp_id': self.args.identifier,
                    }
            save_params['record_saver_builder'] = MongoDBSaver
        self.params['save_params'] = save_params

        rank, _ = get_dist_info()
        if rank == 0:
            mkdir_or_exist(work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(work_dir, 'train_{}.log'.format(timestamp))
        logger = get_root_logger(log_file)
        self.params['logger'] = logger

    def build_train_dataset(self):
        # data loading code
        traindir = os.path.join(IMAGENET_FOLDER, 'train')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformations_train = [transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize]

        train_dataset = datasets.ImageFolder(
                traindir,
                transform=transforms.Compose(transformations_train))
        return train_dataset

    def get_train_data_params(self):
        train_data_params = {
                'dataset_builder': self.build_train_dataset,
                'shuffle': True,
                }
        if self.args.dist == 0:
            train_data_params.update({
                'batch_size': self.args.batch_size,
                'num_workers': self.args.workers,
                })
        else:
            assert self.args.batch_size % self.args.dist == 0
            assert self.args.workers % self.args.dist == 0
            train_data_params.update({
                'batch_size': self.args.batch_size // self.args.dist,
                'num_workers': self.args.workers // self.args.dist,
                'distributed': True,
                })
        self.params['train_data_params'] = train_data_params

    def build_model_optimizer(self):
         
        if self.args.dist == 0:
            self.model = torch.nn.DataParallel(models.resnet18()).cuda()
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(
                    models.resnet18().cuda(),
                    device_ids=[torch.cuda.current_device()])
        self.optimizer = torch.optim.SGD(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                0.1, momentum=0.9, weight_decay=1e-4)
        return self.model, self.optimizer

    def get_model_optimizer_params(self):
        model_optimizer_params = {
                'builder': self.build_model_optimizer}
        self.params['model_optimizer_params'] = model_optimizer_params

    def get_loss_params(self):
        loss_params = {}
        def loss_builder():
            _criterion = nn.CrossEntropyLoss().cuda()
            def loss_func(data_batch, model_outputs):
                _, labels = data_batch
                labels = torch.autograd.Variable(labels.cuda())
                loss_value = _criterion(model_outputs, labels)
                return loss_value
            return loss_func
        loss_params['builder'] = loss_builder
        self.params['loss_params'] = loss_params

    def get_learning_rate_params(self):
        learning_rate_params = {
                'builder': StepLrUpdaterHook,
                'builder_kwargs': {
                    'step': [30, 60, 90]},
                }
        self.params['learning_rate_params'] = learning_rate_params

    def run_model_get_loss(self, model, loss_func, data_batch):
        img, labels = data_batch
        img = torch.autograd.Variable(img.cuda())
        model_outputs = model(img)
        loss_value = loss_func(data_batch, model_outputs)
        iter_outputs = {'loss': loss_value}
        return iter_outputs

    def get_batch_processor_params(self):
        batch_processor_params = {
                'func': self.run_model_get_loss
                }
        self.params['batch_processor_params'] = batch_processor_params

    def build_eval_data_loader(self):
        valdir = os.path.join(IMAGENET_FOLDER, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformations_val = [transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize]
        val_dataset = datasets.ImageFolder(
            valdir,
            transform=transforms.Compose(transformations_val))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=128,
                                                 shuffle=False,
                                                 num_workers=10)
        return val_loader

    def run_model_get_acc(self, model, data_batch):
        img, labels = data_batch
        img = torch.autograd.Variable(img.cuda())
        model_outputs = model(img)

        topk = (1, 5)
        maxk = max(topk)
        target = labels.cuda()
        batch_size = target.size(0)
        _, pred = model_outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res['top'+str(k)] = correct_k.mul_(100.0 / batch_size)
        return res

    def get_validation_params(self):
        topn_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model_get_acc,
                'agg_func': lambda results: {
                    key: np.mean(value) 
                    for key, value in results.items()},
                }
        validation_params = {'topn': topn_params}
        self.params['validation_params'] = validation_params

    def build_params(self):
        self.get_save_params()
        self.get_train_data_params()
        self.get_model_optimizer_params()
        self.get_loss_params()
        self.get_learning_rate_params()
        self.get_batch_processor_params()
        self.get_validation_params()
        return self.params


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.dist == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.dist > 0:
        init_dist('pytorch')

    params_builder = ParamsBuilder(args)
    params = params_builder.build_params()

    runner = EpochBasedRunner(**params)
    runner.train()


if __name__ == '__main__':
    main()
