import argparse
import copy
import torch
import pdb
import torch.backends.cudnn as cudnn
import os
import sys
import json
import numpy as np
import pickle
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import time
from torch.nn.parallel import parallel_apply

IMAGENET_FOLDER = os.environ.get(
        'IMAGENET_FOLDER',
        '/data5/chengxuz/Dataset/imagenet_raw/')
MODEL_SAVE_FOLDER = os.environ.get(
        'MODEL_SAVE_FOLDER',
        '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines')
CFG_PATH = './configs/benchmarks/linear_classification/imagenet/r18_moco.py'


def get_pt_imagenet_transfer_parser():
    parser = argparse.ArgumentParser(
            description='ImageNet transfer for pytorch models')
    parser.add_argument(
            '--gpus', default='0', type=str, action='store')
    parser.add_argument(
            '--ckpts', 
            default=None, type=str, 
            action='store', required=True)
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
            '--val_workers', type=int, 
            default=10,
            action='store')
    parser.add_argument(
            '--init_lr', type=float, 
            default=0.01,
            action='store')
    return parser

        
class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features for resnet"""
    def __init__(self, num_labels=1000):
        super(RegLog, self).__init__()
        self.linear = nn.Linear(512 * 7 * 7, num_labels)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay_boundary(optimizer, epoch, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 
        if epoch >= 60:
            lr *= 0.1
        if epoch >= 80:
            lr *= 0.1
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ImageNetTransfer(object):
    def __init__(self, args):
        self.args = args

    def build_model(self):
        from mmcv import Config
        from openselfsup.models import build_model
        config_path = CFG_PATH
        model_path = self.args.ckpts
        cfg = Config.fromfile(config_path)
        cfg.model.pretrained = model_path
        model = build_model(cfg.model)

        self._model = model.backbone
        self._model = self._model.cuda()

        self._criterion = nn.CrossEntropyLoss().cuda()
        self._linear_pred = RegLog().cuda()

        self._optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self._linear_pred.parameters()),
            self.args.init_lr,
            momentum=0.9,
            weight_decay=0)

    def build_data_provider(self):
        args = self.args
        # data loading code
        traindir = os.path.join(IMAGENET_FOLDER, 'train')
        valdir = os.path.join(IMAGENET_FOLDER, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformations_val = [transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize]
        transformations_train = [transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize]

        train_dataset = datasets.ImageFolder(
            traindir,
            transform=transforms.Compose(transformations_train))

        val_dataset = datasets.ImageFolder(
            valdir,
            transform=transforms.Compose(transformations_val))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.val_workers)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def save_model(self):
        path = os.path.join(
            self.save_folder,
            'checkpoints',
            'checkpoint_' + str(self._epoch) + '.pth.tar',
        )
        torch.save({
            'epoch': self._epoch,
            'state_dict': self._linear_pred.state_dict(),
            'optimizer' : self._optimizer.state_dict()
        }, path)

    def validate(self):
        args = self.args
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self._model.eval()
        softmax = nn.Softmax(dim=1).cuda()
        end = time.time()
        for i, (input_tensor, target) in enumerate(self.val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(
                    input_tensor.cuda(), volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            output = self._linear_pred(self._model(input_var)[0])

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], input_tensor.size(0))
            top5.update(prec5[0], input_tensor.size(0))
            loss = self._criterion(output, target_var)
            losses.update(loss.item(), input_tensor.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(i, len(self.val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
        self.save_model()
        return top1.avg, top5.avg, losses.avg

    def train(self):
        args = self.args
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # freeze also batch norm layers
        self._model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            #adjust learning rate
            lr_decay_boundary(
                    self._optimizer, 
                    self._epoch, 
                    args.init_lr)

            target = target.cuda()
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)
            # compute output

            output = self._model(input_var)
            output = self._linear_pred(output[0])
            loss = self._criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(
                          self._epoch, i, len(self.train_loader), 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=losses, top1=top1, top5=top5))

    def run(self):
        save_folder = os.path.join(MODEL_SAVE_FOLDER, self.args.identifier)
        self.save_folder = save_folder
        os.system('mkdir -p ' + save_folder)
        os.system('mkdir -p ' + os.path.join(save_folder, 'checkpoints'))

        self.build_model()
        self.build_data_provider()

        loss_log = Logger(os.path.join(save_folder, 'loss_log'))
        prec1_log = Logger(os.path.join(save_folder, 'prec1'))
        prec5_log = Logger(os.path.join(save_folder, 'prec5'))

        num_epochs = 100
        for self._epoch in range(num_epochs):
            # train for one epoch
            self.train()

            # evaluate on validation set
            prec1, prec5, loss = self.validate()

            loss_log.log(loss)
            prec1_log.log(prec1)
            prec5_log.log(prec5)


class MultiImageNetTransfer(ImageNetTransfer):
    def build_model(self):
        from mmcv import Config
        from openselfsup.models import build_model

        self.all_models = []
        self.all_criterions = []
        self.all_linear_preds = []
        self.all_optimizers = []

        for idx, _ckpt in enumerate(self.args.ckpts.split(',')):
            config_path = CFG_PATH
            cfg = Config.fromfile(config_path)
            model_path = _ckpt
            cfg.model.pretrained = model_path
            model = build_model(cfg.model)

            _model = model.backbone
            _model = _model.cuda(idx)

            _criterion = nn.CrossEntropyLoss().cuda(idx)
            _linear_pred = RegLog().cuda(idx)

            _optimizer = torch.optim.SGD(
                filter(lambda x: x.requires_grad, _linear_pred.parameters()),
                self.args.init_lr,
                momentum=0.9,
                weight_decay=0)

            self.all_models.append(_model)
            self.all_criterions.append(_criterion)
            self.all_linear_preds.append(_linear_pred)
            self.all_optimizers.append(_optimizer)

    def validate(self):
        pass

    def train(self):
        args = self.args
        batch_time = AverageMeter()
        data_time = AverageMeter()

        num_models = len(self.all_models)
        losses = [AverageMeter() for _ in range(num_models)]
        top1 = [AverageMeter() for _ in range(num_models)]
        top5 = [AverageMeter() for _ in range(num_models)]

        # freeze also batch norm layers
        for _model in self.all_models:
            _model.eval()

        end = time.time()
        for i, (input_img, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            all_target_vars = []
            all_input_vars = []
            for model_idx in range(num_models):
                #adjust learning rate
                lr_decay_boundary(
                        self.all_optimizers[model_idx], 
                        self._epoch, 
                        args.init_lr)

                target_var = target.cuda(model_idx)
                input_var = torch.autograd.Variable(input_img.cuda(model_idx))
                target_var = torch.autograd.Variable(target_var)
                all_input_vars.append(input_var)
                all_target_vars.append(target_var)
                # compute output

            all_outputs = parallel_apply(self.all_models, all_input_vars)
            all_outputs = parallel_apply(self.all_linear_preds, [output[0] for output in all_outputs])
            all_losses = parallel_apply(self.all_criterions, list(zip(all_outputs, all_target_vars)))

            # compute gradient and do SGD step
            parallel_apply(
                    [lambda _optimizer: _optimizer.zero_grad() \
                     for model_idx in range(num_models)],
                    self.all_optimizers)
            parallel_apply(
                    [lambda _loss: _loss.backward() \
                     for model_idx in range(num_models)],
                    all_losses)
            parallel_apply(
                    [lambda _optimizer: _optimizer.step() \
                     for model_idx in range(num_models)],
                    self.all_optimizers)

            for model_idx in range(num_models):
                # measure accuracy and record loss
                prec1, prec5 = accuracy(
                        all_outputs[model_idx].data, all_target_vars[model_idx], topk=(1, 5))
                losses[model_idx].update(all_losses[model_idx].item(), input_img.size(0))
                top1[model_idx].update(prec1[0], input_img.size(0))
                top5[model_idx].update(prec5[0], input_img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                for model_idx in range(num_models):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                          .format(
                              self._epoch, i, len(self.train_loader), 
                              batch_time=batch_time,
                              data_time=data_time, 
                              loss=losses[model_idx], top1=top1[model_idx], top5=top5[model_idx]))

    def run(self):
        self.build_data_provider()
        self.build_model()

        num_epochs = 100
        for self._epoch in range(num_epochs):
            # train for one epoch
            self.train()


def main():
    parser = get_pt_imagenet_transfer_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if not ',' in args.ckpts:
        transfer_cls = ImageNetTransfer(args)
        transfer_cls.run()
    else:
        num_gpus = len(args.gpus.split(','))
        num_models = len(args.ckpts.split(','))
        num_ids = len(args.identifier.split(','))
        assert num_gpus == num_models
        assert num_gpus == num_ids

        multi_transfer_cls = MultiImageNetTransfer(args)
        multi_transfer_cls.run()


if __name__ == '__main__':
    main()
