
import argparse
import os
import random
import copy
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

import datetime
import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--scheduler_step_size', default=30, type=int,
                    help='scheduler_step_size (default: 30)',
                    dest='scheduler_step_size')

parser.add_argument('--image_transform_loader',   
                    default=  'TileImagenet',
                     choices=['dummy', 'RescaleImagenet', 'TileImagenet'],
                    help='type of image perturbation to run')
 
parser.add_argument('--num_models_ensemble', default=2, type=int,
                    help='number of models in ensemble (default: 2)',
                    dest='num_models_ensemble')
parser.add_argument('--voting_strategy', default="average_logits", type=str,
                    help='voting strategy for ensemble',
                    dest='voting_strategy')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument(
            '--save_dir', 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/renorm",
            default="/scratch/gpfs/qanguyen/imagenet_info",
            type=str, 
            action='store') 

class TileImagenet(datasets.ImageFolder):
    def __init__(self, root = "./data",  
                 target_size = 224,
                 tile = [1, 1],
                 tiling_orientation_ablation = False,
                 phase = "train",
                 normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                 num_augs = 1
                ):
        
        assert phase in ["train", "test"], f"phase must be either 'train' or 'test', got {phase} instead"
        self.target_size = target_size
        assert isinstance(tile, list) and len(tile) == 2, "tile must be a list of 2 numbers, default is [1, 1] to return original image"
        self.nrow, self.ncol = tile
        self.tile = tile
        self.tiling_orientation_ablation = tiling_orientation_ablation
        self.num_augs = num_augs
        if phase == "train":
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.target_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                normalize,
            ])
        
        super(TileImagenet, self).__init__(root, transform)
        
    #def __len__(self):
    #    return 256*10
    
    
    #def __getitem__(self, index: int):
    #    sample, target = super(TileImagenet, self).__getitem__(index)
    #    return  sample , target
def main():
    args = parser.parse_args()
    print("Args", args)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.dist_url = f"{args.dist_url}:{utils.find_free_port()}"
        print("Freeport", args.dist_url, flush=True)
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    # Save parameters
    args.exp_name = f"Ensemble36Epochs{args.arch}" \
                + f"_num_ensemble_{args.num_models_ensemble}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    print(f"saved to {args.save_dir}/{args.exp_name}", )
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("ngpus_per_node * args.world_size", ngpus_per_node, args.world_size)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        ensemble_kwargs = dict(
            model_func = models.__dict__[args.arch],
            num_models = args.num_models_ensemble
        )
        model = utils.models_registry["Ensemble"](**ensemble_kwargs)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print('using distributed')
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.1)
    
    # optionally resume from a record
    if args.resume:
        if os.path.isfile('{}/{}'.format(args.save_dir, args.resume)):
            print("=> loading record '{}/{}'".format(args.save_dir, args.resume))
            if args.gpu is None:
                record = torch.load('{}/{}'.format(args.save_dir, args.resume))
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                record = torch.load('{}/{}'.format(args.save_dir, args.resume), map_location=loc)
            model.load_state_dict(record.state_dict)
            optimizer.load_state_dict(record.optimizer)
            scheduler.load_state_dict(record.scheduler)
            print("=> loaded record '{}' (epoch {})"
                  .format(args.resume, record.curr_epoch))
        else:
            raise ValueError("=> no record found at '{}/{}'".format(args.save_dir, args.resume))
    else:
        # Initialize record
        record = utils.dotdict(
                curr_epoch = 0,
                args = args,
                
                best_val_acc1 = -1e10,
                
                
                
                train_params = utils.dotdict(
                    lr = args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    num_train_epochs = args.epochs
                ),
                metrics = utils.dotdict(
                    train_losses = utils.dotdict(),
                    train_acc5 = utils.dotdict(),
                    train_acc1 = utils.dotdict(),
                    val_losses = utils.dotdict(),
                    val_acc5 = utils.dotdict(),
                    val_acc1 = utils.dotdict(),
                    

                    train_submodel_losses = utils.dotdict(),
                    train_submodel_top5 = utils.dotdict(),
                    train_submodel_top1 = utils.dotdict(),
                    val_submodel_losses = utils.dotdict(),
                    val_submodel_top5 = utils.dotdict(),
                    val_submodel_top1 = utils.dotdict(),
                ),
                
        )  
        record_weights = utils.dotdict(
            state_dict = None,
            best_model = None,
            optimizer = None,
            scheduler = None,
            gradient_stats = utils.dotdict(default=[])
        )
    
    # Data loading code
    if args.image_transform_loader == "dummy":
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(10, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50, (3, 224, 224), 1000, transforms.ToTensor())
    elif args.image_transform_loader == "TileImagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_dataset = TileImagenet(
            root = traindir,  
            phase = "train", 
            tile = [1,1],
            num_augs = args.num_models_ensemble
            )
        
        val_dataset = TileImagenet(
            root = valdir, 
            phase = "test", 
            tile = [1,1],
            num_augs = args.num_models_ensemble
        )
    else:
        raise ValueError(f"{args.image_transform_loader} not supported")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    best_model = None
    
    for epoch in range(record.curr_epoch, record.train_params.num_train_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_losses, train_acc5, train_acc1, train_submodel_losses, train_submodel_top1, train_submodel_top5 = train(train_loader, model, criterion, optimizer, epoch, device, args, record, record_weights)

        # evaluate on validation set
        val_losses, val_acc5, val_acc1, val_submodel_losses, val_submodel_top1, val_submodel_top5 = validate(val_loader, model, criterion, args)
        
        scheduler.step()
        
        
        
        
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            record.curr_epoch = epoch
            
            # Record weights
            record_weights.state_dict = copy.deepcopy(model.state_dict())
            record_weights.optimizer = copy.deepcopy(optimizer.state_dict())
            record_weights.scheduler = copy.deepcopy(scheduler.state_dict())

            if val_acc1 > record.best_val_acc1:
                record_weights.best_model = copy.deepcopy(model.state_dict())
                record.best_val_acc1 = max(val_acc1, record.best_val_acc1)
                
            record.metrics.train_losses[epoch] = train_losses
            record.metrics.train_acc5[epoch] = train_acc5
            record.metrics.train_acc1[epoch] = train_acc1
            record.metrics.val_losses[epoch] = val_losses
            record.metrics.val_acc5[epoch] = val_acc5
            record.metrics.val_acc1[epoch] = val_acc1
                
            record.metrics.train_submodel_losses[epoch] = train_submodel_losses
            record.metrics.train_submodel_top5[epoch] = train_submodel_top5
            record.metrics.train_submodel_top1[epoch] = train_submodel_top1
            record.metrics.val_submodel_losses[epoch] = val_submodel_losses
            record.metrics.val_submodel_top5[epoch] = val_submodel_top5
            record.metrics.val_submodel_top1[epoch] = val_submodel_top1
                
            utils.save_checkpoint(record, save_dir = args.save_dir, filename = args.exp_name)
            utils.save_checkpoint(record_weights, save_dir = args.save_dir, filename = f'weights_{args.exp_name}')


def train(train_loader, model, criterion, optimizer, epoch, device, args, record, record_weights):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    submodel_losses = [AverageMeter('Loss', ':.4e') for _ in range(args.num_models_ensemble)]
    top1 = AverageMeter('Acc@1', ':6.2f')
    submodel_top1 = [AverageMeter('Acc@1', ':6.2f') for _ in range(args.num_models_ensemble)]
    top5 = AverageMeter('Acc@5', ':6.2f')
    submodel_top5 = [AverageMeter('Acc@5', ':6.2f') for _ in range(args.num_models_ensemble)]
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    print(device, id(losses), id(top1), id(top5))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        
        images = images.to(device, non_blocking=True)  
        target = target.to(device, non_blocking=True)

        # compute output
        output, submodel_scores = model(images)
        loss = criterion(output, target)
        
        
        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        for m in range(args.num_models_ensemble):
            submodel_losses[m].update(criterion(submodel_scores[m], target).item(), images.size(0))
            acc1, acc5 = utils.accuracy(submodel_scores[m], target, topk=(1, 5))
            submodel_top1[m].update(acc1[0], images.size(0))
            submodel_top5[m].update(acc5[0], images.size(0))
            
            
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
         
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            #for name, t in model.named_parameters():
            #    record_weights.gradient_stats[name].append(t.grad.mean())
            
    if args.distributed:
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()
        for m in range(args.num_models_ensemble):
            submodel_losses[m].all_reduce()
            submodel_top1[m].all_reduce()
            submodel_top5[m].all_reduce()
            
        
    return losses.avg, top5.avg, top1.avg, submodel_losses, submodel_top1, submodel_top5


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.to(args.gpu, non_blocking=True)  
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output, submodel_scores = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                for m in range(args.num_models_ensemble):
                    submodel_losses[m].update(criterion(submodel_scores[m], target).item(), images.size(0))
                    acc1, acc5 = utils.accuracy(submodel_scores[m], target, topk=(1, 5))
                    submodel_top1[m].update(acc1[0], images.size(0))
                    submodel_top5[m].update(acc5[0], images.size(0))


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    submodel_losses = [AverageMeter('Loss', ':.4e', Summary.NONE) for _ in range(args.num_models_ensemble)]
    submodel_top1 = [AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE) for _ in range(args.num_models_ensemble)]
    submodel_top5 = [AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE) for _ in range(args.num_models_ensemble)]
    
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return losses.avg, top5.avg, top1.avg, submodel_losses, submodel_top1, submodel_top5




class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries),flush=True)
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




if __name__ == '__main__':
    main()
