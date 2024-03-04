import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import torch
import torchvision
import torchvision.models as models


import numpy as np
import pickle
import functools
import shutil

IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """default dict"""
    def __init__(self, *args, **kwargs):
        if 'default' in kwargs:
            self.default = kwargs['default']
            del kwargs['default']
        else:
            self.default = None
        dict.__init__(self, *args, **kwargs)
        
    def __repr__(self):
        return 'defaultdict(%s, %s)' % (self.default,
                                        dict.__repr__(self))

    def __missing__(self, key):
        if self.default is not None:
            return self.default
        else:
            raise KeyError(key)
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)
    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            return super(DictionaryLike, self).__getattr__(key)
        return self.__getitem__(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def try_or(expr, default=None, expected_exc=(Exception,)):
    try:
        return expr
    except expected_exc:
        return default
    
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def freeze_layer(net, *layers):
    for layer_name in layers:
        layer = rgetattr(net, layer_name)
        for param in layer.parameters():
            param.requires_grad = False


################# METRICS ########################
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
################# MODELS ########################


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation, keep_rate=1.0, N_CLASSES = 10, batch_norm = False,
                randomfeatures = False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
       
        # Output layer
        if self.hidden_sizes is not None:
            self.out = torch.nn.Linear(self.hidden_sizes[-1], N_CLASSES)
             # Set up perceptron layers and add dropout
            self.fc1 = torch.nn.Linear(self.input_size,
                                       self.hidden_sizes[0])
            
            if randomfeatures == True: 
                assert len(hidden_sizes) == 1, f"Random features require a 2-layer network, get a {len(hidden_sizes) + 1}-layer network"
                self.fc1.requires_grad_(False)
              
            self.dropout = torch.nn.Dropout(1 - keep_rate)
            self.hidden = nn.ModuleList()
            for k in range(len(self.hidden_sizes) - 1):
                self.hidden.append(nn.Linear(self.hidden_sizes[k], self.hidden_sizes[k+1]))
             
            # Batch norm
            self.batch_norm_layers = nn.ModuleList()
            self.batch_norm = batch_norm
            if batch_norm == True:
                for k in range(len(self.hidden_sizes)):
                    self.batch_norm_layers.append(nn.BatchNorm1d(self.hidden_sizes[k]))

        else:
            self.out = torch.nn.Linear(self.input_size, N_CLASSES)
     
    def forward(self, x):
        x = x.reshape(x.size(0), -1) # flatten if needed
        if self.hidden_sizes is not None:
            if self.activation == "tanh":
                x = self.fc1(x)
                Tanh = torch.nn.Tanh()
                if self.batch_norm == True:
                    x = self.batch_norm_layers[0](x)
                x = Tanh(x)
            elif self.activation == "relu":
                x = self.fc1(x)
                if self.batch_norm == True:
                    x = self.batch_norm_layers[0](x)
                x = torch.nn.functional.relu(x)
            elif self.activation == "line":
                x = self.fc1(x)
                if self.batch_norm == True:
                    x = self.batch_norm_layers[0](x)
                

            x = self.dropout(x)
            for k, lay in enumerate(self.hidden):
                x = lay(x)
                if self.batch_norm == True:
                    x = self.batch_norm_layers[k+1](x)
                if self.activation == "tanh":
                    x = Tanh(x)
                elif self.activation == "relu":
                    x = torch.nn.functional.relu(x)
                
                x = self.dropout(x)
            
        # Else just return the linear transformation
        return (self.out(x))
    
class ResNet(MLP):
    def forward(self, x):
        x = x.view(-1, self.input_size)
        if self.activation == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        for lay in self.hidden:
            if self.activation == "sigmoid":
                x = x + sigmoid(lay(x))
            elif self.activation == "relu":
                x = x + torch.nn.functional.relu(lay(x))
            x = self.dropout(x)
        return (self.out(x))
    
class CNN(nn.Module):
    def __init__(self, conv1_chans = 6, conv2_chans = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_chans, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_chans, conv2_chans, 5)
        self.fc1 = nn.Linear(conv2_chans * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, n_classes):
        """ Alexnet Classifier
        Borrowed from https://github.com/ayshrv/cs7641-img-classification/blob/master/models/alexnet.py
        Arguments:
            n_classes (int): Number of classes to score
        """
        super(AlexNet, self).__init__()

        self.alexnet = models.alexnet(pretrained=False, num_classes=n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        """
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width
        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        unnormalised_scores = self.alexnet(images)
        #logits = self.softmax(unnormalised_scores)
        return unnormalised_scores
    
class VGG(nn.Module):
    def __init__(self, model_name, n_classes):
        """ VGG11 Classifier
        Arguments:
            n_classes (int): Number of classes to score
        """
        super(VGG, self).__init__()
        if model_name == "vgg11":
            self.VGG = models.vgg11(pretrained=False, num_classes=n_classes)

    def forward(self, images):
        """
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width
        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        unnormalised_scores = self.VGG(images)
        return unnormalised_scores

class Ensemble(nn.Module):
    def __init__(self, num_models, model_func, model_kwargs = {}, voting_strategy = "average_logits"):
        
        super(Ensemble, self).__init__()
        
        self.models = nn.ModuleList([model_func(**model_kwargs) for _ in range(num_models)])
        self.voting_strategy = voting_strategy 
    def forward(self, x):
        submodel_scores = [f(x) for f in self.models]
        if self.voting_strategy == "average_logits":
            unnormalised_scores = torch.mean(torch.stack(submodel_scores), dim=0)
            return unnormalised_scores, submodel_scores
        elif self.voting_strategy == "average_softmax":
            softmaxes = F.softmax(torch.stack(submodel_scores), dim=2)
            average_softmax =  torch.mean(softmaxes, dim=0)
            return average_softmax, submodel_scores
        
    
models_registry = dict(
    MLP = MLP,
    ResNet = ResNet,
    CNN = CNN,
    AlexNet = AlexNet,
    VGG = VGG,
    Ensemble = Ensemble
)
########### DATASETS ##################################################
class ShuffledCIFAR10:
    def __init__(self, *, 
                 pixel_shuffled = None, image_width = IMAGE_WIDTH, train = True, download=False, transform = None,
                    fix_permutation = None):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=download, transform=transform)
        self.pixel_shuffled = pixel_shuffled
        if pixel_shuffled == True:
            self.image_width = image_width
            self.perm = torch.randperm(image_width * image_width)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        inputs, labels = self.dataset[idx]
        if self.pixel_shuffled == True:
            inputs = inputs.view(IMAGE_CHANNELS, self.image_width * self.image_width)
            inputs = inputs[:, self.perm].view(IMAGE_CHANNELS, self.image_width, self.image_width)
        return inputs, labels
    
########### TRAINERS ##################################################
class BaseTrainer:
    def __init__(self, data_params = dict(),
                       train_params = dict(),
                       model_optim_params = dict(),
                       save_params = dict()):
        
        self.data_params = data_params
        self.train_params = train_params
        self.model_optim_params = model_optim_params
        self.save_params = save_params
        self.setup()
        
    def setup(self):
        self.build_data_loader()
        self.build_model_optimizer()
        self.initialize_record()
        
    def save_record(self):
        # Save record
        fname = f"{self.save_params['save_dir']}/{self.save_params['exp_name']}"
        save_file_pickle(fname, self.record)
        
class CIFAR_trainer(BaseTrainer):
    def build_data_loader(self):
        batch_size = self.train_params['batch_size']
        
        trainset = ShuffledCIFAR10(train=True, **self.data_params) 
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

        testset = ShuffledCIFAR10(train=False, **self.data_params) 
        
        if self.data_params["pixel_shuffled"] == True:
            if self.data_params["fix_permutation"] == True:
                try:
                    trainset.perm = torch.load(f"{self.save_params['save_dir']}/{self.model_optim_params['model_name']}_quenched_permutation.pt")
                except:
                    torch.save(trainset.perm, f"{self.save_params['save_dir']}/{self.model_optim_params['model_name']}_quenched_permutation.pt")
            testset.perm = trainset.perm

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    def build_model_optimizer(self):
        self.model = self.model_optim_params["model"]
        self.criterion = self.model_optim_params["criterion"]
        self.optimizer = self.model_optim_params["optimizer"]
        
    def initialize_record(self):
        self.record = dict(
            metrics = dict(
                train_loss_prog = [],
                train_acc_prog = [],
                test_loss_prog = [],
                test_acc_prog = [],
                test_loss = 0.0,
                test_accuracy = 0.0
            ),    
            print_interval = 5000 // self.train_params['batch_size'],
            data_params = self.data_params,
            train_params = self.train_params,
            model_optim_params = self.model_optim_params,
            save_params = self.save_params,
            model = None
        )
        
    def before_train_epoch(self):
        # after a certain epoch, freeze convolutional layers if desired
        if self.epoch > self.train_params["freeze_params"]["epoch"]:
            freeze_layer(self.model, *self.train_params["freeze_params"]["epoch"])
            for n, param in self.model.named_parameters():         
                print(n, param.requires_grad)
    def after_train_iter(self, running_loss, running_accuracy, outputs, labels):
        self.loss.backward()
        self.optimizer.step()
        
        # get top-1 accuracy
        pred_class = torch.argmax(outputs, dim=1)
        running_accuracy += torch.mean((pred_class == labels).float()).item()

        # print statistics
        running_loss += self.loss.item()
        if self.iter % self.record["print_interval"] == self.record["print_interval"] - 1:    # print every self.record["print_interval"] mini-batches
            running_loss /= self.record["print_interval"]
            running_accuracy /= self.record["print_interval"]
            self.record["metrics"]["train_loss_prog"].append(running_loss)
            self.record["metrics"]["train_acc_prog"].append(running_accuracy)
            print(f'[{self.epoch + 1}, {self.iter + 1:5d}] loss: {running_loss:.3f}, accuracy:  {running_accuracy:.3f}')
            running_loss = 0.0
            running_accuracy = 0.0
        return running_loss, running_accuracy
    
    def after_train_epoch(self):
        # get test loss and accuracy
        test_loss, test_accuracy = self.test()
        self.record["metrics"]["test_loss_prog"].append(test_loss)
        self.record["metrics"]["test_acc_prog"].append(test_accuracy)
        
        # put model back into train mode
        self.model.train()
        
        
    def train(self):
        num_epochs = self.train_params['num_epochs']
        for self.epoch in range(num_epochs):  # loop over the dataset multiple times
            self.before_train_epoch()
            running_loss, running_accuracy = 0.0, 0.0
            for self.iter, data in enumerate(self.trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                self.loss = self.criterion(outputs, labels)
                
                running_loss, running_accuracy = self.after_train_iter(running_loss, running_accuracy, outputs, labels)

            self.after_train_epoch()

        print('Finished Training')
        self.after_run()
        
    def test(self):
        running_loss, running_accuracy = 0.0, 0.0
        self.model.eval()
        for i, data in enumerate(self.testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward and evaluate
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            
            # get top-1 accuracy
            pred_class = torch.argmax(outputs, dim=1)
            running_accuracy += torch.mean((pred_class == labels).float()).item()

        # print loss
        running_loss /= len(self.testloader)
        running_accuracy /= len(self.testloader)
        print(f'Test loss: {running_loss:.3f}, accuracy: {running_accuracy:.3f}')
        return running_loss, running_accuracy
    
    def after_run(self):
        # Get final test loss and accuracy
        test_loss, test_accuracy = self.test()
        self.record["metrics"]["test_loss"] = test_loss
        self.record["metrics"]["test_accuracy"] = test_accuracy
        
        # Save model
        self.record["model"] = list(self.model.named_parameters())
        
        # Save record
        self.save_record()

        

################# METERS ########################
from enum import Enum
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



################# SAVING FILES ########################
def save_file_pickle(fname, file):
    with open(f"{fname}.pkl", 'wb') as f:
        pickle.dump(file, f)
        
def save_checkpoint(state, save_dir = "./", filename='checkpoint.pth.tar'):
    torch.save(state, f"{save_dir}/{filename}")
    
                               
def load_file_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
                               
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
                               
                               
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def show_plt_if_in_notebook(title=None):
    if in_notebook():
        plt.show()
    else:
        plt.savefig(title)

from IPython.display import display 
from collections import defaultdict
import glob 
import seaborn as sns 
import math 
import matplotlib.pyplot as plt 
import pandas as pd 

def get_record(  is_online, extra = "", title = None,  
               zero_out_list = None,
               image_transform_loader_list = None,
               tiling_orientation_ablation_list = None,
               tiling_list = None,
               outdir = "/scratch/gpfs/qanguyen/poly1/",
              palette = sns.color_palette("Set3", 10),
               hue_variable = "data_rescale",
               num_hidden_features_list = None,
               max_epoch = 0, 
               num_runs = 1000,
):
     
    train_pars = defaultdict(list)
    val_pars = defaultdict(list)
   
    
    record_names = glob.glob(f"{outdir}/lr0.0005gradient_descent_noisyJUL18*pth.tar")
     
    for _, f in enumerate(record_names[:num_runs]):
        #print(f)
        try:
            
            record = torch.load(f, map_location="cpu") 
        except Exception as e: 
            print(e)
        
        try:
            
            
            if (num_hidden_features_list is not None) and (record.args.num_hidden_features not in num_hidden_features_list):   
                continue
        except Exception as e: 
            print(e)
            print(f, "continue", record.curr_epoch )
            continue
        
        #print(f, "plotting" )
        for epoch in range(max_epoch):
         
            
             
            
            width_after_pool = math.floor((224 - record.args.coarsegrain_blocksize) / record.args.coarsegrain_blocksize + 1)
            D = 3*(width_after_pool)*(width_after_pool)
            
            N =  record.args.num_train_samples
            
            P = record.args.num_hidden_features
            if P < N:
                continue 
            train_pars["block_size"].extend( [record.args.coarsegrain_blocksize]) 
            train_pars["N"].extend([N])
            train_pars["D"].extend([D]) 
            train_pars["P"].extend([ f"P={P},N={N}" ])
            train_pars["logP/D"].extend([np.log( record.args.num_hidden_features / D) ])
            train_pars["logN/D"].extend([np.log( N / D) ])
            train_pars["epoch"].append(epoch)
            train_pars["train_loss"].append( (record.metrics.train_mse[epoch]) )

    # plot train_loss as a function of epochs 
    train_pars = pd.DataFrame(train_pars) 
    train_pars = train_pars.sort_values(by=["P", "block_size", "epoch"]) 
    sns.lineplot(x="epoch", y="train_loss", hue="P", data=train_pars, palette=palette)  
    plt.title(title) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show() 

def small_lr_regression_exp(  is_online, extra = "", title = None,  
               zero_out_list = None,
               image_transform_loader_list = None,
               tiling_orientation_ablation_list = None,
               tiling_list = None,
               outdir = "/scratch/gpfs/qanguyen/poly1/",
              palette = sns.color_palette("Set3", 10),
               hue_variable = "data_rescale",
               num_hidden_features_list = None,
               max_epoch = 0
):
    pd.set_option('display.max_rows', 1000)
    train_pars = defaultdict(list)
    val_pars = defaultdict(list)
     
    record_names = glob.glob(f"{outdir}/lr0.01gradient_descent_noisyJUL24*pth.tar")
     
    for _, f in enumerate(record_names) :
        #print(f)
        try:
            
            record = torch.load(f, map_location="cpu") 
        except Exception as e: 
            print(e)
        
        try:
            #if record.curr_epoch > 27:
            #    continue
            #print(w)
              
            #print( record.args )

            #if (zero_out_list is not None) and (record.args.zero_out not in zero_out_list):
            #    continue
            
            if (num_hidden_features_list is not None) and (record.args.num_hidden_features not in num_hidden_features_list):   
                continue
            #print(f, "plotting" )
            for epoch in range(max_epoch):
            #for epoch in range( 1):
                #print(epoch, f)
                
                #pars["data_rescale"].append(record.data_rescale)
                #pars["data_rescale"].append(record.args.growth_factor)
                #pars["tiling_imagenet"].append(record.args.tiling_imagenet)
                val_pars["block_size"].append(f'{record.args.coarsegrain_blocksize}')
                train_pars["block_size"].extend( [record.args.coarsegrain_blocksize]) 
                #val_pars["lr"].append(f'{record.args.lr}')
                #train_pars["lr"].extend([ record.args.lr]  * len(record.metrics.train_losses[epoch]))
                val_pars["P"].append(record.args.num_hidden_features)
                train_pars["P"].extend([ record.args.num_hidden_features ])
                width_after_pool = math.floor((224 - record.args.coarsegrain_blocksize) / record.args.coarsegrain_blocksize + 1)
                D = 3*(width_after_pool)*(width_after_pool)
                val_pars["D"].append( D)
                train_pars["D"].extend([D]) 
                N =  record.args.num_train_samples
                val_pars["N"].append(N)
                train_pars["N"].extend([N])
                val_pars["logP/D"].append( np.log( record.args.num_hidden_features / D ))
                train_pars["logP/D"].extend([np.log( record.args.num_hidden_features / D) ])
                val_pars["logN/D"].append( np.log( N / D ))
                train_pars["logN/D"].extend([np.log( N / D) ])
                #val_pars["P"].append(f'{record.args.num_hidden_features}')
                #train_pars["P"].extend([ record.args.num_hidden_features ])
                val_pars["epoch"].append(epoch)
                train_pars["epoch"].append(epoch)
                #mean_train_loss = np.mean([i  for i in record.metrics.train_losses["default"][epoch] if i != 0.0])
                train_pars["train_loss"].append( (record.metrics.train_mse[epoch]) )
                train_pars["distance_to_true"].append( (record.metrics.distance_to_true[epoch]).item() )
                #print( record.metrics.train_losses)
                train_pars["test_loss"].append( (record.metrics.test_mse  ) )
                val_pars["test_loss"].append( (record.metrics.test_mse ) )
            
        except Exception as e: 
            print(e)
            continue
        
        
             
        #if _ > 40: break  
        #print("record.args.coarsegrain_blocksize, record.args.num_hidden_features,  record.args.train_fraction", record.args.coarsegrain_blocksize, record.args.num_hidden_features,  record.args.train_fraction,   record.metrics.val_losses[epoch])
    line_kwargs    = dict(linewidth=0.5, alpha=0.7, style=hue_variable,
             markers=False, markersize=8, markeredgecolor='white',
             dashes=False)
    figheight, figwidth = (12, 8)
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    train_pars = pd.DataFrame.from_dict(train_pars) 
    #display(train_pars) 
    train_pars = train_pars.sort_values('test_loss', ascending=True)  
    grouped = train_pars.groupby(["P", "N"])
    for name, group in grouped:
        #if name not in [(100, 10), (1000, 100), (10000, 1000), (10000, 100)]:
        #    continue
        #display(grouped)
        group = group.sort_values(by=["D", "epoch"], ascending=True)
        display(group)
        sns.lineplot(x="epoch",y= "train_loss", data=group, label=name)
    ax.set(  yscale="log")
    plt.title(f"Train loss vs. D")
    plt.legend()
    plt.show()
    
    for name, group in grouped:
        #if name not in [(100, 10), (1000, 100), (10000, 1000), (10000, 100)]:
        #    continue
        #display(grouped)
        group = group.sort_values(by="D", ascending=True)
        sns.lineplot(x="epoch",y= "distance_to_true", data=group, label=name)
    ax.set(  yscale="log")
    plt.title(f"Distance to ridge regression solution vs. D")
    plt.legend()
    plt.show() 
    
    
    # Interpolate 
    x = np.array(train_pars["logN/D"])
    y = np.array(train_pars["logP/D"])
    z = np.array(train_pars["train_loss"])
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
            #   vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha=0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    
  
    plt.title(f"Train loss block_size vs. num_hidden_features heatmap")
    show_plt_if_in_notebook("train_loss_vs_epochs.png")
    
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    val_pars = pd.DataFrame.from_dict(val_pars) 
    val_pars = val_pars.sort_values(by=hue_variable, ascending=True)[val_pars["epoch"] == max_epoch ]
#     val_loss = val_pars.pivot("logP/D", "logN/D", "test_loss")
    val_loss = pd.pivot_table(val_pars, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    display(val_pars.sort_values(by="test_loss", ascending=True))
    x = np.array(val_pars["logN/D"])
    y = np.array(val_pars["logP/D"])
    z = np.array(val_pars["test_loss"])
    xx = np.array(val_loss.columns)
    yy = np.array(val_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
        #     vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
              #  vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha= 0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title(f"Test loss block_size vs. num_hidden_features heatmap")
    show_plt_if_in_notebook("test_loss_vs_epochs.png")
    
    # LOG TRAIN & TEST LOSSES
    figheight, figwidth = (12, 8)
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    train_pars = pd.DataFrame.from_dict(train_pars) 
    #display(train_pars) 
    train_pars = train_pars.sort_values(by="block_size", ascending=True)[train_pars["epoch"] == max_epoch ]
    train_loss = pd.pivot_table(train_pars, columns="logP/D", index="logN/D", values="train_loss",
                               aggfunc='mean'
                               )
    display(train_loss)
    # Interpolate 
    x = np.array(train_pars["logN/D"])
    y = np.array(train_pars["logP/D"])
    z = np.log(np.array(train_pars["train_loss"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
            #   vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha=0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    
  
    plt.title(f"Train log loss block_size vs. num_hidden_features heatmap")
    show_plt_if_in_notebook("log_train_loss_vs_epochs.png")
    
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    val_pars = pd.DataFrame.from_dict(val_pars) 
    val_pars = val_pars.sort_values(by=hue_variable, ascending=True)[val_pars["epoch"] == max_epoch ]
    #val_loss = val_pars.pivot("logP/D", "logN/D", "test_loss")
    val_loss = pd.pivot_table(val_pars, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    x = np.array(val_pars["logN/D"])
    y = np.array(val_pars["logP/D"])
    z = np.log(np.array(val_pars["test_loss"]))
    xx = np.array(val_loss.columns)
    yy = np.array(val_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
        #     vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
              #  vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha= 0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title(f"Test log loss block_size vs. num_hidden_features heatmap")
    show_plt_if_in_notebook("log_test_loss_vs_epochs.png")

# MNIST 
# def mnist_classification_exp(      
#                outdir = "",
#                hue_variable = "data_rescale",
#                max_epoch = 0,
#                num_runs_to_analyze=1000
# ):

# MNIST 
def mnist_classification_calibrate_exp(      
               outdir = "",
               hue_variable = "data_rescale",
               max_epoch = 0,
               num_runs_to_analyze=1000
):
    pd.set_option('display.max_rows', 1000)
    train_pars = defaultdict(list)
     
    record_names = glob.glob(f"{outdir}/linelr*_wd1em5_cifar_fullnetOCT30*pth.tar")
     
    for _, f in enumerate(record_names[:num_runs_to_analyze]) :
        
        try:
            
            record = torch.load(f, map_location="cpu") 
            if np.log(record.metrics.test_mse[max_epoch]) > 10:
                continue
        except Exception as e: 
            print(e)
        
        try:
              
            # for epoch in range(max_epoch):
            for epoch in [max_epoch]:
                train_pars["block_size"].extend( [record.args.coarsegrain_blocksize]) 
                train_pars["lr"].append(f'{record.args.lr}')
                train_pars["block"].append(f'{record.args.coarsegrain_blocksize}')
                train_pars["P"].extend([ record.args.num_hidden_features ])
                if "mnist" in f:
                    width_after_pool = math.floor((28 - record.args.coarsegrain_blocksize) / record.args.coarsegrain_blocksize + 1)
                    D = 1*(width_after_pool)*(width_after_pool)
                elif "cifar" in f:
                    width_after_pool = math.floor((32 - record.args.coarsegrain_blocksize) / record.args.coarsegrain_blocksize + 1)
                    D = 3*(width_after_pool)*(width_after_pool)
                 
                train_pars["D"].extend([D]) 
                N =  record.args.num_train_samples
                
                train_pars["N"].extend([N])
                train_pars["logP/D"].extend([np.log( record.args.num_hidden_features / D) ])
                train_pars["logN/D"].extend([np.log( N / D) ])
                train_pars["epoch"].append(epoch)
                train_pars["train_loss"].append( (record.metrics.train_mse[epoch]) )
                train_pars["test_loss"].append( (record.metrics.test_mse[epoch]  ) )
                # train_pars["train_top5"].append( (record.metrics.train_top5[epoch]  ) )
                train_pars["test_top5"].append( (record.metrics.test_top5[epoch].item()  ) )
                # train_pars["train_top1"].append( (record.metrics.train_top1[epoch]  ) )
                train_pars["test_top1"].append( (record.metrics.test_top1[epoch].item()  ) ) 
                # train_pars["l1_calibration"].append( (record.metrics.l1_calibration[epoch].item()  ) ) 
                # train_pars["l2_calibration"].append( (record.metrics.l2_calibration[epoch].item()  ) )
                # train_pars["lmax_calibration"].append( (record.metrics.lmax_calibration[epoch].item()  ) )
                #train_pars["weight_norm"].append( (record.metrics.weight_norm[epoch].item()  ) )

        except Exception as e: 
            print(e, record.metrics.test_mse[epoch])
            raise ValueError
        
        
              
    figheight, figwidth = (12, 8)
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    train_pars = pd.DataFrame.from_dict(train_pars) 
    #display(train_pars) 
    train_pars = train_pars.sort_values('test_loss', ascending=True)  
    
    # plot train loss
    grouped = train_pars.groupby(["P", "N"])
    num_groups = 0
    for name, group in grouped:
        p, n = name
        if (p > n) and (n > 10):
            num_groups += 1 
    # make palette with num_groups colors
    palette = sns.color_palette("icefire", num_groups)
    num_groups = 0
    for name, group in grouped:
    
        group = group.sort_values(by=["D", "test_loss"  ], ascending=[True, True])
    
        group = group.drop_duplicates(subset=["D"], keep="first")
 
        p, n = name 
        if (p > n) and (n > 10):
            
            sns.lineplot(x="D",y= "train_loss", data=group, label=name, color=palette[num_groups])
            num_groups += 1
    ax.set(  yscale="log")
    plt.title(f"Train loss vs. D")
    plt.legend(loc=(1.04,0))
    plt.show()

    grouped = train_pars.groupby(["P", "N"])
    num_groups = 0
    for name, group in grouped:
        p, n = name
        if (p > n) and (n > 10):
            num_groups += 1 
    # make palette with num_groups colors
    palette = sns.color_palette("icefire", num_groups)
    num_groups = 0
    for name, group in grouped:
      
        group = group.sort_values(by=["D", "test_loss"  ], ascending=[True, True])
        group = group.drop_duplicates(subset=["D"], keep="first")
        p, n = name 
        if (p > n) and (n > 10):
            
            sns.lineplot(x="D",y= "test_loss", data=group, label=name, color=palette[num_groups])
            num_groups += 1
    ax.set(  yscale="log")
    plt.title(f"Test loss vs. D")
    plt.legend(loc=(1.04,0))
    plt.show()

    
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    grouped = train_pars.groupby(["P", "N"])
    num_groups = 0
    for name, group in grouped:
        p, n = name
        if (p > n) and (n > 10):
            num_groups += 1 
    # make palette with num_groups colors
    palette = sns.color_palette("icefire", num_groups)
    num_groups = 0
    
    for name, group in grouped:
        #if name not in [(100, 10), (1000, 100), (10000, 1000), (10000, 100)]:
        #    continue
        #display(grouped)
        group = group.sort_values(by=["D", "test_top1"  ], ascending=[True, True])
        #display(group)
        group = group.drop_duplicates(subset=["D"], keep="first")
        #display(group)
        p, n = name 
        if (p > n) and (n > 10):
            
            sns.lineplot(x="D",y= "test_top1", data=group, label=name, color=palette[num_groups])
            num_groups += 1
    # ax.set(  yscale="log")
    plt.title(f"Test top 1 accuracy vs. D")
    plt.legend(loc=(1.04,0))
    plt.show()

    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    #display(train_pars) 
    grouped = train_pars.groupby(["P", "N"])
    num_groups = 0
    for name, group in grouped:
        p, n = name
        if (p > n) and (n > 10):
            num_groups += 1 
    # make palette with num_groups colors
    palette = sns.color_palette("icefire", num_groups)
    num_groups = 0
    for name, group in grouped:
        #if name not in [(100, 10), (1000, 100), (10000, 1000), (10000, 100)]:
        #    continue
        #display(grouped)
        group = group.sort_values(by=["D", "test_top5"  ], ascending=[True, True])
        #display(group)
        group = group.drop_duplicates(subset=["D"], keep="first")
        #display(group)
        p, n = name 
        if (p > n) and (n > 10):
             
            sns.lineplot(x="D",y= "test_top5", data=group, label=name, color=palette[num_groups])
            num_groups += 1
    # ax.set(  yscale="log")
    plt.title(f"Test top 5 accuracy vs. D")
    plt.legend(loc=(1.04,0))
    plt.show()

    # plot weight norm 
    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # grouped = train_pars.groupby(["P", "N"])
    # num_groups = 0
    # for name, group in grouped:
    #     p, n = name
    #     if (p > n) and (n > 10):
    #         num_groups += 1 
    # # make palette with num_groups colors
    # palette = sns.color_palette("icefire", num_groups)
    # num_groups = 0
    
    # for name, group in grouped:
         
    #     group = group.sort_values(by=["D", "test_loss"  ], ascending=[True, True])
    #     #display(group)
    #     group = group.drop_duplicates(subset=["D"], keep="first")
    #     #display(group)
    #     p, n = name 
    #     if (p > n) and (n > 10):
            
    #         sns.lineplot(x="D",y= "weight_norm", data=group, label=name, color=palette[num_groups])
    #         num_groups += 1
    # ax.set(  yscale="log")
    # plt.title(f"Weight norm of best test loss run vs. D")
    # plt.legend(loc=(1.04,0))
    # plt.show()
    
    train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    # print("train_pars")
    # display(train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).iloc[:1000 , :])
    # print("train_pars_heatmap")
    # display(train_pars_heatmap)
    train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    #display(train_loss)
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    x = np.array(train_pars_heatmap["logN/D"])
    y = np.array(train_pars_heatmap["logP/D"])
    z =np.log(np.array(train_pars_heatmap["test_loss"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
            #   vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha=0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title("Log est loss heat map")
    plt.show()
    
    # top 1 acc heatmap 
    train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    x = np.array(train_pars_heatmap["logN/D"])
    y = np.array(train_pars_heatmap["logP/D"])
    z = (np.array(train_pars_heatmap["test_top1"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
            #   vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha=0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title("Test top 1 accuracy heat map")
    plt.show()
     
    # top 5 acc heatmap 
    train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    x = np.array(train_pars_heatmap["logN/D"])
    y = np.array(train_pars_heatmap["logP/D"])
    z = (np.array(train_pars_heatmap["test_top5"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    plt.colorbar()
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',
            #   vmin=-25,vmax=30,
               )
    plt.scatter(x, y, 
               c = "blue",
                marker="+",
                alpha=0.1
               )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title("Test top 5 accuracy heat map")
    plt.show()

    # l1 calibration heatmap
    # train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    # train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
    #                            aggfunc='mean'
    #                            )
    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # x = np.array(train_pars_heatmap["logN/D"])
    # y = np.array(train_pars_heatmap["logP/D"])
    # z = np.log(np.array(train_pars_heatmap["l1_calibration"]))
    # xx = np.array(train_loss.columns)
    # yy = np.array(train_loss.index.values.tolist())
    # xx, yy = np.meshgrid(xx, yy)
    # points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    # plt.contourf(xx, yy, 
    #              interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
    #           #   vmin=-25,vmax=30,
    #              cmap='Spectral_r')
    # plt.colorbar()
    # plt.scatter(x, y, 
    #            c = z,
    #              cmap='Spectral_r',
    #         #   vmin=-25,vmax=30,
    #            )
    # plt.scatter(x, y, 
    #            c = "blue",
    #             marker="+",
    #             alpha=0.1
    #            )
    # plt.plot(x, x, '-')
    # plt.xlabel("logN/D")
    # plt.ylabel("logP/D")
    # plt.title("log l1 calibration heat map")
    # plt.show()

    # # l2 calibration heatmap
    # train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    # train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
    #                            aggfunc='mean'
    #                            )
    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # x = np.array(train_pars_heatmap["logN/D"])
    # y = np.array(train_pars_heatmap["logP/D"])
    # z = np.log(np.array(train_pars_heatmap["l2_calibration"]))
    # xx = np.array(train_loss.columns)
    # yy = np.array(train_loss.index.values.tolist())
    # xx, yy = np.meshgrid(xx, yy)
    # points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    # plt.contourf(xx, yy, 
    #              interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
    #           #   vmin=-25,vmax=30,
    #              cmap='Spectral_r')
    # plt.colorbar()
    # plt.scatter(x, y, 
    #            c = z,
    #              cmap='Spectral_r',
    #         #   vmin=-25,vmax=30,
    #            )
    # plt.scatter(x, y, 
    #            c = "blue",
    #             marker="+",
    #             alpha=0.1
    #            )
    # plt.plot(x, x, '-')
    # plt.xlabel("logN/D")
    # plt.ylabel("logP/D")
    # plt.title("log l2 calibration heat map")
    # plt.show()

    # # lmax calibration heatmap
    # train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    # train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
    #                            aggfunc='mean'
    #                            )
    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # x = np.array(train_pars_heatmap["logN/D"])
    # y = np.array(train_pars_heatmap["logP/D"])
    # z = np.log(np.array(train_pars_heatmap["lmax_calibration"]))
    # xx = np.array(train_loss.columns)
    # yy = np.array(train_loss.index.values.tolist())
    # xx, yy = np.meshgrid(xx, yy)
    # points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    # plt.contourf(xx, yy, 
    #              interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
    #           #   vmin=-25,vmax=30,
    #              cmap='Spectral_r')
    # plt.colorbar()
    # plt.scatter(x, y, 
    #            c = z,
    #              cmap='Spectral_r',
    #         #   vmin=-25,vmax=30,
    #            )
    # plt.scatter(x, y, 
    #            c = "blue",
    #             marker="+",
    #             alpha=0.1
    #            )
    # plt.plot(x, x, '-')
    # plt.xlabel("logN/D")
    # plt.ylabel("logP/D")
    # plt.title("log lmax calibration heat map")
    # plt.show()

def fractional_mnist(      
               outdir = "",
               exp_name = "",
               hue_variable = "data_rescale",
               max_epoch = 0,
               num_runs_to_analyze=1000
):
    pd.set_option('display.max_rows', 1000)
    train_pars = defaultdict(list)
    def save_checkpoint(state, save_dir = "./", filename='checkpoint.pth.tar'):
        torch.save(state, f"{save_dir}/{filename}")
    record_names = glob.glob(f"{outdir}/{exp_name}")
    print ("num runs", len(record_names), flush=True)
    for _, f in enumerate(record_names[:num_runs_to_analyze]) :
        # if _ % 1000 == 0: print (_, flush=True)
        try:
            
            record = torch.load(f, map_location="cpu") 
            # record.metrics.weight_norm = [] 
            # save_checkpoint(record, save_dir = record.args.save_dir, filename = record.args.exp_name)
            if record.args.num_hidden_features > 10000:
                continue
            if record.args.num_train_samples < 10:
                continue
        except Exception as e: 
            print(e)
            continue 
        try:
            
            # for epoch in range(max_epoch):
            for epoch in [max_epoch]:
                train_pars["target_size"].extend( [record.args.target_size]) 
                train_pars["lr"].append(f'{record.args.lr}')
                train_pars["P"].extend([ record.args.num_hidden_features ])
                 
                D = record.args.target_size #* record.args.target_size
                train_pars["D"].extend([D]) 
                N =  record.args.num_train_samples
                
                train_pars["N"].extend([N])
                train_pars["logP/D"].extend([np.log( record.args.num_hidden_features / D) ])
                train_pars["logN/D"].extend([np.log( N / D) ])
                train_pars["epoch"].append(epoch)
                train_pars["train_loss"].append( (record.metrics.train_mse[epoch]) )
                train_pars["test_loss"].append( (record.metrics.test_mse[epoch]  ) )
                # train_pars["train_top5"].append( (record.metrics.train_top5[epoch]  ) )
                train_pars["test_top5"].append( (record.metrics.test_top5[epoch].item()  ) )
                # train_pars["train_top1"].append( (record.metrics.train_top1[epoch]  ) )
                train_pars["test_top1"].append( (record.metrics.test_top1[epoch].item()  ) ) 
                # train_pars["l1_calibration"].append( (record.metrics.l1_calibration[epoch].item()  ) ) 
                # train_pars["l2_calibration"].append( (record.metrics.l2_calibration[epoch].item()  ) )
                # train_pars["lmax_calibration"].append( (record.metrics.lmax_calibration[epoch].item()  ) )
                #train_pars["weight_norm"].append( (record.metrics.weight_norm[epoch].item()  ) )
                
        except Exception as e: 
            print(e )
            raise ValueError
        
        
              
    figheight, figwidth = (12, 8)
    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    train_pars = pd.DataFrame.from_dict(train_pars) 
    # #display(train_pars) 
    train_pars = train_pars.sort_values('test_loss', ascending=True)  
    
    # plot train loss
    # grouped = train_pars.groupby(["P", "N"])
    # num_groups = 0 
    # # make palette with num_groups colors
    # palette = sns.color_palette("icefire", len(grouped))
    # num_groups = 0
    # for name, group in grouped:
    #     print (name)
    #     group = group.sort_values(by=["D", "test_loss"  ], ascending=[True, True])
    #     # print ("pre drop")
    #     # display(group)
    #     group = group.drop_duplicates(subset=["P", "N" ,"D"], keep="first")
    #     # print ("post drop")
    #     # display(group)
    #     p, n = name 
    #     # if (p > n) and (n > 10):
            
    #     sns.lineplot(x="D",y= "train_loss", data=group, label=name, color=palette[num_groups], ax=ax)
    #     num_groups += 1
    # ax.set(  yscale="log")
    # plt.title(f"Train loss vs. D")
    # plt.legend(loc=(1.04,0))
    # plt.show()

    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # grouped = train_pars.groupby(["P", "N"])
    # # make palette with num_groups colors
    # num_groups = 0
    # for name, group in grouped:
    #     print (name)
    #     group = group.sort_values(by=["D", "test_loss"  ], ascending=[True, True])
    #     # print ("pre drop")
    #     # display(group)
    #     group = group.drop_duplicates(subset=["P", "N" ,"D"], keep="first")
    #     # print ("post drop")
    #     # display(group)
        
    #     sns.lineplot(x="D",y= "test_loss", data=group, label=name, color=palette[num_groups], ax=ax)
    #     num_groups += 1
    # # ax.set(  yscale="log")
    # plt.ylim(0, 5)
    # plt.title(f"Test loss vs. D")
    # plt.legend(loc=(1.04,0))
    # plt.show()

    
    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # grouped = train_pars.groupby(["P", "N"])
   
    # # make palette with num_groups colors    
    # for name, group in grouped:
    #     print (name)
    #     #if name not in [(100, 10), (1000, 100), (10000, 1000), (10000, 100)]:
    #     #    continue
    #     #display(grouped)
    #     group = group.sort_values(by=["D", "test_top1"  ], ascending=[True, True])
    #     #display(group)
    #     group = group.drop_duplicates(subset=["D"], keep="first")
    #     #display(group)
    #     p, n = name 
    #     # if (p > n) and (n > 10):
            
    #     sns.lineplot(x="D",y= "test_top1", data=group, label=name, color=palette[num_groups])
    #     num_groups += 1
    # # ax.set(  yscale="log")
    # plt.title(f"Test top 1 accuracy vs. D")
    # plt.legend(loc=(1.04,0))
    # plt.show()

    # fig, ax = plt.subplots(figsize=(figheight, figwidth))
    # #display(train_pars) 
    # grouped = train_pars.groupby(["P", "N"])
    # num_groups = 0
    # for name, group in grouped:
    #     p, n = name
    #     # if (p > n) and (n > 10):
    #     num_groups += 1 
    # # make palette with num_groups colors
    # palette = sns.color_palette("icefire", num_groups)
    # num_groups = 0
    # for name, group in grouped:
    #     #if name not in [(100, 10), (1000, 100), (10000, 1000), (10000, 100)]:
    #     #    continue
    #     #display(grouped)
    #     group = group.sort_values(by=["D", "test_top5"  ], ascending=[True, True])
    #     #display(group)
    #     group = group.drop_duplicates(subset=["D"], keep="first")
    #     #display(group)
    #     p, n = name 
    #     # if (p > n) and (n > 10):
             
    #     sns.lineplot(x="D",y= "test_top5", data=group, label=name, color=palette[num_groups])
    #     num_groups += 1
    # # ax.set(  yscale="log")
    # plt.title(f"Test top 5 accuracy vs. D")
    # plt.legend(loc=(1.04,0))
    # plt.show()
 
    
    train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    
    train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    #display(train_loss)
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    x = np.array(train_pars_heatmap["logN/D"]) 
    x += np.random.normal(0, 0.02, x.shape)
    y = np.array(train_pars_heatmap["logP/D"])
    y += np.random.normal(0, 0.02, y.shape)
    z =np.log(np.array(train_pars_heatmap["test_loss"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',#marker="+",
                 alpha=0.3
               )
    plt.colorbar()
    # plt.scatter(x, y, 
    #            c = "blue",
    #             marker="+",
    #             alpha=0.1
    #            )

    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title("Log test loss heat map")
    plt.show()
    
    # top 1 acc heatmap 
    train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    x = np.array(train_pars_heatmap["logN/D"])
    x += np.random.normal(0, 0.02, x.shape)
    y = np.array(train_pars_heatmap["logP/D"])
    y += np.random.normal(0, 0.02, y.shape)
    z = (np.array(train_pars_heatmap["test_top1"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
              #   vmin=-25,vmax=30,
                 cmap='Spectral_r')
    
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',#marker="+",
                 alpha=0.3
               )
    plt.colorbar()
    # plt.scatter(x, y, 
    #            c = z,
    #              cmap='Spectral_r',marker="+",
    #              alpha=0.5
    #            )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title("Test top 1 accuracy heat map")
    plt.show()
     
    # top 5 acc heatmap 
    train_pars_heatmap = train_pars.sort_values(["P","N","D",'test_loss'], ascending=True).drop_duplicates(subset=["P","N","D"], keep="first")  
    train_loss = pd.pivot_table(train_pars_heatmap, columns="logP/D", index="logN/D", values="test_loss",
                               aggfunc='mean'
                               )
    fig, ax = plt.subplots(figsize=(figheight, figwidth))
    x = np.array(train_pars_heatmap["logN/D"])
    x += np.random.normal(0, 0.02, x.shape)
    y = np.array(train_pars_heatmap["logP/D"])
    y += np.random.normal(0, 0.02, y.shape)
    z = (np.array(train_pars_heatmap["test_top5"]))
    xx = np.array(train_loss.columns)
    yy = np.array(train_loss.index.values.tolist())
    xx, yy = np.meshgrid(xx, yy)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    plt.contourf(xx, yy, 
                 interpolate.griddata(np.vstack((x, y)).T, z, points).reshape(xx.shape), 
                 cmap='Spectral_r')
    plt.scatter(x, y, 
               c = z,
                 cmap='Spectral_r',#marker="+",
                 alpha=0.3
               )
    plt.colorbar()
    # plt.scatter(x, y, 
    #            c = "blue",
    #             marker="+",
    #             alpha=0.1
    #            )
    plt.plot(x, x, '-')
    plt.xlabel("logN/D")
    plt.ylabel("logP/D")
    plt.title("Test top 5 accuracy heat map")
    plt.show()

def lbfgs_plot(      
               outdir = "",
               exp_name = "",
               hue_variable = "data_rescale",
               max_epoch = 0,
               num_runs_to_analyze=1000,
):
    pd.set_option('display.max_rows', 1000)
    train_pars = defaultdict(list) 
    record_names = glob.glob(f"{outdir}/{exp_name}")
    print ("num runs", len(record_names), flush=True)
    for _, f in enumerate(record_names[:num_runs_to_analyze]) :
        try:
            
            record = torch.load(f, map_location="cpu")  
        except Exception as e: 
            print(e)
            continue 
        try:
            
            # epoch = max([int(i) for i in record.metrics.train_mse.keys() if i != "default"]) 
            # for epoch in [max_epoch]:
            # train_pars["target_size"].extend( [record.args.target_size]) 
            train_pars["target_size"].extend( [record["target_size"] ])
            train_pars["n_pca_components_kept"].extend( [record["target_size"] ])
            # train_pars["lr"].append(f'{record.args.lr}') 
            # train_pars["wd"].append(f'{record.args.weight_decay}') 
            # print ('lr',record.args.lr)
            # train_pars["P"].extend([ record.args.num_hidden_features ])
            
            # D = record.args.target_size #* record.args.target_size
            train_pars["D"].extend([record["target_size"] ])
            # train_pars["D"].extend([record.args.target_size]) 
            # N =  record.args.num_train_samples
            # P = record.args.num_hidden_features
            train_pars["N"].extend([record["num_train_samples"] ])
            # train_pars["N"].extend([record.args.num_train_samples ])
            # train_pars["P,N"].extend([f'{P},{N}']/) 
            # train_pars["logP+logN"].extend([1/2*np.log(P/D) + 1/2*np.log(N/D) ])
            # train_pars["logP-logN"].extend([ np.round (np.log(P/D) - np.log(N/D), 1) ])
            # train_pars["logP/D"].extend([np.log( record.args.num_hidden_features / D) ])
            # train_pars["logN/D"].extend([np.log( N / D) ])
            # train_pars["epoch"].append(epoch)
            # train_pars["train_loss"].append( (record.metrics.train_mse[epoch]) )
            train_pars["train_loss"].append( (record["train_loss"]))
            # train_pars["test_loss"].append( (record.metrics.test_mse[epoch]  ) )
            train_pars["test_loss"].append ( (record["test_loss"]  ) )
            # train_pars["train_top5"].append( (record.metrics.train_top5[epoch]  ) )
            # train_pars["test_top5"].append( (record.metrics.test_top5[epoch].item()  ) )
            # print ("record.metrics.train_top1", record.metrics.train_top1)
            # train_pars["train_top1"].append( (record.metrics.train_top1[epoch].item()  ) )
            # print ('P', record.args.num_hidden_features, "epoch", epoch, "train_loss", record.metrics.train_mse[epoch], "test_loss", record.metrics.test_mse[epoch], "train_top1", record.metrics.train_top1[epoch].item(), "nonlin", record.args.nonlinearity)
            # train_pars["test_top1"].append( (record.metrics.test_top1[epoch].item()  ) ) 
            train_pars["test_top1"].append( (record["test_score"]  ) )
            train_pars["train_top1"].append( (record["train_score"]  ) )
                
                
        except Exception as e: 
            print(e )
            raise ValueError
    train_pars = pd.DataFrame.from_dict(train_pars) 
    # train_pars = train_pars.sort_values(["N","P","D",'test_loss'], ascending=True)
    train_pars = train_pars.sort_values(["N", "D",'test_loss'], ascending=True) 

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "train_loss", data=train_pars, hue="N", marker="+")
    # ax.set(  yscale="log")
    plt.title(f"Train loss vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Train loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "train_top1", data=train_pars, hue="N", marker="+")
    plt.title(f"Train top1 vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Train top1")
    plt.show() 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "test_loss", data=train_pars, hue="N", marker="+")
    # ax.set(  yscale="log")
    plt.title(f"Test loss vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Test loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "test_top1", data=train_pars, hue="N", marker="+")
    plt.title(f"Test top1 vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Test top1")
    plt.show()

     
 
    return train_pars
          
def slices_plot(      
               outdir = "",
               exp_name = "",
               hue_variable = "data_rescale",
               max_epoch = 0,
               num_runs_to_analyze=1000,
               P_list = None
):
    pd.set_option('display.max_rows', 1000)
    train_pars = defaultdict(list) 
    record_names = glob.glob(f"{outdir}/{exp_name}")
    print ("P", P_list, "num runs", len(record_names), flush=True)
    for _, f in enumerate(record_names[:num_runs_to_analyze]) :
        try:
            
            record = torch.load(f, map_location="cpu") 
            if (P_list is not None) and (record.args.num_hidden_features not in P_list):
                continue
            # if record.args.num_hidden_features > 10000:
                # continue
            # if record.args.num_train_samples < 10:
                # continue
        except Exception as e: 
            print(e)
            continue 
        try:
            
            epoch = max([int(i) for i in record.metrics.train_mse.keys() if i != "default"]) 
            # for epoch in [max_epoch]:
            train_pars["target_size"].extend( [record.args.target_size]) 
            # train_pars["target_size"].extend( [record["target_size"] ])
            # train_pars["lr"].append(f'{record.args.lr}') 
            # train_pars["wd"].append(f'{record.args.weight_decay}') 
            # print ('lr',record.args.lr)
            train_pars["P"].extend([ record.args.num_hidden_features ])
            
            # D = record.args.target_size #* record.args.target_size
            # train_pars["D"].extend([record["target_size"] ])
            train_pars["D"].extend([record.args.target_size]) 
            # N =  record.args.num_train_samples
            # P = record.args.num_hidden_features
            # train_pars["N"].extend([record["num_train_samples"] ])
            train_pars["N"].extend([record.args.num_train_samples ])
            # train_pars["P,N"].extend([f'{P},{N}']/) 
            # train_pars["logP+logN"].extend([1/2*np.log(P/D) + 1/2*np.log(N/D) ])
            # train_pars["logP-logN"].extend([ np.round (np.log(P/D) - np.log(N/D), 1) ])
            # train_pars["logP/D"].extend([np.log( record.args.num_hidden_features / D) ])
            # train_pars["logN/D"].extend([np.log( N / D) ])
            # train_pars["epoch"].append(epoch)
            train_pars["train_loss"].append( (record.metrics.train_mse[epoch]) )
            # train_pars["train_loss"].append( (record["train_loss"]))
            train_pars["test_loss"].append( (record.metrics.test_mse[epoch]  ) )
            # train_pars["test_loss"].append ( (record["test_loss"]  ) )
            # train_pars["train_top5"].append( (record.metrics.train_top5[epoch]  ) )
            # train_pars["test_top5"].append( (record.metrics.test_top5[epoch].item()  ) )
            # print ("record.metrics.train_top1", record.metrics.train_top1)
            train_pars["train_top1"].append( (record.metrics.train_top1[epoch].item()  ) )
            # print ('P', record.args.num_hidden_features, "epoch", epoch, "train_loss", record.metrics.train_mse[epoch], "test_loss", record.metrics.test_mse[epoch], "train_top1", record.metrics.train_top1[epoch].item(), "nonlin", record.args.nonlinearity)
            train_pars["test_top1"].append( (record.metrics.test_top1[epoch].item()  ) ) 
            # train_pars["test_top1"].append( (record["test_score"]  ) )
            # train_pars["train_top1"].append( (record["train_score"]  ) )
                
                
        except Exception as e: 
            print(e )
            raise ValueError

    train_pars = pd.DataFrame.from_dict(train_pars) 
    # train_pars = train_pars.sort_values(["N","P","D",'test_loss'], ascending=True)
    # train_pars = train_pars.sort_values(["N", "D",'test_loss'], ascending=True)
    # display(train_pars)
    # get unique Ps

    print ("unique Ps", train_pars["P"].unique())
    # display(train_pars.iloc[:100, :]) 
    # train_pars = train_pars.drop_duplicates(subset=["N","P","D"], keep="first")  
    # train_pars = train_pars.drop_duplicates(subset=["N", "D"], keep="first")  
    # display(train_pars.iloc[:100, :]) 
    # fig, ax = plt.subplots(figsize=(12, 8))
    # sns.lineplot(x="N",y= "train_loss", data=train_pars, hue="D")
    # ax.set(  xscale="log",  yscale="log")
    # plt.title(f"Train loss vs. N")
    # plt.legend(loc=(1.04,0))
    # # plt.legend([],[], frameon=False)

    # plt.xlabel("N")
    # plt.ylabel("Log Train loss")
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    # sns.lineplot(x="N",y= "test_loss", data=train_pars, hue="D")
    # ax.set(  xscale="log",  yscale="log")
    # plt.title(f"Test loss vs. N")
    # plt.legend(loc=(1.04,0))
    # # plt.legend([],[], frameon=False)

    # plt.xlabel("N")
    # plt.ylabel("Log Test loss")
    # plt.show()


    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "train_loss", data=train_pars, hue="N", marker="+")
    # ax.set(  yscale="log")
    plt.title(f"Train loss vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Train loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "train_top1", data=train_pars, hue="N", marker="+")
    plt.title(f"Train top1 vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Train top1")
    plt.show() 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "test_loss", data=train_pars, hue="N", marker="+")
    # ax.set(  yscale="log")
    plt.title(f"Test loss vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Test loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="D",y= "test_top1", data=train_pars, hue="N", marker="+")
    plt.title(f"Test top1 vs. D")
    plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)

    plt.xlabel("D")
    plt.ylabel("Test top1")
    plt.show()

    
    # sort train_pars by logP-logN 
    # train_pars = train_pars.sort_values(by=["logP-logN", "logP+logN"  ], ascending=[True, True])
    # # display first 1000 rows of train_pars
    # # display(train_pars.iloc[:100, :]) 
    # # get unique values of logP-logN
    # logP_logNs = train_pars["logP-logN"].unique() 
    # # get every 5th value of logP_logN
    # nrows = int(len(logP_logNs)//4)+1
    
    # fig, ax = plt.subplots(nrows, 4, figsize=(30, 4*nrows)) 
    # print (nrows, len(logP_logNs), len(ax.flatten()))
    # for i, logP_logN in enumerate(logP_logNs):
    #     # only keep values of train_pars where logP-logN is in logP_logN 
    #     sns.scatterplot(x="logP+logN",y= "test_loss", data=train_pars[train_pars["logP-logN"].isin([logP_logN])],
    #             color="black", marker="*", ax=ax.flatten()[i])
    #     ax.flatten()[i].set(  yscale="log", title=f"logP-logN={logP_logN}", ylim=(0.1, 3))
    # plt.suptitle(f"Test loss vs. logP/D+logN/D, keeping logP-logN constant", y=1.0)
    # # plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)
    # plt.xlabel("logP/D+logN/D")
    # plt.ylabel("Log Test loss")
    # plt.tight_layout(pad=1.15)
    # plt.show()

    # fig, ax = plt.subplots(nrows, 4, figsize=(30, 4*nrows)) 
    
    # for i, logP_logN in enumerate(logP_logNs):
    #     # only keep values of train_pars where logP-logN is in logP_logN 
    #     sns.scatterplot(x="logP+logN",y= "test_top1", data=train_pars[train_pars["logP-logN"].isin([logP_logN])],
    #             color="black", marker="*", ax=ax.flatten()[i])
    #     ax.flatten()[i].set(   title=f"logP-logN={logP_logN}", ylim=(0, 100))
    # plt.suptitle(f"Test top1 acc. vs. logP/D+logN/D, keeping logP-logN constant", y=1.0)
    # # plt.legend(loc=(1.04,0))
    # plt.legend([],[], frameon=False)
    # plt.xlabel("logP/D+logN/D")
    # plt.ylabel("Log Test loss")
    # plt.tight_layout(pad=1.15)
    # plt.show()

    # fig,ax = plt.subplots(figsize=(12, 8)) 
    # # hue="logP-logN", 
    # sns.scatterplot(x="logP/D",y= "logN/D", data=train_pars, hue="test_loss", 
    #                 palette=sns.color_palette("Spectral", as_cmap=True)) 
    # plt.xlabel("logP/D") 
    # plt.ylabel("logN/D")
    # plt.title("LogN/D vs. LogP/D")
    # plt.show()
    
 
    return train_pars

import torchvision.datasets as datasets         
class RandomFeaturesMNIST(datasets.MNIST):
    def __init__(self, root = "./data",  
                 train = True,
                 transform = None,
                 block_size = 1, 
                 upsample = False,
                 target_size = None,
                ):
        
        
        # self.block_size = block_size        
        # self.avg_kernel = nn.AvgPool2d(block_size, stride=block_size)
        # # self.avg_kernel no grad 
        # for param in self.avg_kernel.parameters():
        #     param.requires_grad = False

        # if target_size is not None:
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
        # if self.upsample:
        sample = torch.matmul (sample.view(1, -1), self.retransform_matrix)
        sample = sample.view(1, 28, 28) * 28 * 28
        # else:
        #     sample = self.avg_kernel(sample)
        #     if self.upsample:
        #         sample = torch.repeat_interleave(sample,   self.block_size, dim=1)
        #         sample = torch.repeat_interleave(sample,   self.block_size, dim=2)
        #         sample_size = sample.shape[-1]
        #         if sample_size != 28: # if not the original size, pad the last few pixels
        #             remainder = 28 - sample_size # size of imagenet - current sample size
        #             sample = torch.cat([sample, sample[:, :, -remainder:]], dim=-1) # pad the last few pixels
        #             sample = torch.cat([sample, sample[:, -remainder:, :]], dim=-2) # pad the last few pixels
                

        return sample, target
    
import torchvision.transforms as transforms
def analyze_error(      
               outdir = "",
               exp_name = "",
               hue_variable = "data_rescale",
               max_epoch = 0,
               num_runs_to_analyze=1000,
               P_list = None,
               target_size_list=None,
                N_list=None
):
    pd.set_option('display.max_rows', 1000)
    train_pars = defaultdict(list) 
    record_names = glob.glob(f"{outdir}/{exp_name}")
    print ("P", P_list, "num runs", len(record_names), flush=True)
    for _, f in enumerate(record_names[:num_runs_to_analyze]) :
        try:
            
            record = torch.load(f, map_location="cpu") 
            if (P_list is not None) and (record.args.num_hidden_features not in P_list):
                continue
            if (target_size_list is not None) and (record.args.target_size not in target_size_list):
                continue
            if (N_list is not None) and (record.args.num_train_samples not in N_list):
                continue
                
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_dataset = RandomFeaturesMNIST(root = "./data", 
                                        train = True,
                                        transform = transform,
                                        target_size = record.args.target_size,
                                        upsample = record.args.upsample
                                         ) 
            train_sampler=None
            train_kwargs = {'batch_size': 100}
            train_loader = torch.utils.data.DataLoader(
                train_dataset, sampler=train_sampler, shuffle=(train_sampler is None),
            **train_kwargs)
            
            test_dataset = RandomFeaturesMNIST(root = "./data", 
                                        train = False,
                                        transform = transform,
                                        target_size = record.args.target_size,
                                        upsample = record.args.upsample
                                         ) 
            test_sampler=None
            test_kwargs = {'batch_size': 100}
            test_loader = torch.utils.data.DataLoader(
                test_dataset, sampler=test_sampler, shuffle=(test_sampler is None),
            **test_kwargs)
            
            
            width_after_pool = 28
            if record.args.nonlinearity == "relu":
                nonlinearity = nn.ReLU()
            elif record.args.nonlinearity == "tanh":
                nonlinearity = nn.Tanh()
            elif record.args.nonlinearity == "line":
                nonlinearity = nn.Identity()
            else:
                raise ValueError
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width_after_pool**2, record.args.num_hidden_features),
                nonlinearity,
                nn.Linear(record.args.num_hidden_features, 10),
            )
            model.load_state_dict(record.model)
            
            
            # if record.args.num_hidden_features > 10000:
                # continue
            # if record.args.num_train_samples < 10:
                # continue
        except Exception as e: 
            print(e)
            continue 
        print("Results on Train")
        criterion = nn.CrossEntropyLoss()
        for i, (images, target) in enumerate(train_loader):
            images = images.to("cpu")
            target = target.to("cpu")
            output = model(images) 
            logits = []
            digits = []
            target_digits = []
            for dig in range(10):
                logits.extend(output[target == dig].detach().numpy().flatten())
                digits.extend([0,1,2,3,4,5,6,7,8,9]*len(output[target == dig] ))
                target_digits.extend([dig]*10*len(output[target == dig] ))
                print("train logits for digit:", dig)
                for o in output[target == dig].detach().numpy():
                    plt.plot(o)
                plt.show()
            # print lens of each list 
            # print (len(logits), len(digits), len(target_digits))
            df = pd.DataFrame({"logits": logits, "digits": digits,
                               "target_digits": target_digits,
                               })
            
            sns.lineplot(x="digits",y= "logits", hue = "target_digits",data=df)
            plt.show()
            # min max output 
            print (output.min(), output.max())
            plt.figure(figsize=(12, 8))
            plt.imshow(output[torch.argsort(target)][:100].detach().numpy().T,vmin=-100,vmax=100)
            plt.show()
            # print (output[torch.argsort(target)], torch.argsort(target))
            break 
            
        print("Results on Test")
        for i, (images, target) in enumerate(test_loader):
            images = images.to("cpu")
            target = target.to("cpu")
            output = model(images) 
            logits = []
            digits = []
            target_digits = []
            print("batch", i, "N", record.args.num_train_samples, "target", record.args.target_size, "cross entropy", criterion(output, target))
            for dig in range(10):
                logits.extend(output[target == dig].detach().numpy().flatten())
                digits.extend([0,1,2,3,4,5,6,7,8,9]*len(output[target == dig] ))
                target_digits.extend([dig]*10*len(output[target == dig] ))
                print("test logits for digit:", dig)
                for o in output[target == dig].detach().numpy():
                    plt.plot(o)
                plt.show()
            # print lens of each list 
            # print (len(logits), len(digits), len(target_digits))
            df = pd.DataFrame({"logits": logits, "digits": digits,
                               "target_digits": target_digits,
                               })
            
            sns.lineplot(x="digits",y= "logits", hue = "target_digits",data=df)
            plt.show()
            # min max output 
            print (output.min(), output.max())
            plt.figure(figsize=(12, 8))
            plt.imshow(output[torch.argsort(target)][:100].detach().numpy().T,vmin=-100,vmax=100)
            plt.show()
            # print (output[torch.argsort(target)], torch.argsort(target))
            break 
         
         