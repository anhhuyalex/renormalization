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
import io

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
        
class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True,):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1,1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim = 1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X.T @ X 
        rhs = X.T @ y
        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0], device=X.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution
            
    def predict(self, X: torch.tensor) -> None:
        # print ("w", self.w.shape)
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim = 1)
        return X @ self.w
    
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

################# SCHEDULERS ########################
class EmptyScheduler:
    def __init__(self):
        pass
    def step(self):
        pass
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass

################# SAVING FILES ########################
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
def save_file_pickle(fname, file):
    with open(f"{fname}.pkl", 'wb') as f:
        pickle.dump(file, f)
        
def save_checkpoint(state, save_dir = "./", filename='checkpoint.pth.tar'):
    torch.save(state, f"{save_dir}/{filename}")
    
                               
def load_file_pickle(fname):
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except:
        with open(fname, 'rb') as f:
            return CPU_Unpickler(f).load()
                               
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

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

