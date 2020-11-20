import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from mine.models.layers import ConcatLayer, CustomSequential
from mine.models.mine import EMALoss, ema, ema_loss
import dill
import gc

    
import ray

ray.init()
class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)
        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])
            #print("t_marg", t_marg.shape, t_marg)
            #print("second_term", second_term)
        #print(-t, second_term)
        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, lr=1e-4, opt=None, 
                 train_loader = None, schedule = None, iter_mi = 30):
        best_mi = -1
        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        if schedule is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 
                                                        step_size=schedule['step_size'], 
                                                        gamma=schedule['gamma'])


        for iter in range(1, iters + 1):
            
            mu_mi = 0
            # for x, y in mine.utils.batch(X, Y, batch_size):
            for batch, (x, y) in enumerate(train_loader):
                opt.zero_grad()
                loss = self.forward(x.float().cuda(), y.float().cuda())
                loss.backward()
                opt.step()

                mu_mi -= loss.item()
            #iter_print = iter //  3
            iter_print = 3
            if iter % (iter_print) == 0:
                # pass
                print(f"It {iter} - MI: {mu_mi / batch_size} ")
            
            if iter % (iter_mi) == 0:
                current_mi = self.mi(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
                print(f"It {iter} - Current MI: {current_mi} ")
                if best_mi < current_mi:
                    best_mi = current_mi
        final_mi = self.mi(torch.Tensor(X).cuda(), torch.Tensor(Y).cuda())
        print(f"Final MI: {final_mi}")
        if best_mi < final_mi:
            best_mi = final_mi
        return final_mi, best_mi



class AltNet(nn.Module):
    """
    Neural network to compute mutual information
    """
    def __init__(self, x_dim, y_dim, H = 100):
        super(AltNet, self).__init__()
        self.x_fc1 = nn.Linear(x_dim + y_dim, H)
        self.x_fc2 = nn.Linear(H, H)
        self.x_fc3 = nn.Linear(H, H)
        self.x_fc4 = nn.Linear(H, 1)
        
        
    def forward(self, x, y):
        xy = torch.cat([x, y], 1) # dim [batch, 2]
        inter_xy = F.relu(self.x_fc1(xy)) # dim [batch, 100]
        inter_xy = inter_xy + F.relu(self.x_fc2(inter_xy)) # dim [batch, 100]
        inter_xy = inter_xy + F.relu(self.x_fc3(inter_xy)) # dim [batch, 100]
        
        h2 = (self.x_fc4(inter_xy))
        return h2    

class DatasetVar(Dataset):
    def __init__(self, varX, varY):
        self.varX = varX
        self.varY = varY
        
        assert len(varX) == len(varY), "X and Y must have equal length"
    def __len__(self):
        return len(self.varX)
    def __getitem__(self, idx):
        return self.varX[idx], self.varY[idx]
    
def get_borders(start, size, boundary):
    assert 0 < boundary - size, "Size of square must be less than size of lattice"
    start_x, start_y = start
    indices = [[], []]
    for k in range(size):
        indices[0].append(start_x + k )
        indices[1].append(start_y)
    for k in range(1, size):
        indices[0].append(start_x + size - 1 )
        indices[1].append(start_y + k )
    for k in range(size-2, 0, -1):
        indices[0].append(start_x + k )
        indices[1].append(start_y + size - 1 )
    for k in range(size-1, 0, -1):
        indices[0].append(start_x  )
        indices[1].append(start_y + k )
    return indices


@ray.remote(num_gpus=1)
def get_mi_block_size(k):
    dat = np.load("ising200x200from2187x2187.npy")

    Xlist = []
    Ylist = []
    while len(Xlist) < 50000:
        num = np.random.randint(dat.shape[0])
        i, j = np.random.randint(1, dat.shape[1] - k, size = 2)
        inside = dat[num][tuple(get_borders([i, j], k, dat.shape[1]))]
        outside = dat[num][tuple(get_borders([i-1, j-1], k+2, dat.shape[1]))]
        Xlist.append(inside)
        Ylist.append(outside)
    X = np.array(Xlist).astype(float) 
    Y = np.array(Ylist).astype(float) 
    
    mine = Mine(
        T = AltNet(x_dim =  X.shape[1], y_dim = Y.shape[1]),
        loss = 'mine' #mine_biased, fdiv
    ).cuda()
    
    print(X[:5], Y[:5])
    train_loader = DataLoader(DatasetVar(X, Y), batch_size=512)
    final_mi, best_mi = mine.optimize(X, Y, iters = 30+k, batch_size = 512,  iter_mi = 2,
                       train_loader = train_loader, lr = 0.003, 
                       schedule = {'step_size': int((30+k*1.5)/2), 'gamma': 0.5})
    return best_mi
from collections import defaultdict

with open('saved_MI_best_dictionary_4_to_100.pkl', 'rb') as file:
    mis = dill.load(file)
for k in range(18, 100):
    jobs = []
    for _ in range(4):
         jobs.append(get_mi_block_size.remote(k))
    best_mi = ray.get(jobs)
    
    print(k, best_mi)
    mis[k].append(best_mi)
    with open('saved_MI_best_dictionary_18_to_100.pkl', 'wb') as file:
        dill.dump(mis, file)
    
                                                                  
