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
import os
import gc
import sys
from collections import defaultdict

# import ray

# ray.init()
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

    def mi_on_eval_set(self, eval_loader):
        """
        return MI on evaluation set dataloader
        """
        num_batches = 0
        current_mi = 0
        for batch, (x, y) in enumerate(eval_loader):
            current_mi += self.mi(x.float().cuda(), y.float().cuda())
            num_batches += 1
        current_mi /= num_batches
        return current_mi
    
    def optimize(self, iters, batch_size, lr=1e-4, opt=None, 
                 train_loader = None, 
                 eval_loader = None,
                 schedule = None, iter_mi = 30):
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
                loss.backward(retain_graph=True)
                opt.step()

                mu_mi -= loss.item()
            #iter_print = iter //  3
            iter_print = 3
            
            if iter % (iter_mi) == 0:
                current_mi = self.mi_on_eval_set(eval_loader)
                print(f"It {iter} - Current MI: {current_mi} ")
                
                if best_mi < current_mi:
                    best_mi = current_mi
        final_mi = self.mi_on_eval_set(eval_loader)
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
    def __init__(self, k, dat, num_batches = 256):
        self.k = k
        self.dat = dat
        self.banksize, self.width = self.dat.shape 
        self.num_batches = num_batches
        
    def __len__(self):
        return self.banksize
    def __getitem__(self, idx):
        
        i = np.random.randint(3, self.width - self.k) # avoid left and right boundary for easy slicing
        X = self.dat[idx,i:(i+self.k)]
        Y = self.dat[idx,(i-3):i]
        return X, Y
    
    
    
def get_line_dataset():
    lattice2187 = glob("../data_2187_1571810501/lattice_*")
    dataset = []
    for _ in range(1000):
        dat = np.load(np.random.choice(lattice2187))['arr_0']
        i = np.random.choice(dat.shape[0], size=np.random.randint(1, 20), replace=False)
        dataset.append(dat[i])
        if _ % 100 == 0:
            print(_)
            del dat
            gc.collect()
    dataset = np.vstack(dataset)
    print(dataset.shape)
    return dataset

    
def compute_linemi(k, random_loc = True, line_dataset_fpath = "isinglinefrom2187x2187.npy"):
    dat = np.load(line_dataset_fpath)
    # compute MI
    mine = Mine(
        T = AltNet(x_dim = k, y_dim = 3),
        loss = 'mine' #mine_biased, fdiv
    ).cuda()
    
    
    train_loader = DataLoader(DatasetVar(k=k, dat=dat, num_batches = 256), batch_size=512)
    eval_loader = DataLoader(DatasetVar(k=k, dat=dat, num_batches = 20), batch_size=512)
    final_mi, best_mi = mine.optimize(iters = 150, batch_size = 512,  iter_mi = 2,
                       train_loader = train_loader,
                                      eval_loader = eval_loader,
                                      lr = 0.003, 
                       schedule = {'step_size': (30+k*2)/2, 'gamma': 0.5})
    return best_mi

if __name__ == "__main__":
    # if os.path.exists("isinglinefrom2187x2187.npy") == False:
    #     for i in range(10):
    #         dataset = get_line_dataset()
    #         np.save(f"isinglinefrom2187x2187_batch{i}.npy", dataset)
    #     dat = np.vstack([np.load(path) for path in glob("isingline*")])
    #     np.save("isinglinefrom2187x2187.npy", dat)
    # else:
    #     dat = np.load("isinglinefrom2187x2187.npy")
    job_num = sys.argv[1]
    best_mi_dict_savepath = f"smallseg_saved_lineMI_dictionary_randomloc_150iters_job{job_num}.pkl"
    if os.path.exists(best_mi_dict_savepath) == False:
        print(f"{best_mi_dict_savepath} does NOT exist!")
        mis = defaultdict(list)
    else:
        print(f"{best_mi_dict_savepath} exists!")
        with open(best_mi_dict_savepath, 'rb') as file:
            mis = dill.load(file)

    for _ in range(10):
        for k in range(100, 200):      
            print("k", k)
            jobs = []
            for _ in range(1):
                 best_mi = compute_linemi(k, random_loc = True)
    #         best_mi = ray.get(jobs)

            print(k, best_mi)
            mis[k].append(best_mi)
            with open(best_mi_dict_savepath, 'wb') as file:
                dill.dump(mis, file)
