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

    def forward(self, x, zleft, zright):
        z_marg_left = zleft[torch.randperm(x.shape[0])]
        z_marg_right = zright[torch.randperm(x.shape[0])]

        t = self.T(x, zleft, zright).mean()
        t_marg = self.T(x, z_marg_left, z_marg_right)
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

    def mi(self, x, zleft, zright):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, zleft, zright)
        return mi

    def mi_on_eval_set(self, eval_loader):
        """
        return MI on evaluation set dataloader
        """
        num_batches = 0
        current_mi = 0
        for batch, (x, yleft, yright) in enumerate(eval_loader):
            current_mi += self.mi(x.float().cuda(), yleft.float().cuda(), yright.float().cuda())
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
            for batch, (x, yleft, yright) in enumerate(train_loader):
                opt.zero_grad()
                loss = self.forward(x.float().cuda(), yleft.float().cuda(), yright.float().cuda())
                loss.backward(retain_graph=True)
                opt.step()

                mu_mi -= loss.item()
                
            #iter_print = iter //  3
            iter_print = 3
            print("x, y", x.shape, y.shape)
            if iter % (iter_mi) == 0:
                current_mi = self.mi_on_eval_set(eval_loader)
                print(f"It {iter} - Current MI: {current_mi} ")
                
                if best_mi < current_mi:
                    best_mi = current_mi
            
#             print("conv_1", self.T.state_dict()['conv_1.bias'])
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
#         self.x_fc1 = nn.Linear(x_dim + y_dim, H)
#         self.x_fc2 = nn.Linear(H, H)
#         self.x_fc3 = nn.Linear(H, H)
        self.x_fc_conv = nn.Linear(30, 1)
        self.conv_1_x, self.conv_1_y = [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1) 
                                         for _ in range(2)  ]
        self.conv_2, self.conv_3, self.conv_4 = [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2) \
                                        for _ in range(3)]
        self.x_fc1 = nn.Linear(x_dim + y_dim, H)
        self.x_fc2 = nn.Linear(H, H)
        self.x_fc3 = nn.Linear(H, H)
        self.x_fc4 = nn.Linear(H, 1)
        
    def forward(self, x, yleft, yright):
        x = x.unsqueeze(1)
        yleft = yleft.unsqueeze(1)
        print("x", x.shape, x[:10,:,:])
        print("yleft", yleft.shape)
        x = F.relu(self.conv_1_x(x))
        yleft = F.relu(self.conv_1_y(yleft))
        print("x1", x.shape, x[:10,:,:])
        print("yleft", yleft.shape, yleft[:10,:,:])

#         xy = F.relu(self.conv_2(xy))
#         xy = F.relu(self.conv_3(xy))
#         xy = F.relu(self.conv_4(xy))
#         # inter_xy = F.relu(self.x_fc1(xy)) # dim [batch, 100]
#         # inter_xy = inter_xy + F.relu(self.x_fc2(inter_xy)) # dim [batch, 100]
#         # inter_xy = inter_xy + F.relu(self.x_fc3(inter_xy)) # dim [batch, 100]
#         h2 = self.x_fc_conv(xy.view(xy.size(0), -1))
#         xy = torch.cat([x, y], 1) # dim [batch, 2]
#         inter_xy = F.relu(self.x_fc1(xy)) # dim [batch, 100]
#         inter_xy = inter_xy + F.relu(self.x_fc2(inter_xy)) # dim [batch, 100]
#         inter_xy = inter_xy + F.relu(self.x_fc3(inter_xy)) # dim [batch, 100]
        
#         h2 = (self.x_fc4(inter_xy))
#         return h2    


class DatasetVar(Dataset):
    def __init__(self, k, dat, left_start = 5, 
                 right_end = 1, num_batches = 256):
        self.k = k
        self.dat = dat
        self.banksize, self.width = self.dat.shape 
        self.num_batches = num_batches
        self.left_start = left_start
        self.right_end = right_end
        
    def __len__(self):
        return self.banksize
    
    def __getitem__(self, idx):
        i = np.random.randint(self.width - self.k) # avoid right boundary for easy slicing
        
        X = np.take(self.dat[idx], range(i, i+self.k), mode='wrap') 
        Yleft = np.take(self.dat[idx], range(0, i), mode='wrap') 
        Yright = np.take(self.dat[idx], range(i+self.k, self.width), mode='wrap') 
#         Y = np.concatenate([self.dat[idx, :i], self.dat[idx, (i + k):]])
#         print(Y.shape)
        return X, Yleft, Yright
    
    
    


    
def compute_linemi(k, left_start, right_end, random_loc = True, line_dataset_fpath = "isinglinefrom2187x2187.npy"):
    dat = np.load(line_dataset_fpath)
    # compute MI
    mine = Mine(
        T = AltNet(x_dim = k, y_dim = left_start+right_end),
        loss = 'mine' #mine_biased, fdiv
    ).cuda()
    
    
    train_loader = DataLoader(DatasetVar(k=k, dat=dat, left_start = left_start, 
                                         right_end = right_end,
                                         num_batches = 256), batch_size=512)
    eval_loader = DataLoader(DatasetVar(k=k, dat=dat, left_start = left_start, 
                                        right_end = right_end,
                                        num_batches = 20), batch_size=512)
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
    best_mi_dict_savepath = f"k=5_rightend_rg_smallseg_saved_lineMI_dictionary_randomloc_150iters_job{job_num}.pkl"
    if os.path.exists(best_mi_dict_savepath) == False:
        print(f"{best_mi_dict_savepath} does NOT exist!")
        mis = defaultdict(list)
    else:
        print(f"{best_mi_dict_savepath} exists!")
        with open(best_mi_dict_savepath, 'rb') as file:
            mis = dill.load(file)

    for _ in range(100):
        for k in range(5, 6):      
            for right_end in range(20, 100):
                print("k", k, "right_end", right_end)
                jobs = []
                for _ in range(1):
                     best_mi = compute_linemi(k, left_start = 2000, right_end = right_end, random_loc = True)
        #         best_mi = ray.get(jobs)

                print(k, best_mi)
                mis[k].append(best_mi)
                with open(best_mi_dict_savepath, 'wb') as file:
                    dill.dump(mis, file)
