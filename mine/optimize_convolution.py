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
                loss.backward(retain_graph=True)
                opt.step()

                mu_mi -= loss.item()
            #iter_print = iter //  3
            iter_print = 3
            if iter % (iter_print) == 0:
                # pass
                print(f"It {iter} - MI: {mu_mi / batch_size} ")
            
            if iter % (iter_mi) == 0:
                current_mi = self.mi(X, Y)
                print(f"It {iter} - Current MI: {current_mi} ")
                if best_mi < current_mi:
                    best_mi = current_mi
        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        if best_mi < final_mi:
            best_mi = final_mi
        return final_mi, best_mi, opt



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



class CoarseGrain(nn.Module):
    """
    Neural network to coarse-grain a 2x2 model
    Assume input has been flattened
    """
    def __init__(self, x_dim):
        super(CoarseGrain, self).__init__()
        self.weights = nn.Linear(x_dim, 1, bias = False)

        
    def forward(self, fine_grained):
        coarse_grained = self.weights(fine_grained)
        return coarse_grained
        
        






def maximize_coarse_grain_MI(train_coarse_grain_iters, 
                             train_mine_iters, lr=0.01, 
                             coarse_grainer_opt = None, coarse_grain_schedule = None):
    coarse_grainer = CoarseGrain(9).cuda()
    if coarse_grainer_opt is None:
        coarse_grainer_opt = torch.optim.Adam(coarse_grainer.parameters(), lr=lr)
    # if coarse_grain_schedule is not None:
    #     coarse_grain_scheduler = torch.optim.lr_scheduler.StepLR(coarse_grainer_opt, 
    #                                                 step_size=schedule['step_size'], 
    #                                                 gamma=schedule['gamma'])

    mine = Mine(
        T = AltNet(x_dim =  1, y_dim = 1),
        loss = 'mine' #mine_biased, fdiv
    ).cuda()
    mine_opt = None

    dat = np.load("ising200x200from2187x2187.npy")
    for coarsegrain_iter in range(train_coarse_grain_iters):
        
        Xlist = []
        Ylist = []
        for _ in range(3000):
            num = np.random.randint(dat.shape[0])
            i = np.random.randint(dat.shape[1] - 3)
            j = np.random.randint(dat.shape[1] - 6)
            Xlist.append(dat[num, i:(i+3), j:(j+3)])
            Ylist.append(dat[num, i:(i+3), (j+3):(j+6)])
        X = np.array(Xlist).astype(float).reshape(-1, 9)
        Y = np.array(Ylist).astype(float).reshape(-1, 9)
        X = torch.Tensor(np.array(Xlist).astype(float).reshape(-1, 9)).cuda()
        Y = torch.Tensor(np.array(Ylist).astype(float).reshape(-1, 9)).cuda()

        coarse_grained_X = coarse_grainer(X)
        coarse_grained_Y = coarse_grainer(Y)

        train_loader = DataLoader(DatasetVar(coarse_grained_X, coarse_grained_Y), 
                                  batch_size=256, shuffle=True)
        # Update MINE estimator
        final_mi, best_mi, mine_opt = mine.optimize(coarse_grained_X, coarse_grained_Y, 
                                     iters = train_mine_iters, batch_size = 512, 
                       train_loader = train_loader, lr = 0.005, opt = mine_opt,
                       schedule = {'step_size': 30, 'gamma': 0.5})
        print("Done with mine iter", coarsegrain_iter, final_mi, best_mi)
        # Update coarse graining
        train_loader = DataLoader(DatasetVar(X, Y), batch_size=512, shuffle=True)
        batching = 10 if _ < 5 else 1000000
        for batch, (x, y) in enumerate(train_loader):
            # print("In coarse grainer")
            coarse_grainer_opt.zero_grad()
            loss = mine(coarse_grainer(x), coarse_grainer(y))
            loss.backward()
            coarse_grainer_opt.step()
            if batch > batching:
                break

        iter_print = 1
        if _ % (iter_print) == 0:
            # pass
            print(f"It {coarsegrain_iter} - MI: {mine.mi(coarse_grained_X, coarse_grained_Y)} ")
            for param in coarse_grainer.parameters():
                print (param)
            

maximize_coarse_grain_MI(100, 5)   
# print(coarse_grainer(X).shape)
# for param in coarse_grainer.parameters():
#     print (param)
# print(X[:10])

