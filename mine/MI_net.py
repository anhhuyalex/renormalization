from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
arrs = []
# for arr in tqdm(glob("../data_2187_1571810501/*")[2500:5000]):
#     z = np.load(arr)['arr_0']
#     x, y = np.random.randint(2187, size=2)
#     z = np.pad(z, pad_width=27, mode='wrap')[x:(x+27), y:(y+27)]
#     arrs.append(z)
# arrs = np.stack(arrs)

arrs = np.load("./ising27x27from2187x2187.npy")[:, None, :, :]
arrs.shape

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
S=9
H=15
n_epoch = 100
data_size = 20000
def pad_circular(x, pad):
    """
    function inspired by https://github.com/pytorch/pytorch/issues/3858
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x

class Net(nn.Module):
    def __init__(self, out_channels = 1, filter_size = 3):
        super(Net, self).__init__()

        self.x_fc1 = nn.Linear(S, S)
        self.x_fc2 = nn.Linear(S, S)
        self.x_fc3 = nn.Linear(S, H)

        self.y_fc1 = nn.Linear(S, S)
        self.y_fc2 = nn.Linear(S, S)
        self.y_fc3 = nn.Linear(S, H)

        self.fc3 = nn.Linear(H, 1)

        self.conv1 = nn.Conv2d(1, out_channels, filter_size, padding=0, stride = filter_size)


    def MI_net(self, x, y):
        inter_x = x + F.relu(self.x_fc1(x))
        inter_x = inter_x + F.relu(self.x_fc2(inter_x))
        inter_x = self.x_fc3(inter_x)

        inter_y = y + F.relu(self.y_fc1(y))
        inter_y = inter_y + F.relu(self.y_fc2(inter_y))
        inter_y = self.y_fc3(inter_y)


        h1 = F.relu(inter_x + inter_y)
        h2 = self.fc3(h1)
        return h2

    def forward(self, x):
        # coarse grain 27x27 model to get 9x9 model
        x = (self.conv1(x))

        rectangle = []
        # get a random 3x6 rectangle
        for data in x:
            i, j = np.random.randint(x.size()[2], size = 2)
            rectangle.append( pad_circular(data[0], 6)[i:(i+3), j:(j+6)])


        rectangle = torch.stack(rectangle)
        # separate rectangle into 2 patches of shape 3x3
        patch1 = rectangle[:, :3, :3].reshape(-1, 9)
        patch2 = rectangle[:, :3, 3:].reshape(-1, 9)
        patch2_shuffled = patch2[torch.randperm(patch2.shape[0])]

        pred_xy = self.MI_net(patch1, patch2)
        pred_x_y = self.MI_net(patch1, patch2_shuffled)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        return -ret

class IsingDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, arrs):
        'Initialization'
        self.arrs = torch.Tensor(arrs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.arrs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.arrs[index]

# Generators
# Parameters
params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 6}
training_set = IsingDataset(arrs)
training_generator = torch.utils.data.DataLoader(training_set, **params)

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
plot_loss = []
for epoch in tqdm(range(n_epoch)):
    for batch_id, x in enumerate(training_generator):
        loss = model(x)
        plot_loss.append(loss.data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()

print(plot_loss)
torch.save(model, "./model1")
