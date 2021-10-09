import torch
import torch.nn as nn
from .registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module
class NaiveVectorDataset(Dataset):
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim

    def __len__(self):
        return 256 * 5000

    def __getitem__(self, idx):
        vector = torch.randn(self.vector_dim)
        vector = nn.functional.normalize(vector, dim=0)
        return dict(img=vector)


@DATASETS.register_module
class SeqVectorDataset(Dataset):
    def __init__(self, vector_dim=128, seq_len=32):
        self.vector_dim = vector_dim
        self.seq_len = seq_len

    def __len__(self):
        return 256 * 5000

    def __getitem__(self, idx):
        vector = torch.randn(self.seq_len, self.vector_dim)
        vector = nn.functional.normalize(vector, dim=1)
        return dict(img=vector)

@DATASETS.register_module
class RandomClusterVectorDataset(Dataset):
    def __init__(self, vector_dim=128, seq_len=8, 
                uncorr_items = 4):
        self.vector_dim = vector_dim
        self.seq_len = seq_len
        self.uncorr_items = uncorr_items
        
    def __len__(self):
        return 256 * 5000

    def __getitem__(self, idx):
        seq = []
        for _ in range(self.uncorr_items):
            self.corr_level = torch.rand(1)*3+0.25
            seq1 = torch.randn(1, self.vector_dim)
            seq2 = torch.randn(self.seq_len-1, self.vector_dim)/self.corr_level 
            seq2 += seq1
            seq.extend([seq1, seq2])
        seq = torch.cat(seq, dim = 0)
        seq = nn.functional.normalize(seq, dim=1)
        return dict(img=seq)