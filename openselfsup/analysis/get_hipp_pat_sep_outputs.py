from mmcv import Config
import argparse
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pdb
import torch
import pickle
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
from openselfsup.framework.dist_utils import get_dist_info
from openselfsup.analysis.local_paths import PAT_SEP_MODEL_KWARGS
from openselfsup.analysis.response_extractor import build_load_model
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')
RESULT_FOLDER = '/mnt/fs4/chengxuz/hippocampus_change/pat_sep_outputs'


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='aha_dg',
                        action='store')
    parser.add_argument('--num_steps', type=int, default=100, action='store')
    parser.add_argument('--seq_len', type=int, default=32, action='store')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get outputs from VAE models')
    parser = add_general_argument(parser)
    return parser


class PatSepOutputExtractor:
    def __init__(self, args):
        self.args = args
        self.batch_size = 32
        self.build_eval_dataloader()
        self.build_model()

    def build_model(self):
        if self.args.which_model in ['aha_dg', 'aha_dg_more']:
            from openselfsup.models.aha_modules.dg import DG
            dg_config = {
                "inhibition_decay": 0.95,
                "knockout_rate": 0.25,
                "init_scale": 10.0,
                "num_units": 225,
                "sparsity": 10,
                "use_stub": False,
            }
            if self.args.which_model == 'aha_dg_more':
                dg_config['num_units'] = 1000
                dg_config['sparsity'] = 50
            self.model = DG(
                    [self.batch_size * (self.args.seq_len+1), 128], 
                    dg_config)
            self.model = self.model.cuda()
        elif self.args.which_model == 'random_mlp':
            hidden_size = 256
            mlp_hidden =  nn.Sequential(
                    nn.Linear(128, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 256),
                    )
            stdv = 1.0 / math.sqrt(hidden_size)
            for weight in mlp_hidden.parameters():
                init.normal_(weight, 0, stdv)
            self.model = mlp_hidden.cuda()
        elif self.args.which_model in PAT_SEP_MODEL_KWARGS:
            _kwargs = PAT_SEP_MODEL_KWARGS[self.args.which_model]
            cfg = Config.fromfile(_kwargs['cfg_path'])
            cfg = _kwargs['cfg_func'](cfg)
            self.model = build_load_model(cfg, _kwargs['ckpt_path'])
        else:
            raise NotImplementedError

    def get_loader_from_val_nn(self, val_nn):
        self.val_dataset = build_dataset(val_nn)
        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                self.val_dataset, world_size, rank, 
                shuffle=False)
        val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=10,
                                                 sampler=sampler,
                                                 )
        self.eval_val_loader = val_loader

    def build_eval_dataloader(self):
        val_nn = {
            'type': 'SAYCamSeqVecDataset',
            'seq_len': self.args.seq_len,
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(
                SY_DATASET_DIR, 'infant_headcam/embd_val_meta.txt'),
            'data_len': 50000,
            'which_model': 'simclr_mst_pair_ft_in',
            }
        self.get_loader_from_val_nn(val_nn)

    def get_outputs(self, num_steps):
        self.model.eval()
        data_iter = iter(self.eval_val_loader)
        ret_dict = {
                'pattern': [],
                'input': [],
                }
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            if not self.args.which_model in PAT_SEP_MODEL_KWARGS:
                pattern = self.model(data_batch['img'].reshape(
                    -1, 128))
            else:
                pattern = self.model(data_batch['img'], mode='extract')

            ret_dict['input'].append(data_batch['img'].cpu().data.numpy())
            ret_dict['pattern'].append(
                    pattern.reshape(
                        self.batch_size, self.args.seq_len+1, -1
                        ).cpu().data.numpy())
        return ret_dict


def main():
    parser = get_parser()
    args = parser.parse_args()


def main():
    parser = get_parser()
    args = parser.parse_args()

    response_extractor = PatSepOutputExtractor(args=args)
    result_folder = os.path.join(
            args.result_folder, args.which_model)
    act_dict = response_extractor.get_outputs(args.num_steps)

    os.system('mkdir -p ' + result_folder)
    for key, value in act_dict.items():
        file_name = key
        value = np.asarray(value)
        value = value.reshape([args.num_steps, -1] + list(value.shape[1:]))
        np.save(os.path.join(result_folder, file_name), value)
    print(result_folder)


if __name__ == '__main__':
    main()
