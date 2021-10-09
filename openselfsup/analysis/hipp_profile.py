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

from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
from openselfsup.analysis.response_extractor import ResponseExtractor
from openselfsup.framework.dist_utils import get_dist_info
from openselfsup.analysis.local_paths import HIPP_MODEL_KWARGS
from torch.profiler import profile, record_function, ProfilerActivity
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')


class HippSyEmbdProfiler(ResponseExtractor):
    def __init__(self, fixed_len=32, *args, **kwargs):
        self.fixed_len = fixed_len
        super().__init__(*args, **kwargs)
        self.build_eval_dataloader()

    def get_loader_from_val_nn(self, val_nn):
        self.val_dataset = build_dataset(val_nn)
        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                self.val_dataset, world_size, rank, 
                shuffle=False)
        val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=32,
                                                 num_workers=1,
                                                 sampler=sampler,
                                                 )
        self.eval_val_loader = val_loader

    def get_model(self):
        model = build_model(self.cfg.model)
        if self.ckpt_path is not None:
            model_dict = torch.load(self.ckpt_path)
            model.load_state_dict(model_dict['state_dict'])
        model = model.cuda()
        self.model = model

    def build_eval_dataloader(self):
        val_nn = {
            'type': 'SAYCamSeqVecDataset',
            'seq_len': self.fixed_len,
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(
                SY_DATASET_DIR, 'infant_headcam/embd_val_meta.txt'),
            'data_len': 50000,
            'which_model': self.cfg.data.train['which_model'],
            'sub_dim': self.cfg.data.train.get('sub_dim', None),
            }
        self.get_loader_from_val_nn(val_nn)

    def register_one_hook(self, layer, layer_name, target_dict):
        pass

    def register_raw_pr_hook(self):
        pass

    def register_hooks(self):
        pass

    def run_profile(self):
        self.model.eval()
        data_iter = iter(self.eval_val_loader)
        data_batch = next(data_iter)
        for key, value in data_batch.items():
            data_batch[key] = value.cuda()

        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.model(**data_batch)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=30))

    def run_multi_steps(self, num_steps=20):
        self.model.eval()
        data_iter = iter(self.eval_val_loader)
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            self.model(**data_batch)


def add_general_argument(parser):
    parser.add_argument('--which_model',
                        type=str, default='dynca1_hlf',
                        action='store', choices=HIPP_MODEL_KWARGS.keys())
    parser.add_argument('--fixed_len', type=int, default=32, action='store')
    parser.add_argument('--use_torch_profiler', action='store_true')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get gates for Hipp Sy Eval inputs')
    parser = add_general_argument(parser)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    layers = []
    model_kwargs = copy.deepcopy(HIPP_MODEL_KWARGS[args.which_model])

    model_profiler = HippSyEmbdProfiler(
            fixed_len=args.fixed_len,
            layers=layers,
            **model_kwargs)
    if args.use_torch_profiler:
        model_profiler.run_profile()
    else:
        model_profiler.run_multi_steps()


if __name__ == '__main__':
    main()
