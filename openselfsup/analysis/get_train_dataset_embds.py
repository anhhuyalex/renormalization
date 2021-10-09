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
from openselfsup.analysis.local_paths import MODEL_KWARGS
RESULT_FOLDER = '/mnt/fs4/chengxuz/openselfsup_models/train_dataset_outputs'


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='simclr_in_ctl',
                        action='store', choices=MODEL_KWARGS.keys())
    parser.add_argument('--num_steps', type=int, default=20, action='store')
    parser.add_argument('--shuffle_dataset', action='store_true')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get embds from models using the training datasets')
    parser = add_general_argument(parser)
    return parser


class TrainEmbdExtractor(ResponseExtractor):
    def __init__(self, shuffle_dataset, *args, **kwargs):
        self.shuffle_dataset = shuffle_dataset
        super().__init__(*args, **kwargs)
        self.build_train_dataloader()

    def build_train_dataloader(self):
        self.dataset = build_dataset(self.cfg.data['train'])
        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                self.dataset, world_size, rank, 
                shuffle=self.shuffle_dataset)
        loader = torch.utils.data.DataLoader(self.dataset,
                                             batch_size=256,
                                             num_workers=10,
                                             sampler=sampler,
                                             )
        self.loader = loader

    def get_outputs(self, num_steps):
        self.model.eval()
        data_iter = iter(self.loader)
        ret_dict = {
                'embds': [],
                }
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            data_batch['img'] = data_batch['img'][:, 0]
            embds = self.model(**data_batch, mode='test')
            ret_dict['embds'].append(embds['embd'].cpu().data.numpy())
        return ret_dict


def main():
    parser = get_parser()
    args = parser.parse_args()

    layers = []
    model_kwargs = copy.deepcopy(MODEL_KWARGS[args.which_model])
    model_name_in_res = args.which_model

    embd_extractor = TrainEmbdExtractor(
            shuffle_dataset=args.shuffle_dataset,
            layers=layers,
            **model_kwargs)
    result_folder = os.path.join(
            args.result_folder, model_name_in_res)
    embd_dict = embd_extractor.get_outputs(args.num_steps)

    os.system('mkdir -p ' + result_folder)
    for key, value in embd_dict.items():
        file_name = key
        value = np.asarray(value)
        value = value.reshape([args.num_steps, -1] + list(value.shape[1:]))
        np.save(os.path.join(result_folder, file_name), value)
    print(result_folder)


if __name__ == '__main__':
    main()
