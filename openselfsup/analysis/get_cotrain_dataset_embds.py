from mmcv import Config
import argparse
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pdb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
from tqdm import tqdm
import copy

from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
from openselfsup.analysis.response_extractor import ResponseExtractor
from openselfsup.framework.dist_utils import get_dist_info
from openselfsup.framework.dist_utils import init_dist
from openselfsup.analysis.local_paths import COTR_MODEL_KWARGS
RESULT_FOLDER = '/mnt/fs4/chengxuz/openselfsup_models/train_dataset_outputs'


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='simclr_mlp4_sy_hctr_is112',
                        action='store', choices=COTR_MODEL_KWARGS.keys())
    parser.add_argument('--num_steps', type=int, default=20, action='store')
    parser.add_argument('--epoch_to_set', type=int, default=200, action='store')
    parser.add_argument('--cnd_dataset', action='store_true')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get embds from cotraining models using the training datasets')
    parser = add_general_argument(parser)
    return parser


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class TrainEmbdExtractor(ResponseExtractor):
    def __init__(self, epoch_to_set, cnd_dataset=False, *args, **kwargs):
        self.epoch_to_set = epoch_to_set
        self.cnd_dataset = cnd_dataset
        super().__init__(*args, **kwargs)
        self.build_train_dataloader()

    def build_train_dataloader(self):
        data_cfgs = [
                self.cfg.data['train1'],
                self.cfg.data['train2']]
        datasets = []
        for _data_cfg in data_cfgs:
            if 'data_source' in _data_cfg:
                _data_cfg['data_source']['memcached'] = False
            datasets.append(build_dataset(_data_cfg))
        self.dataset = ConcatDataset(*datasets)

        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                self.dataset, world_size, rank, 
                shuffle=False)
        loader = torch.utils.data.DataLoader(self.dataset,
                                             batch_size=128,
                                             num_workers=10,
                                             sampler=sampler,
                                             )
        self.loader = loader

    def get_outputs(self, num_steps):
        self.model.eval()
        if not self.cnd_dataset:
            for dataset in self.dataset.datasets:
                data_source = dataset.data_source
                assert hasattr(data_source, 'set_epoch')
                data_source.set_epoch(self.epoch_to_set)
        else:
            datasets = self.dataset.datasets
            assert len(datasets) == 2
            cont_data_source = datasets[0].data_source
            cont_data_source.set_epoch(self.epoch_to_set)
            cnd_acc_data_source = datasets[1].data_source
            cnd_acc_data_source.cnd_set_epoch(
                    self.epoch_to_set, self.model, 
                    datasets[0].data_source)
            self.model.eval()

        data_iter = iter(self.loader)
        ret_dict = {
                'embds': [],
                }
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)

            new_data_batch = {}
            for key in data_batch[0].keys():
                new_data_batch[key] = torch.cat(
                        [data_batch[0][key], data_batch[1][key]], dim=0)
            data_batch = new_data_batch

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
    model_kwargs = copy.deepcopy(COTR_MODEL_KWARGS[args.which_model])
    model_name_in_res = args.which_model

    embd_extractor = TrainEmbdExtractor(
            epoch_to_set=args.epoch_to_set,
            cnd_dataset=args.cnd_dataset,
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
