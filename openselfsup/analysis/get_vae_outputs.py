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
from openselfsup.analysis.local_paths import VAE_MODEL_KWARGS
RESULT_FOLDER = '/mnt/fs4/chengxuz/openselfsup_models/vae_outputs'


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='default_vae',
                        action='store', choices=VAE_MODEL_KWARGS.keys())
    parser.add_argument('--num_steps', type=int, default=20, action='store')
    parser.add_argument('--epoch', type=int, default=None, action='store')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get outputs from VAE models')
    parser = add_general_argument(parser)
    return parser


class VAEOutputExtractor(ResponseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_train_dataloader()

    def build_train_dataloader(self):
        self.dataset = build_dataset(self.cfg.data['train'])
        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                self.dataset, world_size, rank, 
                shuffle=False)
        loader = torch.utils.data.DataLoader(self.dataset,
                                             batch_size=32,
                                             num_workers=10,
                                             sampler=sampler,
                                             )
        self.loader = loader

    def get_outputs(self, num_steps):
        self.model.eval()
        data_iter = iter(self.loader)
        ret_dict = {
                'recon': [],
                'input': [],
                }
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            recon = self.model(mode='recon', **data_batch)
            ret_dict['input'].append(data_batch['img'].cpu().data.numpy())
            ret_dict['recon'].append(recon.cpu().data.numpy())
        return ret_dict


class InterVAEOutputExtractor(VAEOutputExtractor):
    def get_outputs(self, num_steps):
        self.model.eval()
        data_iter = iter(self.loader)
        ret_dict = {
                'inter_recon': [],
                'inter_input': [],
                'input': [],
                }
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            recon_dict = self.model(mode='recon', **data_batch)
            ret_dict['input'].append(data_batch['img'].cpu().data.numpy())
            ret_dict['inter_recon'].append(recon_dict['x_recon'].cpu().data.numpy())
            ret_dict['inter_input'].append(recon_dict['x'].cpu().data.numpy())
        return ret_dict


class InterBNVAEOutputExtractor(VAEOutputExtractor):
    def get_outputs(self, num_steps):
        self.model.eval()
        data_iter = iter(self.loader)
        ret_dict = {
                'inter_recon': [],
                'inter_input': [],
                'input': [],
                'recon': [],
                }
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            recon_dict = self.model(mode='recon', **data_batch)
            ret_dict['input'].append(recon_dict['input'].cpu().data.numpy())
            ret_dict['recon'].append(recon_dict['recon'].cpu().data.numpy())
            ret_dict['inter_recon'].append(recon_dict['x_recon'].cpu().data.numpy())
            ret_dict['inter_input'].append(recon_dict['x'].cpu().data.numpy())
        return ret_dict


def color_denorm(images):
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
    std = std[np.newaxis, :, np.newaxis, np.newaxis]
    images = images * std + mean
    images = np.clip(images, 0, 1)
    images = images * 255
    images = images.astype(np.uint8)
    return images


def main():
    parser = get_parser()
    args = parser.parse_args()

    layers = []
    model_kwargs = copy.deepcopy(VAE_MODEL_KWARGS[args.which_model])
    model_name_in_res = args.which_model
    if args.epoch is not None:
        model_kwargs['ckpt_path'] = os.path.join(
                os.path.dirname(model_kwargs['ckpt_path']),
                'epoch_{}.pth'.format(args.epoch),
                )
        model_name_in_res += '_ep{}'.format(args.epoch)

    if 'inter_vae' in args.which_model:
        response_extractor = InterVAEOutputExtractor(
                layers=layers,
                **model_kwargs)
    elif 'interbn_vae' in args.which_model:
        response_extractor = InterBNVAEOutputExtractor(
                layers=layers,
                **model_kwargs)
    else:
        response_extractor = VAEOutputExtractor(
                layers=layers,
                **model_kwargs)
    result_folder = os.path.join(
            args.result_folder, model_name_in_res)
    act_dict = response_extractor.get_outputs(args.num_steps)

    os.system('mkdir -p ' + result_folder)
    for key, value in act_dict.items():
        file_name = key
        value = np.asarray(value)
        value = value.reshape([args.num_steps, -1] + list(value.shape[1:]))
        if 'inter' not in key:
            value = color_denorm(value)
        np.save(os.path.join(result_folder, file_name), value)
    print(result_folder)


if __name__ == '__main__':
    main()
