from mmcv import Config
import argparse
from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pdb
import torch
import pickle
from tqdm import tqdm

MODEL_INFO = {
        'moco_in': dict(
            cfg_path='./configs/selfsup/moco/r18_v2.py',
            ckpt_path='./work_dirs/selfsup/moco/r18_v2/latest.pth'),
        'moco_sy': dict(
            cfg_path='./configs/selfsup/moco/r18_v2.py',
            ckpt_path='./work_dirs/new_pipelines/moco/r18_v2_sy_ctl/latest_cached.pth'),
        'simclr_in': dict(
            cfg_path='./configs/selfsup/simclr/r18.py',
            ckpt_path='./work_dirs/new_pipelines/simclr/r18_ep300/latest_cached.pth'),
        'simclr_sy': dict(
            cfg_path='./configs/selfsup/simclr/r18.py',
            ckpt_path='./work_dirs/new_pipelines/simclr/r18_sy_ctl/latest_cached.pth'),
        'simclr_mst_ft_in': dict(
            cfg_path='./configs/selfsup/simclr/r18.py',
            ckpt_path='./work_dirs/new_pipelines/simclr_mstft/r18_in10_fx/epoch_302.pth'),
        'simclr_mst_ft_in_10': dict(
            cfg_path='./configs/selfsup/simclr/r18.py',
            ckpt_path='./work_dirs/new_pipelines/simclr_mstft/r18_in10_fx/epoch_311.pth'),
        'simclr_mst_pair_ft_in': dict(
            cfg_path='./configs/selfsup/simclr/r18.py',
            ckpt_path='./work_dirs/new_pipelines/simclr_mstft/r18_pair_in/epoch_302.pth'),
        'simclr_mst_pair_ft_sy': dict(
            cfg_path='./configs/selfsup/simclr/r18.py',
            ckpt_path='./work_dirs/new_pipelines/simclr_mstft/r18_pair_syctl/epoch_210.pth'),
        }
local_SAYCam_basedir = '/data5/chengxuz/Dataset/infant_headcam'
SAYCam_jpg_dir = os.path.join(local_SAYCam_basedir, 'jpgs_extracted')
frns_SAYCam_basedir = '/mnt/fs1/Dataset/infant_headcam'
SAYCam_num_frames_meta_path = os.path.join(
        frns_SAYCam_basedir, 'num_frames_meta.txt')
Output_base_folder = os.path.join(frns_SAYCam_basedir, 'embeddings')
SAYCam_FPS = 25


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate embeddings for SAYCam')
    parser.add_argument(
            '--which_model', 
            default=None, type=str, 
            action='store', required=True,
            choices=MODEL_INFO.keys())
    parser.add_argument(
            '--frame_offset', type=int, default=0,
            help='Offset between 0 and FPS')
    parser.add_argument(
            '--num_aug', type=int, default=4,
            help='Number of augmentations')
    return parser


class EmbedingExtractor:
    def __init__(self, args):
        self.args = args
        self.cfg = Config.fromfile(
                MODEL_INFO[args.which_model]['cfg_path'])
        self.get_dataset()
        self.get_model()

    def get_dataset(self):
        data_cfg = self.cfg.data.train
        # Assume ImageNet dataloader
        assert data_cfg.data_source.type == 'ImageNet'
        data_cfg.data_source.memcached = False
        data_cfg.type = 'ExtractDataset'
        # Flip images up-down for SAYCam
        data_cfg.pipeline.insert(
                -2, dict(type='RandomVerticalFlip', p=1))
        self.dataset = build_dataset(data_cfg)

    def get_dataloader(self, num_workers=20):
        data_loader = DataLoader(
                self.dataset,
                batch_size=256,
                shuffle=False,
                pin_memory=False,
                num_workers=num_workers)
        return data_loader

    def get_model(self):
        model = build_model(self.cfg.model)
        model_dict = torch.load(
                MODEL_INFO[self.args.which_model]['ckpt_path'])
        model.load_state_dict(model_dict['state_dict'])
        model = model.cuda()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model

    def load_meta(self):
        self.num_frames_meta = {}
        with open(SAYCam_num_frames_meta_path, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.num_frames_meta[video_name] = num_frames

    def build_fns(self):
        self.load_meta()
        self.video_idx_sta_end = {}
        fns = []
        for video_name, num_frames in self.num_frames_meta.items():
            sta = len(fns)
            fns.extend(
                    [os.path.join(
                        SAYCam_jpg_dir, video_name, 
                        '%06i.jpg' % (_frame+1))
                     for _frame in range(self.args.frame_offset, num_frames, 
                                         SAYCam_FPS)])
            end = len(fns)
            self.video_idx_sta_end[video_name] = (sta, end)
        self.dataset.data_source.fns = fns

    def get_all_embeddings(self, data_loader):
        all_embeddings = []
        for idx, data in enumerate(tqdm(data_loader)):
            imgs = data['img']
            imgs = imgs.cuda()
            embeddings = self.model(imgs, mode='test')['embd']
            all_embeddings.append(embeddings.detach().numpy())
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings

    def extract(self):
        self.build_fns()

        data_loader = self.get_dataloader()
        for aug_idx in range(self.args.num_aug):
            all_embeddings = self.get_all_embeddings(data_loader)
            for video_name, (sta, end) in tqdm(self.video_idx_sta_end.items()):
                if end > len(all_embeddings):
                    break
                _curr_embds = all_embeddings[sta:end]
                save_dir = os.path.join(
                        Output_base_folder, self.args.which_model, video_name)
                if not os.path.exists(save_dir):
                    os.system('mkdir -p ' + save_dir)
                save_path = os.path.join(
                        save_dir, 
                        'offset_%i_aug_%i.npy' % (
                            self.args.frame_offset, aug_idx))
                np.save(save_path, _curr_embds)


def main():
    parser = get_parser()
    args = parser.parse_args()

    extractor = EmbedingExtractor(args)
    extractor.extract()


if __name__ == '__main__':
    main()
