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
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')
RESULT_FOLDER = '/mnt/fs4/chengxuz/hippocampus_change/sy_embd_gates'


class HippSyEmbdRespExtractor(ResponseExtractor):
    def __init__(self, store_output=False, store_raw_pr=False, *args, **kwargs):
        self.store_output = store_output
        self.store_raw_pr = store_raw_pr
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
                                                 num_workers=10,
                                                 sampler=sampler,
                                                 )
        self.eval_val_loader = val_loader

    def build_eval_dataloader(self):
        val_nn = {
            'type': self.cfg.data.train['type'],
            'seq_len': self.cfg.data.train['seq_len'],
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_val_meta.txt'),
            'data_len': 50000,
            'which_model': self.cfg.data.train['which_model'],
            'sub_dim': self.cfg.data.train.get('sub_dim', None),
            }
        self.get_loader_from_val_nn(val_nn)

    def register_one_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if name not in target_dict:
                target_dict[name] = []
            if not isinstance(output, list):
                target_dict[name].append(output.cpu().data.numpy())
            else:
                target_dict[name].append(
                        output[0].cpu().data.numpy())

        hook = layer.register_forward_hook(hook_function)
        return hook

    def register_raw_pr_hook(self):
        layer = self.get_layer('hipp_head.rnn.mlp_softmax')
        def hook_function(_layer, _input, output, name='raw_pat_resp'):
            if name not in self.target_dict:
                self.target_dict[name] = []
            self.target_dict[name].append(
                    _input[0].cpu().data.numpy())
        hook = layer.register_forward_hook(hook_function)
        self.hooks.append(hook)

    def register_hooks(self):
        super().register_hooks()
        if self.store_output:
            layer = self.get_layer('hipp_head.rnn')
            def hook_function(_layer, _input, output, name='rnn_output'):
                if name not in self.target_dict:
                    self.target_dict[name] = []
                self.target_dict[name].append(
                        output[0][-1].cpu().data.numpy())
            hook = layer.register_forward_hook(hook_function)
            self.hooks.append(hook)
        if self.store_raw_pr:
            self.register_raw_pr_hook()

    def get_activations(self, num_steps):
        self.model.eval()
        self.register_hooks()
        data_iter = iter(self.eval_val_loader)
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            self.model(**data_batch)
        for hook in self.hooks:
            hook.remove()
        return self.target_dict


class HippSyFixLenEmbdRespExtractor(HippSyEmbdRespExtractor):
    def __init__(
            self, fixed_len, store_input, orig_input=False, *args, **kwargs):
        self.fixed_len = fixed_len
        self.store_input = store_input
        self.orig_input = orig_input
        super().__init__(*args, **kwargs)
        self.build_eval_dataloader()

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

    def store_now_inputs(self, data_batch):
        if 'input' not in self.target_dict:
            self.target_dict['input'] = []
        self.target_dict['input'].append(
                data_batch['img'].cpu().data.numpy())
        if 'input_target' not in self.target_dict:
            self.target_dict['input_target'] = []
        self.target_dict['input_target'].append(
                data_batch['target'].cpu().data.numpy())

    def get_activations(self, num_steps):
        self.model.eval()
        self.register_hooks()
        data_iter = iter(self.eval_val_loader)
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            if not self.orig_input:
                data_batch['img'] = torch.cat(
                        [data_batch['img'], data_batch['img'][:, -1:]], axis=1)
            self.model(**data_batch)
            if self.store_input:
                self.store_now_inputs(data_batch)
        for hook in self.hooks:
            hook.remove()
        return self.target_dict


def test():
    layers = [
            'hipp_head.rnn.gate_pat_resp_to_hid',
            'hipp_head.rnn.gate_pat_resp_to_hid_2']
    response_extractor = HippSyEmbdRespExtractor(
            layers=layers,
            **HIPP_MODEL_KWARGS['simple_gate'])
    act_dict = response_extractor.get_activations(5)
    pdb.set_trace()
    pass


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='simple_gate',
                        action='store', choices=HIPP_MODEL_KWARGS.keys())
    parser.add_argument('--num_steps', type=int, default=200, action='store')
    parser.add_argument('--fixed_len', type=int, default=None, action='store')
    parser.add_argument('--suffix',
                        type=str, default=None,
                        action='store', help='Name suffix')
    parser.add_argument('--store_input', action='store_true')
    parser.add_argument('--store_output', action='store_true')
    parser.add_argument('--store_raw_pr', action='store_true')
    parser.add_argument('--orig_input', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, action='store')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get gates for Hipp Sy Eval inputs')
    parser = add_general_argument(parser)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if 'ca1' not in args.which_model:
        layers = [
                'hipp_head.rnn.gate_pat_resp_to_hid',
                'hipp_head.rnn.gate_pat_resp_to_hid_2',
                'hipp_head.rnn.mlp_hidden',
                'hipp_head.rnn.mlp_hidden_2',
                'hipp_head',
                'hipp_head.rnn.mlp_softmax',
                ]
    else:
        layers = [
                'hipp_head.rnn.dg_mlp',
                'hipp_head.rnn.ca3_mlp',
                'hipp_head',
                'hipp_head.rnn.mlp_softmax',
                ]
    model_kwargs = copy.deepcopy(HIPP_MODEL_KWARGS[args.which_model])
    model_name_in_res = args.which_model
    if args.epoch is not None:
        model_kwargs['ckpt_path'] = os.path.join(
                os.path.dirname(model_kwargs['ckpt_path']),
                'epoch_{}.pth'.format(args.epoch),
                )
        model_name_in_res += '_ep{}'.format(args.epoch)

    if args.fixed_len is None:
        response_extractor = HippSyEmbdRespExtractor(
                store_output=args.store_output,
                layers=layers,
                **model_kwargs)
        result_folder = os.path.join(
                args.result_folder, model_name_in_res)
    else:
        response_extractor = HippSyFixLenEmbdRespExtractor(
                fixed_len=args.fixed_len,
                store_input=args.store_input,
                store_output=args.store_output,
                orig_input=args.orig_input,
                store_raw_pr=args.store_raw_pr,
                layers=layers,
                **model_kwargs)
        if not args.orig_input:
            result_folder = os.path.join(
                    args.result_folder, model_name_in_res, 
                    'fixlen'+str(args.fixed_len))
        else:
            result_folder = os.path.join(
                    args.result_folder, model_name_in_res, 
                    'orgip_fixlen'+str(args.fixed_len))
    act_dict = response_extractor.get_activations(args.num_steps)

    os.system('mkdir -p ' + result_folder)

    for key, value in act_dict.items():
        file_name = key
        if args.suffix is not None:
            file_name += '_' + args.suffix
        value = np.asarray(value)
        value = value.reshape([args.num_steps, -1] + list(value.shape[1:]))
        np.save(os.path.join(result_folder, file_name), value)
    print(result_folder)


if __name__ == '__main__':
    #test()
    main()
