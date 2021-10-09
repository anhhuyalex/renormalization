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
RESULT_FOLDER = '/home/an633/project/CuriousContrast/results_alex/mst_analysis'


class HippSyEmbdRespExtractorMST(ResponseExtractor):
    def __init__(self, store_output=False, *args, **kwargs):
        self.store_output = store_output
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
#         val_nn = {
#             'type': self.cfg.data.train['type'],
#             'seq_len': self.cfg.data.train['seq_len'],
#             'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
#             'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_val_meta.txt'),
#             'data_len': 50000,
#             'which_model': self.cfg.data.train['which_model'],
#             }
        val_nn = copy.copy(self.cfg.data.train)
        val_nn['data_len'] = 64*780
        self.get_loader_from_val_nn(val_nn)

    def register_one_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if name not in target_dict:
                target_dict[name] = []
            target_dict[name].append(output.cpu().data.numpy())

        hook = layer.register_forward_hook(hook_function)
        return hook

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
    parser.add_argument('--suffix',
                        type=str, default=None,
                        action='store', help='Name suffix')
    parser.add_argument('--store_input', action='store_true')
    parser.add_argument('--store_output', action='store_true')
    parser.add_argument('--orig_input', action='store_true')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get gates for Hipp Sy Eval inputs')
    parser = add_general_argument(parser)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    layers = [
            'hipp_head.rnn.gate_pat_resp_to_hid',
            'hipp_head.rnn.gate_pat_resp_to_hid_2',
            'hipp_head.rnn.mlp_hidden',
            'hipp_head.rnn.mlp_hidden_2',
            'hipp_head',
            'hipp_head.rnn.mlp_softmax',
            ]
    response_extractor = HippSyEmbdRespExtractorMST(
            store_output=args.store_output,
            layers=layers,
            **HIPP_MODEL_KWARGS[args.which_model])
    result_folder = os.path.join(
            args.result_folder, args.which_model)
    
    act_dict = response_extractor.get_activations(args.num_steps)

    os.system('mkdir -p ' + result_folder)
    print("act_dict keys", list(act_dict.keys()))
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
