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
from openselfsup.analysis.get_hipp_sy_embd_response \
        import HippSyFixLenEmbdRespExtractor
from openselfsup.framework.dist_utils import get_dist_info
from openselfsup.analysis.local_paths import HIPP_MODEL_KWARGS
from openselfsup.analysis.get_responses_for_mst_stim \
        import RESULT_FOLDER as MST_EMBD_RESULT_FOLDER
from openselfsup.analysis.get_responses_for_mst_stim \
        import STIM_FOLDER as MST_STIM_FOLDER
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')
RESULT_FOLDER = '/mnt/fs4/chengxuz/hippocampus_change/mst_sy_embd_gates'


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='simple_gate_ft_vlen_rth',
                        action='store', choices=HIPP_MODEL_KWARGS.keys())
    parser.add_argument('--num_steps', type=int, default=50, action='store')
    parser.add_argument('--suffix',
                        type=str, default=None,
                        action='store', help='Name suffix')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get gates for Hipp Sy models on MST inputs')
    parser = add_general_argument(parser)
    return parser


class MSTSeqHippEmbdRespExtractor(HippSyFixLenEmbdRespExtractor):
    def __init__(self, mst_base_embd_path, *args, **kwargs):
        self.mst_base_embd = np.load(mst_base_embd_path)
        super().__init__(
                fixed_len=None,
                store_input=True,
                orig_input=True,
                store_output=False,
                store_raw_pr=True,
                *args, **kwargs)

    def build_eval_dataloader(self):
        val_nn = {
            'type': 'MSTSynthVectorDataset',
            'root': MST_STIM_FOLDER,
            'real_embds': self.mst_base_embd,
            }
        self.get_loader_from_val_nn(val_nn)

    def get_activations(self, num_steps):
        self.model.eval()
        self.register_hooks()
        data_iter = iter(self.eval_val_loader)
        for _ in tqdm(range(num_steps)):
            data_batch = next(data_iter)
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            model_outputs = self.model(mode='mst', **data_batch)
            if self.store_input:
                self.store_now_inputs(data_batch)
            if 'model_outputs' not in self.target_dict:
                self.target_dict['model_outputs'] = []
            self.target_dict['model_outputs'].append(
                    model_outputs.cpu().data.numpy())
        for hook in self.hooks:
            hook.remove()
        return self.target_dict


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
    model_kwargs = copy.deepcopy(HIPP_MODEL_KWARGS[args.which_model])
    model_name_in_res = args.which_model

    mst_base_embd_path = os.path.join(
            MST_EMBD_RESULT_FOLDER, 'simclr_mst_pair_ft_in/Set_1/aug_eval.npy')
    response_extractor = MSTSeqHippEmbdRespExtractor(
            layers=layers,
            mst_base_embd_path=mst_base_embd_path,
            **model_kwargs)
    result_folder = os.path.join(
            args.result_folder, model_name_in_res)
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
    main()
