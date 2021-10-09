from .basic_param_setter import ParamsBuilder
from .jmpstp_valid_hook import JumpStop_ValidateHook
from openselfsup.framework.hooks.hook import Hook
from openselfsup.datasets import build_dataset
from openselfsup.framework.dist_utils import get_dist_info
import torch
import numpy as np
import os
import copy
from openselfsup.datasets.loader.sampler import DistributedSampler
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')
VAL_BATCH_SIZE = int(os.environ.get(
        'VAL_BATCH_SIZE', 64))


class SetEpochHook(Hook):
    def before_epoch(self, runner):
        dataset = runner.data_loader.dataset
        assert hasattr(dataset, 'set_epoch')
        dataset.set_epoch(runner.epoch)


class SAYCamEmbdParamBuilder(ParamsBuilder):
    def __init__(self, vary_len_val=None, **kwargs):
        self.vary_len_val = vary_len_val
        super().__init__(**kwargs)

    def add_one_hook_params(self, one_hook_params):
        if 'extra_hook_params' not in self.params:
            self.params['extra_hook_params'] = []
        self.params['extra_hook_params'].append(one_hook_params)

    def add_set_epoch_hook(self):
        set_epoch_hook_params = {'builder': SetEpochHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def run_model_to_get_loss(self, model, data_batch):
        res = model(mode='train', **data_batch)
        for key in res:
            res[key] = res[key].unsqueeze(0).repeat(VAL_BATCH_SIZE)
        return res

    def get_ave_loss(self, results):
        new_results = {}
        for key, value in results.items():
            new_value = np.mean(value)
            new_results[key] = new_value
        return new_results

    def get_loader_from_val_nn(self, val_nn):
        val_dataset = build_dataset(val_nn)
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(
                val_dataset, world_size, rank, 
                shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=VAL_BATCH_SIZE,
                                                 num_workers=10,
                                                 sampler=sampler,
                                                 drop_last=True,
                                                 )
        return val_loader

    def build_eval_data_loader(self):
        val_nn = {
            'type': self.cfg.data.train['type'],
            'seq_len': self.cfg.data.train['seq_len'],
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_val_meta.txt'),
            'data_len': 64*784,
            'which_model': self.cfg.data.train['which_model'],
            'sub_dim': self.cfg.data['train'].get('sub_dim', None),
            }
        if val_nn['type'] in [
                'VaryLenSAYCamSeqVec', 'FilterVLenSCSeqVec',
                'OSFilterVLenSCSeqVec']:
            val_nn['min_seq_len'] = self.cfg.data.train['min_seq_len']
            val_nn['batch_size'] = VAL_BATCH_SIZE

        val_loader = self.get_loader_from_val_nn(val_nn)
        return val_loader

    def build_fix_len_eval_data_loader(self, seq_len):
        val_nn = {
            'type': 'SAYCamSeqVecDataset',
            'seq_len': seq_len,
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_val_meta.txt'),
            'data_len': 64*784,
            'which_model': self.cfg.data.train['which_model'],
            'sub_dim': self.cfg.data['train'].get('sub_dim', None),
            }
        val_loader = self.get_loader_from_val_nn(val_nn)
        return val_loader

    def get_validation_params(self):
        loss_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model_to_get_loss,
                'agg_func': self.get_ave_loss,
                'interval': self.valid_interval,
                'initial': False,
                }
        validation_params = {'loss': loss_params}
        if self.vary_len_val is None:
            self.params['validate_hook'] = JumpStop_ValidateHook
        else:
            # vary_len_val should be a sequence of length
            for seq_len in self.vary_len_val:
                loss_params = {
                        'data_loader_builder': self.build_fix_len_eval_data_loader,
                        'data_loader_builder_kwargs': {'seq_len': seq_len},
                        'batch_processor': self.run_model_to_get_loss,
                        'agg_func': self.get_ave_loss,
                        'interval': self.valid_interval,
                        'initial': False,
                        }
                validation_params['loss_'+str(seq_len)] = loss_params
        self.params['validation_params'] = validation_params

    def build_params(self):
        super().build_params()
        self.add_set_epoch_hook()
        self.params['train_data_params']['shuffle'] = False
        return self.params
