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


class SetEpochHook(Hook):
    def before_epoch(self, runner):
        dataset = runner.data_loader.dataset
        assert hasattr(dataset, 'set_epoch')
        dataset.set_epoch(runner.epoch)


class MSTParamBuilder(ParamsBuilder):
    def __init__(self, **kwargs):
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
            res[key] = res[key].unsqueeze(0).repeat(64)
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
                                                 batch_size=64,
                                                 num_workers=10,
                                                 sampler=sampler,
                                                 )
        return val_loader

    def build_eval_data_loader(self):
        val_nn = copy.copy(self.cfg.data.train)
        val_nn['data_len'] = 64*780
        val_loader = self.get_loader_from_val_nn(val_nn)
        return val_loader


    def get_validation_params(self):
        loss_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model_to_get_loss,
                'agg_func': self.get_ave_loss,
                }
        validation_params = {'loss': loss_params}

        self.params['validate_hook'] = JumpStop_ValidateHook
        self.params['validation_params'] = validation_params

    def build_params(self):
        super().build_params()
        # self.add_set_epoch_hook()
        # self.params['train_data_params']['shuffle'] = False
        return self.params
