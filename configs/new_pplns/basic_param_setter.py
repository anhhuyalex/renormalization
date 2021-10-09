import argparse
import copy
import pdb
import os
import os.path as osp
import sys
import json
import numpy as np
import logging
import time
import torch
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from mmcv import Config
from openselfsup.datasets import build_dataset
from openselfsup.models import build_model
from openselfsup.apis.train import build_optimizer, batch_processor
from openselfsup.apis import set_random_seed

import openselfsup.framework.hooks.lr_updater as lr_updater
from openselfsup.framework.dist_utils import get_dist_info, init_dist
from openselfsup.framework.utils import mkdir_or_exist, get_root_logger
from openselfsup.framework.hooks.record_saver import MongoDBSaver
from openselfsup.framework.hooks.optimizer import OptimizerHook, DistOptimizerHook
import torchvision.models as models
MODEL_SAVE_FOLDER = os.environ.get(
        'MODEL_SAVE_FOLDER',
        '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines')
SAVE_REC_TO_FILE = os.environ.get('SAVE_REC_TO_FILE', '0')
DEBUG = os.environ.get('DEBUG', '0')


class ParamsBuilder(object):
    def __init__(
            self, args, exp_id, cfg_path, 
            add_svm_val=False,
            col_name=None,
            cfg_change_func=None,
            col_name_in_work_dir=False,
            save_rec_to_file=False,
            opt_update_interval=None,
            opt_grad_clip=None,
            opt_use_fp16=False,
            model_find_unused=True,
            valid_interval=1,
            seed=None):
        self.args = args
        self.exp_id = exp_id
        self.cfg = Config.fromfile(cfg_path)
        if cfg_change_func is not None:
            self.cfg = cfg_change_func(self.cfg)
        self.params = {'max_epochs': self.cfg.total_epochs}
        self.add_svm_val = add_svm_val
        if col_name is None:
            col_name = osp.basename(osp.dirname(cfg_path))
        self.col_name = col_name
        self.col_name_in_work_dir = col_name_in_work_dir
        self.save_rec_to_file = save_rec_to_file \
                or (int(SAVE_REC_TO_FILE) == 1)
        self.opt_update_interval = opt_update_interval
        self.opt_grad_clip = opt_grad_clip
        self.opt_use_fp16 = opt_use_fp16
        self.model_find_unused = model_find_unused
        self.valid_interval = valid_interval

        if seed is not None:
            set_random_seed(seed)

    def get_save_params(self):
        if not self.col_name_in_work_dir:
            work_dir = os.path.join(MODEL_SAVE_FOLDER, self.exp_id)
        else:
            work_dir = os.path.join(
                    MODEL_SAVE_FOLDER, self.col_name, self.exp_id)
        save_params = {
                'ckpt_hook_kwargs': {
                    'interval': 10,
                    'out_dir': work_dir,
                    'cache_interval': 1,
                    },
                }
        if self.save_rec_to_file:
            save_params['record_saver_kwargs'] = {
                    'out_dir': work_dir}
        else:
            save_params['record_saver_kwargs'] = {
                    'port': 26001,
                    'database_name': 'new_unsup',
                    'collection_name': self.col_name,
                    'exp_id': self.exp_id,
                    }
            save_params['record_saver_builder'] = MongoDBSaver
        self.params['save_params'] = save_params

        rank, _ = get_dist_info()
        if rank == 0:
            mkdir_or_exist(work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(work_dir, 'train_{}.log'.format(timestamp))
        logger = get_root_logger(log_file)
        self.params['logger'] = logger

    def build_train_dataset(self):
        if 'data_source' in self.cfg.data.train:
            self.cfg.data.train['data_source']['memcached'] = False
        train_dataset = build_dataset(self.cfg.data.train)
        return train_dataset

    def get_train_data_params(self):
        train_data_params = {
                'dataset_builder': self.build_train_dataset,
                'shuffle': True,
                }
          
        num_workers = os.environ.get(
                'NUM_WORKERS', self.cfg.data.workers_per_gpu)
        batch_size = os.environ.get(
                'BATCH_SIZE', self.cfg.data.imgs_per_gpu)
        train_data_params.update({
            'batch_size': int(batch_size),
            'num_workers': int(num_workers),
            'distributed': True,
            'data_loader_kwargs': {'drop_last': True},
            })
        self.params['train_data_params'] = train_data_params

    def build_model_optimizer(self):
        self.model = build_model(self.cfg.model).cuda()
        self.optimizer = build_optimizer(
                self.model, 
                self.cfg.optimizer)
        if self.opt_use_fp16:
            import apex
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer)
        self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=self.model_find_unused,
                )
        return self.model, self.optimizer

    def get_model_optimizer_params(self):
        model_optimizer_params = {
                'builder': self.build_model_optimizer}
        self.params['model_optimizer_params'] = model_optimizer_params

    def get_loss_params(self):
        loss_params = {}
        loss_params['builder'] = lambda: None
        self.params['loss_params'] = loss_params

    def get_learning_rate_params(self):
        builder_name = self.cfg.lr_config.pop('policy')
        if builder_name == 'CosineAnealing':
            builder_name = 'CosineAnnealing'
        if builder_name == 'step':
            builder_name = 'Step'
        builder = getattr(lr_updater, builder_name + 'LrUpdaterHook')
        kwargs = self.cfg.lr_config
        learning_rate_params = {
                'builder': builder,
                'builder_kwargs': kwargs,
                }
        self.params['learning_rate_params'] = learning_rate_params

    def naive_processor(self, model, loss_func, data_batch):
        model_outputs = batch_processor(model, data_batch, True)
        return model_outputs

    def get_batch_processor_params(self):
        batch_processor_params = {
                'func': self.naive_processor,
                }
        self.params['batch_processor_params'] = batch_processor_params

    def build_val_loader_from_dataset(self, batch_size=32):
        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                self.val_dataset, world_size, rank, 
                shuffle=False)
        val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=10,
                                                 sampler=sampler,
                                                 )
        return val_loader

    def build_eval_data_loader(self):
        data_val_nn_list = 'data/imagenet/meta/part_train_val_labeled.txt'
        data_root = 'data/imagenet'
        data_source_cfg = dict(
                type='ImageNet',
                memcached=False,
                mclient_path=None)
        img_norm_cfg = dict(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Assuming first one is RandomResizedCrop
        if 'pipeline' in self.cfg.data['train']:
            input_size = self.cfg.data['train']['pipeline'][0]['size']
        elif 'pipeline1' in self.cfg.data['train']:
            input_size = self.cfg.data['train']['pipeline1'][0]['size']
        else:
            input_size = 224
        resize_size = int(256.0 * input_size / 224)
        test_pipeline = [
            dict(type='Resize', size=resize_size),
            dict(type='CenterCrop', size=input_size),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]
        val_nn=dict(
                type='NPIDNNDataset',
                data_source=dict(
                    list_file=data_val_nn_list, 
                    root=data_root, **data_source_cfg),
                pipeline=test_pipeline)

        self.val_dataset = build_dataset(val_nn)
        return self.build_val_loader_from_dataset()

    def run_model(self, model, data_batch):
        res = model(mode='test', **data_batch)
        return res

    def run_model_extract(self, model, data_batch):
        res = model(mode='extract', **data_batch)
        if len(res[0].shape) == 4:
            res = torch.mean(res[0], dim=(2, 3))
        elif len(res[0].shape) == 2:
            res = res[0]
        else:
            raise NotImplementedError
        return {'embd': res}

    def get_nn_acc(self, results):
        scores = torch.from_numpy(results['embd'])
        target = torch.LongTensor(
                self.val_dataset.data_source.labels)

        dp_results = torch.mm(scores[20000:], scores[:20000].transpose(0, 1))
        _, pred_nn = dp_results.topk(1, dim=1, largest=True, sorted=True)
        pred_nn = pred_nn.squeeze(1)
        pred_label = torch.index_select(target[:20000], 0, pred_nn)
        corr = pred_label.eq(target[20000:]).float().sum(0).item()
        acc = corr * 100.0 / 5000

        eval_res = {}
        eval_res["nn_perf"] = acc
        return eval_res

    def get_svm_acc(self, results):
        features = results['embd']
        target = self.val_dataset.data_source.labels
        alpha_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

        train_len = 20000
        train_features = features[:train_len]
        train_labels = target[:train_len]

        test_features = features[train_len:]
        test_labels = target[train_len:]

        all_perf = []
        for _alpha in tqdm(alpha_list, desc='SVM Param Search'):
            clf = SGDClassifier(alpha=_alpha, n_jobs=15)
            clf.fit(train_features, train_labels)
            _perf = clf.score(test_features, test_labels)
            all_perf.append(_perf)
        eval_res = {}
        eval_res["svm_perf"] = max(all_perf)
        return eval_res

    def get_validation_params(self):
        topn_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model,
                'agg_func': self.get_nn_acc,
                'interval': self.valid_interval,
                }
        validation_params = {'topn': topn_params}
        self.params['validation_params'] = validation_params
        if self.add_svm_val:
            self.get_SVM_validation()

    def get_SVM_validation(self):
        svm_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model_extract,
                'agg_func': self.get_svm_acc,
                'interval': self.valid_interval,
                }
        self.params['validation_params']['svm'] = svm_params

    def get_optimizer_hook_params(self):
        optimizer_hook_params = {
                'builder': OptimizerHook,
                'builder_kwargs': {
                    'grad_clip': self.opt_grad_clip,
                    }}
        if self.opt_update_interval is not None:
            assert isinstance(self.opt_update_interval, int)
            optimizer_hook_params['builder'] = DistOptimizerHook
            optimizer_hook_params['builder_kwargs'].update({
                        'update_interval': self.opt_update_interval})
        if self.opt_use_fp16:
            optimizer_hook_params['builder'] = DistOptimizerHook
            optimizer_hook_params['builder_kwargs'].update({
                        'use_fp16': self.opt_use_fp16})
        self.params['optimizer_hook_params'] = optimizer_hook_params

    def build_params(self):
        self.get_save_params()
        self.get_train_data_params()
        self.get_model_optimizer_params()
        self.get_loss_params()
        self.get_learning_rate_params()
        self.get_batch_processor_params()
        self.get_validation_params()
        self.get_optimizer_hook_params()
        if DEBUG == '1':
            self.params['validation_params'] = {}
        return self.params
