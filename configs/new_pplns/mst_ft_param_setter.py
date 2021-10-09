from .saycam_param_setter import \
        SAYCamParamBuilder, ConcatSetEpochHook, ConcatDataset
from openselfsup.framework.hooks.hook import Hook
from openselfsup.apis.train import batch_processor
from openselfsup.datasets import build_dataset
from openselfsup.framework.dist_utils import get_dist_info
import torch
import numpy as np


class CotrainMSTSYParamBuilder(SAYCamParamBuilder):
    def __init__(self, mix_weight, *args, **kwargs):
        self.mix_weight = mix_weight
        super().__init__(*args, **kwargs)

    def add_set_epoch_hook(self):
        set_epoch_hook_params = {'builder': ConcatSetEpochHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def build_train_dataset(self):
        data_cfgs = [
                self.cfg.data['train1'],
                self.cfg.data['train2']]
        datasets = []
        for _data_cfg in data_cfgs:
            if 'data_source' in _data_cfg:
                _data_cfg['data_source']['memcached'] = False
            datasets.append(build_dataset(_data_cfg))
        train_dataset = ConcatDataset(*datasets)
        return train_dataset

    def naive_processor(self, model, loss_func, data_batch):
        model_outputs0 = batch_processor(model, data_batch[0], True)
        model_outputs1 = batch_processor(model, data_batch[1], True)
        model_outputs = dict(
                loss=model_outputs0['loss']*self.mix_weight \
                     + model_outputs1['loss'],
                log_vars0=model_outputs0['log_vars'],
                log_vars1=model_outputs1['log_vars'],
                num_samples=model_outputs0['num_samples'],
                )
        return model_outputs

    def get_nn_acc(self, results):
        embds = results['embd']
        cross_sim = np.matmul(embds, embds.transpose(1, 0))
        corr = []
        for idx in range(0, len(embds), 2):
            lure_idx = idx+1
            sec_clst_idx = np.argsort(cross_sim[idx, :])[-2]
            if sec_clst_idx == lure_idx:
                corr.append(1)
            else:
                corr.append(0)
        acc = np.mean(corr)
        eval_res = {}
        eval_res["nn_perf"] = acc
        return eval_res

    def build_mst_eval_data_loader(self):
        img_norm_cfg = dict(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_pipeline = [
            dict(type='Resize', size=256),
            dict(type='CenterCrop', size=224),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]
        val_nn=dict(
                type='NPIDNNDataset',
                data_source=dict(
                    type='MSTImageList',
                    root='/mnt/fs4/chengxuz/hippocampus_change/mst_related/MST',
                    which_set='Set 1',
                    ),
                pipeline=test_pipeline)
        val_dataset = build_dataset(val_nn)

        rank, world_size = get_dist_info()
        from openselfsup.datasets.loader.sampler import DistributedSampler
        sampler = DistributedSampler(
                val_dataset, world_size, rank, 
                shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=24,
                                                 num_workers=0,
                                                 sampler=sampler,
                                                 )
        return val_loader

    def get_validation_params(self):
        topn_params = {
                'data_loader_builder': self.build_mst_eval_data_loader,
                'batch_processor': self.run_model,
                'agg_func': self.get_nn_acc,
                'by_epoch': False,
                'interval': 100,
                }
        validation_params = {'mst_topn': topn_params}
        self.params['validation_params'] = validation_params
        self.get_SVM_validation()

    def get_SVM_validation(self):
        svm_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model_extract,
                'agg_func': self.get_svm_acc,
                'by_epoch': False,
                'interval': 1000,
                }
        self.params['validation_params']['svm'] = svm_params
