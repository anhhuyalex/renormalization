from .basic_param_setter import ParamsBuilder
from openselfsup.framework.hooks.hook import Hook
from openselfsup.apis.train import batch_processor
from openselfsup.datasets import build_dataset
import openselfsup.datasets.concat_datasets as concat_datasets
import torch


class SetEpochHook(Hook):
    def before_epoch(self, runner):
        data_source = runner.data_loader.dataset.data_source
        assert hasattr(data_source, 'set_epoch')
        data_source.set_epoch(runner.epoch)


class EWCHook(Hook):
    def after_epoch(self, runner):
        assert hasattr(runner.model.module, 'update_buffer')
        runner.model.module.update_buffer(runner.data_loader)


class SAYCamParamBuilder(ParamsBuilder):
    def __init__(self, need_ewc_hook=False, *args, **kwargs):
        self.need_ewc_hook = need_ewc_hook
        super().__init__(*args, **kwargs)

    def add_one_hook_params(self, one_hook_params):
        if 'extra_hook_params' not in self.params:
            self.params['extra_hook_params'] = []
        self.params['extra_hook_params'].append(one_hook_params)

    def add_set_epoch_hook(self):
        set_epoch_hook_params = {'builder': SetEpochHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def add_ewc_hook(self):
        set_epoch_hook_params = {'builder': EWCHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def build_params(self):
        super().build_params()
        self.add_set_epoch_hook()
        if self.need_ewc_hook:
            self.add_ewc_hook()
        self.params['train_data_params']['shuffle'] = False
        return self.params


class ConcatSetEpochHook(Hook):
    def before_epoch(self, runner):
        datasets = runner.data_loader.dataset.datasets
        for dataset in datasets:
            data_source = dataset.data_source
            assert hasattr(data_source, 'set_epoch')
            data_source.set_epoch(runner.epoch)


class CndSetEpochHook(Hook):
    def before_epoch(self, runner):
        datasets = runner.data_loader.dataset.datasets
        assert len(datasets) == 2
        cont_data_source = datasets[0].data_source
        cont_data_source.set_epoch(runner.epoch)
        cnd_acc_data_source = datasets[1].data_source
        cnd_acc_data_source.cnd_set_epoch(
                runner.epoch, runner.model, 
                datasets[0].data_source)


class CotrainSAYCamParamBuilder(SAYCamParamBuilder):
    def __init__(
            self, mix_weight, use_cnd_hook=False, 
            concat_batches=False,
            scale_ratio=None,
            *args, **kwargs):
        self.mix_weight = mix_weight
        self.use_cnd_hook = use_cnd_hook
        self.concat_batches = concat_batches
        self.scale_ratio = scale_ratio
        super().__init__(*args, **kwargs)

    def add_set_epoch_hook(self):
        if not self.use_cnd_hook:
            set_epoch_hook_params = {'builder': ConcatSetEpochHook}
        else:
            set_epoch_hook_params = {'builder': CndSetEpochHook}
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
        if self.scale_ratio is None:
            train_dataset = concat_datasets.ConcatDataset(*datasets)
        else:
            train_dataset = concat_datasets.ScaledConcatDataset(
                    self.scale_ratio, *datasets)
        return train_dataset

    def naive_processor(self, model, loss_func, data_batch):
        if not self.concat_batches:
            assert len(data_batch) == 2
            model_outputs0 = batch_processor(model, data_batch[0], True)
            model_outputs1 = batch_processor(model, data_batch[1], True)
            model_outputs = dict(
                    loss=model_outputs0['loss']*self.mix_weight \
                         + model_outputs1['loss'],
                    log_vars0=model_outputs0['log_vars'],
                    log_vars1=model_outputs1['log_vars'],
                    num_samples=model_outputs0['num_samples'],
                    )
        else:
            new_data_batch = {}
            for key in data_batch[0].keys():
                new_data_batch[key] = torch.cat(
                        [_data_batch[key] for _data_batch in data_batch], 
                        dim=0)
            model_outputs = batch_processor(model, new_data_batch, True)
        return model_outputs
