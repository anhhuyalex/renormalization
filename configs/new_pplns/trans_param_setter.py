from .basic_param_setter import ParamsBuilder
import torch
import numpy as np
from openselfsup.datasets import build_dataset


class TransParamBuilder(ParamsBuilder):
    def build_eval_data_loader(self):
        val_nn = self.cfg.data['val']
        if 'data_source' in val_nn:
            val_nn['data_source']['memcached'] = False
        self.val_dataset = build_dataset(val_nn)
        return self.build_val_loader_from_dataset(batch_size=128)

    def run_dataset_eval_func(self, results):
        scores = torch.from_numpy(results['head0'])
        return self.val_dataset.evaluate(scores, 'head')

    def get_validation_params(self):
        loss_params = {
                'data_loader_builder': self.build_eval_data_loader,
                'batch_processor': self.run_model,
                'agg_func': self.run_dataset_eval_func,
                }
        validation_params = {'loss': loss_params}
        self.params['validation_params'] = validation_params
