from ..trans_param_setter import TransParamBuilder
BASIC_TRANS_CFG = './configs/benchmarks/linear_classification/imagenet/r18_moco_np.py'


def get_typical_params(args, exp_id, ckpt_path):
    def _cfg_func(cfg):
        cfg.model.pretrained = ckpt_path
        cfg.optimizer['lr'] = 0.01
        return cfg
    param_builder = TransParamBuilder(
            args, exp_id, BASIC_TRANS_CFG, 
            col_name='trans_np_sl',
            cfg_change_func=_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params
