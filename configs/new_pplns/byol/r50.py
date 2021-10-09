from ..basic_param_setter import ParamsBuilder
from .byol_hook import BYOLHook
from .r18 import add_byol_hook_to_params
BASIC_BYOL_CFG = './configs/selfsup/byol/r50_bs256_accumulate16_ep300.py'


def get_typical_params(args, exp_id, cfg_func):
    def _change_opt_inter(cfg):
        cfg = cfg_func(cfg)
        cfg.optimizer_config['update_interval'] = 32
        return cfg
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_BYOL_CFG, 
            add_svm_val=True,
            col_name='byol',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            )
    params = param_builder.build_params()
    params = add_byol_hook_to_params(params, 32)
    return params


def r50_ep300(args):
    return get_typical_params(
            args, 'r50', 
            cfg_func=lambda cfg: cfg)
