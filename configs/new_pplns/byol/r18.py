from ..basic_param_setter import ParamsBuilder
from .byol_hook import BYOLHook, EWCBYOLHook
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.byol_neg_funcs as byol_neg_funcs
import openselfsup.config_related.byol_funcs as byol_funcs
BASIC_BYOL_CFG = './configs/selfsup/byol/r18.py'


def add_byol_hook_to_params(
        params, update_interval=1, 
        use_ewc_hook=False):
    byol_hook_params = {
            'builder': BYOLHook,
            'update_interval': update_interval,
            }
    if use_ewc_hook:
        byol_hook_params['builder'] = EWCBYOLHook
    if 'extra_hook_params' not in params:
        params['extra_hook_params'] = byol_hook_params
    else:
        if isinstance(params['extra_hook_params'], dict):
            params['extra_hook_params'] = [params['extra_hook_params']]
        params['extra_hook_params'].append(byol_hook_params)
    return params


def r18_with_svm(args):
    param_builder = ParamsBuilder(
            args, 'byol_r18', BASIC_BYOL_CFG, 
            add_svm_val=True)
    params = param_builder.build_params()
    params = add_byol_hook_to_params(params)
    return params


BASIC_BYOL_EP300_CFG = './configs/selfsup/byol/r18_ep300.py'
def get_typical_ep300_params(args, exp_id, cfg_func, seed=None):
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_BYOL_EP300_CFG, 
            add_svm_val=True,
            col_name='byol',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            opt_update_interval=16,
            seed=seed,
            )
    params = param_builder.build_params()
    params = add_byol_hook_to_params(params, 16)
    return params


def r18_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ep300', 
            cfg_func=lambda cfg: cfg)


def r18_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s1',
            cfg_func=lambda cfg: cfg,
            seed=1)


def r18_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s2',
            cfg_func=lambda cfg: cfg,
            seed=2)


def r18_mlp4_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_mlp4_ep300',
            cfg_func=byol_funcs.mlp_4layers_cfg_func)


def r18_ws_gn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ws_gn_ep300_2', 
            cfg_func=ws_gn_funcs.byol_ws_gn_cfg_func)


def r18_neg_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_neg_ep300_3', 
            cfg_func=byol_neg_funcs.byol_neg_cfg_func)


def r18_neg_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_neg_ep300_s1',
            cfg_func=byol_neg_funcs.byol_neg_cfg_func,
            seed=1)


def r18_neg_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_neg_ep300_s2',
            cfg_func=byol_neg_funcs.byol_neg_cfg_func,
            seed=2)
