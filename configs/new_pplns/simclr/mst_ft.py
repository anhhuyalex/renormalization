from ..mst_ft_param_setter import CotrainMSTSYParamBuilder
import openselfsup.config_related.mst_ft_funcs as mst_ft_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_mst_ft_params(
        args, exp_id, cfg_func,
        opt_update_interval=16, mix_weight=1.0,
        seed=None):
    param_builder = CotrainMSTSYParamBuilder(
            mix_weight=mix_weight,
            args=args, exp_id=exp_id, 
            cfg_path=BASIC_SIMCLR_CFG, 
            add_svm_val=True, col_name='simclr_mstft',
            cfg_change_func=cfg_func,
            opt_update_interval=opt_update_interval,
            col_name_in_work_dir=True,
            seed=seed)
    params = param_builder.build_params()
    return params


def set_model_load(params):
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/simclr/r18_sy_ctl/latest_cached.pth',
            }
    params['max_epochs'] = 400
    return params


def mst_ft_from_syctl(args):
    params = get_typical_mst_ft_params(
            args, 'r18_syctl', 
            lambda cfg: mst_ft_funcs.cotrain_syctl_mst_cfg_func(cfg, 1e-2),
            opt_update_interval=None)
    params = set_model_load(params)
    return params


def mst_ft_from_syctl10(args):
    params = get_typical_mst_ft_params(
            args, 'r18_syctl10', 
            lambda cfg: mst_ft_funcs.cotrain_syctl_mst_cfg_func(cfg, 1e-1),
            opt_update_interval=None)
    params = set_model_load(params)
    return params


def mst_pair_ft_from_syctl(args):
    params = get_typical_mst_ft_params(
            args, 'r18_pair_syctl', 
            lambda cfg: mst_ft_funcs.cotrain_syctl_mst_pair_cfg_func(cfg, 1e-2),
            opt_update_interval=None)
    params = set_model_load(params)
    params['save_params']['ckpt_hook_kwargs']['interval'] = 1
    return params


def mst_pair_ft_from_syctl10(args):
    params = get_typical_mst_ft_params(
            args, 'r18_pair_syctl10', 
            lambda cfg: mst_ft_funcs.cotrain_syctl_mst_pair_cfg_func(cfg, 1e-1),
            opt_update_interval=None)
    params = set_model_load(params)
    params['save_params']['ckpt_hook_kwargs']['interval'] = 1
    return params


def set_in_model_load(params):
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/simclr/r18_ep300/latest_cached.pth',
            }
    params['max_epochs'] = 400
    return params


def mst_ft_from_in(args):
    params = get_typical_mst_ft_params(
            args, 'r18_in_fx', 
            gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                lambda cfg: mst_ft_funcs.cotrain_syctl_mst_cfg_func(cfg, 1e-2),
                ),
            opt_update_interval=None)
    params = set_in_model_load(params)
    params['save_params']['ckpt_hook_kwargs']['interval'] = 1
    return params


def mst_ft_from_in10(args):
    params = get_typical_mst_ft_params(
            args, 'r18_in10_fx', 
            gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                lambda cfg: mst_ft_funcs.cotrain_syctl_mst_cfg_func(cfg, 1e-1),
                ),
            opt_update_interval=None)
    params = set_in_model_load(params)
    params['save_params']['ckpt_hook_kwargs']['interval'] = 1
    return params


def mst_pair_ft_from_in(args):
    params = get_typical_mst_ft_params(
            args, 'r18_pair_in', 
            gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                lambda cfg: mst_ft_funcs.cotrain_syctl_mst_pair_cfg_func(cfg, 1e-2),
                ),
            opt_update_interval=None)
    params = set_in_model_load(params)
    params['save_params']['ckpt_hook_kwargs']['interval'] = 1
    return params


def mst_pair_ft_from_in10(args):
    params = get_typical_mst_ft_params(
            args, 'r18_pair_in10', 
            gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                lambda cfg: mst_ft_funcs.cotrain_syctl_mst_pair_cfg_func(cfg, 1e-1),
                ),
            opt_update_interval=None)
    params = set_in_model_load(params)
    params['save_params']['ckpt_hook_kwargs']['interval'] = 1
    return params
