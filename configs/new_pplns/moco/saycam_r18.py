from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.moco.r18_funcs as r18_funcs
from .r18 import add_moco_hook_to_params
BASIC_MOCO_CFG = './configs/selfsup/moco/r18_v2_with_val.py'


def get_typical_params(args, exp_id, cfg_func):
    cfg_path = BASIC_MOCO_CFG
    param_builder = SAYCamParamBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_sy_ctl(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_v2_sy_ctl', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=saycam_funcs.random_saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_sy_order(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_v2_sy_order_fx', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=saycam_funcs.saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_sy_order_hp(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_v2_sy_order_hp', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=saycam_funcs.hipp_pred_saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_sy_ctl_two_img(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_v2_sy_ctl_two_img', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=saycam_funcs.random_saycam_two_img_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_sy_ctl_two_img_rd(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_v2_sy_ctl_two_img_rd', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=saycam_funcs.random_saycam_two_img_rd_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_sy_cont(args):
    return get_typical_params(
            args, exp_id='r18_v2_sy_cont', 
            cfg_func=saycam_funcs.cont_saycam_cfg_func)


def get_typical_ep300_params(
        args, exp_id, cfg_func,
        add_moco_hook=False,
        seed=None):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        return cfg
    param_builder = SAYCamParamBuilder(
            args=args, exp_id=exp_id, 
            cfg_path=BASIC_MOCO_CFG, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=_apply_ep300_func,
            opt_update_interval=16,
            col_name_in_work_dir=True,
            seed=seed)
    params = param_builder.build_params()
    if add_moco_hook:
        params = add_moco_hook_to_params(params, 16)
    return params


def r18_sy_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300', 
            saycam_funcs.cont_saycam_ep300_cfg_func)


def r18_sy_cont_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_s1',
            saycam_funcs.cont_saycam_ep300_cfg_func,
            seed=1)


def r18_sy_cont_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_s2',
            saycam_funcs.cont_saycam_ep300_cfg_func,
            seed=2)


def r18_sy_cont_ep300_syncbn(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_syncbn', 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                r18_funcs.simclr_neck_tau,
                r18_funcs.use_sync_bn))


def r18_sy_cont_up_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_up_ep300', 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                r18_funcs.avoid_update_in_forward),
            add_moco_hook=True)


def r18_sy_cont_ep300_smallq(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_smallq', 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                r18_funcs.small_queue))


def r18_sy_ctl_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300', 
            saycam_funcs.random_saycam_1M_cfg_func)


def r18_sy_ctl_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_s1',
            saycam_funcs.random_saycam_1M_cfg_func,
            seed=1)


def r18_sy_ctl_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_s2',
            saycam_funcs.random_saycam_1M_cfg_func,
            seed=2)


def r18_simneck_sy_ctl_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_simneck_sy_ctl_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                r18_funcs.simclr_neck_tau))


def r18_simneck_sy_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_simneck_sy_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                r18_funcs.simclr_neck_tau))


def r18_msct_sy_ctl_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msct_sy_ctl_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                r18_funcs.moco_simclr_concat_more_neg))


def r18_msct_sy_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msct_sy_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                r18_funcs.moco_simclr_concat_more_neg))


def r18_msctl_sy_ctl_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msctl_sy_ctl_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                r18_funcs.moco_simclr_concat_neg))


def r18_msctl_sy_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msctl_sy_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                r18_funcs.moco_simclr_concat_neg))


def r18_sy_cont_ws_gn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ws_gn_ep300', 
            gnrl_funcs.sequential_func(
                ws_gn_funcs.moco_ws_gn_cfg_func,
                saycam_funcs.cont_saycam_ep300_cfg_func))


def get_cotrain_params(
        args, exp_id, cfg_func,
        mix_weight=1.0,
        add_moco_hook=False, batch_size=None):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = CotrainSAYCamParamBuilder(
            mix_weight=mix_weight,
            args=args, exp_id=exp_id, cfg_path=BASIC_MOCO_CFG, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=_apply_ep300_func,
            opt_update_interval=16,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    if add_moco_hook:
        params = add_moco_hook_to_params(params)
    return params


def r18_sy_half_cotrain_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                r18_funcs.avoid_update_in_forward,
                r18_funcs.use_sync_bn),
            add_moco_hook=True, batch_size=32)
