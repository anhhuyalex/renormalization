from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.byol_neg_funcs as byol_neg_funcs
import openselfsup.config_related.byol_funcs as byol_funcs
import openselfsup.config_related.ewc_cfg_funcs as ewc_cfg_funcs
from .r18 import BASIC_BYOL_CFG, add_byol_hook_to_params, BASIC_BYOL_EP300_CFG


def get_typical_params(args, exp_id, cfg_func):
    cfg_path = BASIC_BYOL_CFG
    param_builder = SAYCamParamBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='byol',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    params = add_byol_hook_to_params(params)
    return params


def r18_sy_cont(args):
    return get_typical_params(
            args, 'r18_sy_cont', 
            saycam_funcs.cont_saycam_cfg_func)


def r18_sy_ctl(args):
    return get_typical_params(
            args, 'r18_sy_ctl', 
            saycam_funcs.random_saycam_cfg_func)


def get_typical_ep300_params(
        args, exp_id, cfg_func, 
        cfg_path=BASIC_BYOL_CFG,
        update_freq=16,
        seed=None,
        need_ewc_hook=False, **kwargs):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        return cfg
    param_builder = SAYCamParamBuilder(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='byol',
            cfg_change_func=_apply_ep300_func,
            opt_update_interval=update_freq,
            col_name_in_work_dir=True,
            seed=seed,
            need_ewc_hook=need_ewc_hook,
            **kwargs)
    params = param_builder.build_params()
    params = add_byol_hook_to_params(
            params, update_freq, 
            use_ewc_hook=need_ewc_hook)
    return params


#def r18_sy_cont_ep300(args):
#    return get_typical_ep300_params(
#            args, 'r18_sy_cont_ep300', 
#            saycam_funcs.cont_saycam_ep300_cfg_func)


#def r18_sy_ctl_ep300(args):
#    return get_typical_ep300_params(
#            args, 'r18_sy_ctl_ep300', 
#            saycam_funcs.random_saycam_1M_cfg_func)


def r18_sy_cont_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_fx_2', 
            saycam_funcs.cont_saycam_ep300_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_ep300_fx_s1(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_fx_s1', 
            saycam_funcs.cont_saycam_ep300_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=1)


def r18_sy_cont_ep300_fx_s2(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_fx_s2', 
            saycam_funcs.cont_saycam_ep300_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=2)


def r18_sy_cont_ep300_fx_s3(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_fx_s3',
            saycam_funcs.cont_saycam_ep300_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=3)


def r18_sy_cont_mlp3_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp3_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                byol_funcs.more_mlp_layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_mlp4_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp4_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_mlp5_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp5_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                byol_funcs.mlp_5layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_mlp6_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp6_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                byol_funcs.mlp_6layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_mlp4l1bn_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp4l1bn_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                byol_funcs.mlp_4L1bn_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_mlp4_ir112_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp4_ir112_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_mlp3_ir112_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp3_ir112_ep300_fx', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.more_mlp_layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_ctl_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_fx_2', 
            saycam_funcs.random_saycam_1M_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_ctl_ep300_fx_s1(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_fx_s1', 
            saycam_funcs.random_saycam_1M_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=1)


def r18_sy_ctl_ep300_fx_s2(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_fx_s2', 
            saycam_funcs.random_saycam_1M_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=2)


def r18_sy_ctl_mlp3_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_mlp3_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                byol_funcs.more_mlp_layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_ctl_mlp4_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_mlp4_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_ctl_mlp5_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_mlp5_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                byol_funcs.mlp_5layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_ctl_mlp3_ir112_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_mlp3_ir112_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.more_mlp_layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_ctl_mlp4_ir112_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_mlp4_ir112_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_cont_ewc_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_wrap),
            cfg_path=BASIC_BYOL_EP300_CFG,
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_l30_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_l30_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_s_wrap),
            cfg_path=BASIC_BYOL_EP300_CFG,
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_gd5_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_gd5_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_gd5_wrap),
            cfg_path=BASIC_BYOL_EP300_CFG,
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_l30gd5_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_l30gd5_ep300_fx',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_l30gd5_wrap),
            cfg_path=BASIC_BYOL_EP300_CFG,
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_neg_sy_ctl_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_neg_sy_ctl_ep300',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                saycam_funcs.random_saycam_1M_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            update_freq=8)


def r18_neg_sy_ctl_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_neg_sy_ctl_ep300_s1',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                saycam_funcs.random_saycam_1M_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=1)


def r18_neg_sy_ctl_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_neg_sy_ctl_ep300_s2',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                saycam_funcs.random_saycam_1M_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=2)


def r18_neg_sy_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_neg_sy_cont_ep300',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                saycam_funcs.cont_saycam_ep300_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            update_freq=8)


def r18_neg_sy_cont_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_neg_sy_cont_ep300_s1',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                saycam_funcs.cont_saycam_ep300_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=1)


def r18_neg_sy_cont_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_neg_sy_cont_ep300_s2',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                saycam_funcs.cont_saycam_ep300_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            seed=2)


def r18_sy_cont_ws_gn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ws_gn_ep300', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ws_gn_funcs.byol_ws_gn_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def get_cotrain_params(
        args, exp_id, cfg_func, 
        cfg_path=BASIC_BYOL_CFG, mix_weight=1.0,
        batch_size=None, **kwargs):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = CotrainSAYCamParamBuilder(
            mix_weight=mix_weight,
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='byol',
            cfg_change_func=_apply_ep300_func,
            opt_update_interval=16,
            col_name_in_work_dir=True, **kwargs)
    params = param_builder.build_params()
    params = add_byol_hook_to_params(params, 16)
    return params


def r18_sy_cotrain_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_cotrain_ep300_2', 
            saycam_funcs.cotrain_saycam_ep300_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_sy_half_cotrain_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300',
            saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=32)


def r18_sy_hlf_cotr_is112_mlp4_cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_cotr_is112_mlp4_cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func,
                saycam_funcs.cnd_storage_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hlf_cotr_is112_mlp4_mincnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_cotr_is112_mlp4_mincnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func,
                saycam_funcs.min_cnd_storage_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hlf_cotr_is112_mlp4_max100cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_cotr_is112_mlp4_max100cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max100_cnd_storage_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hlf_cotr_is112_mlp4_max500cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_cotr_is112_mlp4_max500cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max500_cnd_storage_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hlf_catcotr_is112_mlp4_max500cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_catcotr_is112_mlp4_max500cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max500_cnd_storage_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hlf_cotr_is112_mlp4_max2000cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_cotr_is112_mlp4_max2000cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max2000_cnd_storage_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hlf_cotr_is112_mlp4_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_cotr_is112_mlp4_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64)


def r18_sy_hlf_catcotr_is112_mlp4_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hlf_catcotr_is112_mlp4_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=64,
            concat_batches=True)


def r18_sy_half_cotrain_neg_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_neg_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                byol_neg_funcs.byol_neg_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            batch_size=32)
