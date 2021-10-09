from ..basic_param_setter import ParamsBuilder, MODEL_SAVE_FOLDER
from .r18 import get_typical_ep300_params
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.moco.r18_funcs as r18_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import os

BASIC_MOCO_CFG = './configs/selfsup/moco/r18_v2_with_val.py'


# Not working
def r18_v2_remove_me(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        return cfg
    param_builder = ParamsBuilder(
            args, 'r18_v2_remove_me', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


# Not working
def r18_v2_nome_syncbn(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        cfg.model['sync_bn'] = True
        return cfg
    param_builder = ParamsBuilder(
            args, 'r18_v2_nome_syncbn', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


# Not working
def r18_v2_nome_syncbn_sharebn(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        cfg.model['sync_bn'] = True
        cfg.model['shared_bn'] = True
        return cfg
    param_builder = ParamsBuilder(
            args, 'r18_v2_nome_syncbn_sharebn', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


# Not working
def r18_v2_nome_syncbn_sharebn_bninmlp(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        cfg.model['sync_bn'] = True
        cfg.model['shared_bn'] = True
        cfg.model['neck']['type'] = 'NonLinearNeckV2'
        return cfg
    param_builder = ParamsBuilder(
            args, 'r18_v2_nome_syncbn_sharebn_bninmlp', 
            cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


# Not working
def warmup_no_wbc(args):
    params = r18_v2_nome_syncbn_sharebn_bninmlp(args)
    params['load_params'] = {
            'from_checkpoint': './work_dirs/new_pipelines/r18_v2_within_batch_ctr/latest_cached.pth'
            }
    params['save_params']['record_saver_kwargs']['exp_id'] = 'warmup_no_wbc'
    params['save_params']['ckpt_hook_kwargs']['out_dir'] = os.path.join(
            MODEL_SAVE_FOLDER, 'warmup_no_wbc')
    return params


# optimizer
LARS_optimizer = dict(type='LARS', lr=0.3, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})
# learning policy
LARS_lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
def lars_opt(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        cfg.model['sync_bn'] = True
        cfg.model['shared_bn'] = True
        cfg.model['within_batch_ctr'] = 1
        cfg.model['neck']['type'] = 'NonLinearNeckV2'
        cfg.optimizer = LARS_optimizer
        cfg.lr_config = LARS_lr_config
        return cfg
    param_builder = ParamsBuilder(
            args, 'r18_v2_nome_lars_opt_wb',
            cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


# Worse performance than LARS
def within_batch_ctr(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        cfg.model['sync_bn'] = True
        cfg.model['shared_bn'] = True
        cfg.model['neck']['type'] = 'NonLinearNeckV2'
        cfg.model['within_batch_ctr'] = 1
        return cfg
    param_builder = ParamsBuilder(
            args, 'r18_v2_within_batch_ctr',
            cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


# Only works with no_neg, which is siamese
def linear_pred(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['remove_momentum_encoder'] = True
        cfg.model['sync_bn'] = True
        cfg.model['shared_bn'] = True
        cfg.model['neck']['type'] = 'NonLinearNeckV2'
        cfg.model['predictor'] = dict(
                type='LinearNeck',
                in_channels=128,
                out_channels=128, with_avg_pool=False)
        cfg.model['no_neg'] = True
        return cfg
    param_builder = ParamsBuilder(
            args, 'moco_r18_v2_linear_pred_4',
            cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


def moco_simclr(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['type'] = 'MOCOSimCLR'
        cfg.model['neck']['type'] = 'NonLinearNeckV2'
        return cfg
    param_builder = ParamsBuilder(
            args, 'moco_r18_v2_simclr',
            cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


def moco_simclr_lars(args):
    cfg_path = BASIC_MOCO_CFG
    def cfg_change_func(cfg):
        cfg.model['type'] = 'MOCOSimCLR'
        cfg.model['neck']['type'] = 'NonLinearNeckV2'
        cfg.optimizer = LARS_optimizer
        cfg.lr_config = LARS_lr_config
        return cfg
    param_builder = ParamsBuilder(
            args, 'moco_r18_v2_simclr_lars',
            cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=cfg_change_func)
    params = param_builder.build_params()
    return params


def r18_ms_ct_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ms_ct_ep300', 
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                r18_funcs.moco_simclr_concat_neg))


def r18_ms_ct_mr_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ms_ct_mr_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                r18_funcs.moco_simclr_concat_more_neg))


def r18_msct_simneck_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msct_simneck_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                r18_funcs.ms_concat_more_simneck))
