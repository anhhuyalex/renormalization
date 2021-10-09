from .saycam_r18 import get_typical_ep300_params, get_cotrain_params
import openselfsup.config_related.imgnt_cont_funcs as imgnt_cont_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs


def r18_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_ep300',
            imgnt_cont_funcs.imgnt_cont_cfg_func)


def r18_in_bld_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_bld_ep300',
            imgnt_cont_funcs.imgnt_batchLD_cfg_func)


def r18_in_cont_ep300_bs512(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_ep300_bs512',
            imgnt_cont_funcs.imgnt_cont_cfg_func,
            batch_size=128, opt_update_interval=8)


def r18_lowt_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_lowt_in_cont_ep300',
            gnrl_funcs.sequential_func(
                simclr_cfg_funcs.lower_tau,
                imgnt_cont_funcs.imgnt_cont_cfg_func))


def r18_fn_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_fn_in_cont_ep300',
            gnrl_funcs.sequential_func(
                simclr_cfg_funcs.neg_fn_num,
                imgnt_cont_funcs.imgnt_cont_cfg_func))


def r18_mfn_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_mfn_in_cont_ep300',
            gnrl_funcs.sequential_func(
                simclr_cfg_funcs.mneg_fn_num,
                imgnt_cont_funcs.imgnt_cont_cfg_func))


def r18_cotr_in_cont_ep300(args):
    return get_cotrain_params(
            args, 'r18_cotr_in_cont_ep300', 
            imgnt_cont_funcs.cotr_imgnt_cont_cfg_func,
            batch_size=32)


def r18_cotr_in_ct_acsy_ep300(args):
    params = get_cotrain_params(
            args, 'r18_cotr_in_ct_acsy_ep300', 
            imgnt_cont_funcs.cotr_imgnt_cont_acsy_cfg_func,
            batch_size=32)
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/simclr/r18_sy_half_cotrain_ep300/epoch_100.pth',
            }
    return params


def r18_cotr_in_ct_acsy_ep300_md3(args):
    params = get_cotrain_params(
            args, 'r18_cotr_in_ct_acsy_ep300_md3', 
            imgnt_cont_funcs.cotr_imgnt_cont_acsy_cfg_func,
            batch_size=32,
            mix_weight=0.3)
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/simclr/r18_sy_half_cotrain_ep300_md3/epoch_100.pth',
            }
    return params


def r18_cotr_in_ct_acsy_ep300_m3(args):
    params = get_cotrain_params(
            args, 'r18_cotr_in_ct_acsy_ep300_m3',
            imgnt_cont_funcs.cotr_imgnt_cont_acsy_cfg_func,
            batch_size=32,
            mix_weight=3.0)
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/simclr/r18_sy_half_cotrain_ep300_m3/epoch_100.pth',
            }
    return params
