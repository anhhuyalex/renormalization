from .saycam_r18 import get_typical_ep300_params, get_cotrain_params
import openselfsup.config_related.imgnt_cont_funcs as imgnt_cont_funcs
import openselfsup.config_related.moco.r18_funcs as r18_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs


def r18_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_ep300', 
            imgnt_cont_funcs.imgnt_cont_cfg_func)


def r18_in_bld_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_bld_ep300',
            imgnt_cont_funcs.imgnt_batchLD_cfg_func)


def r18_msct_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msct_in_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                r18_funcs.moco_simclr_concat_more_neg))


def r18_msctl_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msctl_in_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                r18_funcs.moco_simclr_concat_neg))


def r18_simneck_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_simneck_in_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                r18_funcs.simclr_neck_tau))


def r18_msct_simneck_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msct_simneck_in_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                r18_funcs.ms_concat_more_simneck))


def r18_msct_simneck_fn_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_msct_simneck_fn_in_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                r18_funcs.ms_concat_more_simneck_fn))


def r18_cotr_in_cont_ep300(args):
    return get_cotrain_params(
            args, 'r18_cotr_in_cont_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                imgnt_cont_funcs.cotr_imgnt_cont_cfg_func,
                r18_funcs.avoid_update_in_forward,
                r18_funcs.use_sync_bn),
            add_moco_hook=True, batch_size=32)
