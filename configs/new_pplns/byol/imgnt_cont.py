from .saycam import get_typical_ep300_params, BASIC_BYOL_EP300_CFG
import openselfsup.config_related.imgnt_cont_funcs as imgnt_cont_funcs
import openselfsup.config_related.byol_neg_funcs as byol_neg_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.byol_funcs as byol_funcs


def r18_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_ep300_2', 
            imgnt_cont_funcs.imgnt_cont_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_in_cont_mlp4_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_mlp4_ep300',
            gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                byol_funcs.mlp_4layers_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_in_bld_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_bld_ep300', 
            imgnt_cont_funcs.imgnt_batchLD_cfg_func,
            cfg_path=BASIC_BYOL_EP300_CFG)


def r18_neg_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_neg_in_cont_ep300',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                imgnt_cont_funcs.imgnt_cont_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            update_freq=8)


def r18_neg_in_bld_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_neg_in_bld_ep300',
            gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                imgnt_cont_funcs.imgnt_batchLD_cfg_func),
            cfg_path=BASIC_BYOL_EP300_CFG,
            update_freq=16)
