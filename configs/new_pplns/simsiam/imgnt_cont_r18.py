from .saycam import get_typical_ep300_params
import openselfsup.config_related.imgnt_cont_funcs as imgnt_cont_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simsiam_cfg_funcs as simsiam_cfg_funcs


def r18_in_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_ep300', 
            imgnt_cont_funcs.imgnt_cont_cfg_func)


def r18_in_bld_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_in_bld_ep300',
            imgnt_cont_funcs.imgnt_batchLD512_cfg_func)


def r18_in_cont_more_hid_bn(args):
    return get_typical_ep300_params(
            args, 'r18_in_cont_more_hid_bn',
            gnrl_funcs.sequential_func(
                imgnt_cont_funcs.imgnt_cont_cfg_func,
                simsiam_cfg_funcs.more_hid_bn))
