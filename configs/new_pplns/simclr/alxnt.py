from ..basic_param_setter import ParamsBuilder
from ..saycam_param_setter import SAYCamParamBuilder
import openselfsup.config_related.alxnt as alxnt
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_params(
        args, exp_id, cfg_func, 
        builder=ParamsBuilder):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = builder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='simclr_alxnt',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    params['validation_params']['svm']['initial'] = False
    params['validation_params']['svm']['interval'] = 10
    return params


def ctl64(args):
    return get_typical_params(
            args, 
            exp_id='ctl64', 
            cfg_func=alxnt.ctl64, 
            builder=ParamsBuilder)


def sy_two_img_ctl64(args):
    return get_typical_params(
            args, 
            exp_id='sy_two_img_ctl64', 
            cfg_func=alxnt.sy_two_img_ctl64, 
            builder=SAYCamParamBuilder)


def sy_two_img_rd_ctl64(args):
    return get_typical_params(
            args, 
            exp_id='sy_two_img_rd_ctl64', 
            cfg_func=alxnt.sy_two_img_rd_ctl64, 
            builder=SAYCamParamBuilder)
