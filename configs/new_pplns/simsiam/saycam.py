from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simsiam_cfg_funcs as simsiam_cfg_funcs
BASIC_SIMSIAM_CFG = './configs/selfsup/siamese/r18.py'


def get_typical_params(args, exp_id, cfg_func):
    cfg_path = BASIC_SIMSIAM_CFG
    def cfg_func_then_change_bs128(cfg):
        cfg = cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 128
        return cfg
    param_builder = SAYCamParamBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='simsiam',
            cfg_change_func=cfg_func_then_change_bs128,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_ctl(args):
    return get_typical_params(
            args, 'r18_sy_ctl', 
            saycam_funcs.random_saycam_cfg_func)


def r18_sy_cont(args):
    return get_typical_params(
            args, 'r18_sy_cont',
            saycam_funcs.cont_saycam_cfg_func)


def get_typical_ep300_params(
        args, exp_id, cfg_func, lr=0.1, seed=None):
    cfg_path = BASIC_SIMSIAM_CFG
    def cfg_func_then_change_bs128_ep300(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 128
        cfg.optimizer['lr'] = lr
        return cfg
    param_builder = SAYCamParamBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='simsiam',
            cfg_change_func=cfg_func_then_change_bs128_ep300,
            col_name_in_work_dir=True,
            seed=seed)
    params = param_builder.build_params()
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


def r18_sy_ctl_ep300_bs1024(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_bs1024_3', 
            saycam_funcs.random_saycam_1M_cfg_func,
            lr=0.1)


def r18_sy_ctl_ep300_bs256(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_bs256', 
            saycam_funcs.random_saycam_1M_cfg_func,
            lr=0.05)


def get_cotrain_params(
        args, exp_id, cfg_func, lr=0.1,
        mix_weight=1.0, **kwargs):
    cfg_path = BASIC_SIMSIAM_CFG
    def cfg_func_then_change_lr_bs64(cfg):
        cfg = cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 64
        cfg.optimizer['lr'] = lr
        return cfg
    param_builder = CotrainSAYCamParamBuilder(
            mix_weight=mix_weight,
            args=args, exp_id=exp_id, cfg_path=cfg_path,
            add_svm_val=True, col_name='simsiam',
            cfg_change_func=cfg_func_then_change_lr_bs64,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_half_cotrain_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300', 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                ep300_funcs.ep300_SGD_cfg_func))


def r18_sy_half_cotrain_ep300_lr(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300_lr_2', 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                ep300_funcs.ep300_SGD_lrfx_cfg_func),
            lr=0.1, opt_grad_clip={'max_norm': 1.0})


def r18_sy_half_cotr_more_hid_bn(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotr_more_hid_bn',
            cfg_func=gnrl_funcs.sequential_func(
                simsiam_cfg_funcs.more_hid_bn,
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                ep300_funcs.ep300_SGD_cfg_func))
