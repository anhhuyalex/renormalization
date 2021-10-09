from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.simsiam_cfg_funcs as simsiam_cfg_funcs
BASIC_SIMSIAM_CFG = './configs/selfsup/siamese/r18.py'


def get_typical_params(args, exp_id, cfg_func, seed=None):
    cfg_path = BASIC_SIMSIAM_CFG
    def cfg_func_then_change_bs128(cfg):
        cfg = cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 128
        return cfg
    param_builder = ParamsBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='simsiam',
            cfg_change_func=cfg_func_then_change_bs128,
            col_name_in_work_dir=True,
            seed=seed)
    params = param_builder.build_params()
    return params


def r18(args):
    return get_typical_params(
            args, 'r18', 
            lambda x: x)


def get_typical_ep300_params(args, exp_id, cfg_func):
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_SIMSIAM_CFG, 
            add_svm_val=True,
            col_name='simsiam',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            opt_update_interval=16,
            )
    params = param_builder.build_params()
    return params


def r18_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ep300', 
            cfg_func=ep300_funcs.ep300_cfg_func)


def r18_ep300_SGD(args):
    def cfg_func_then_change_bs128(cfg):
        cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 128
        return cfg
    return get_typical_ep300_params(
            args, 'r18_ep300_SGD', 
            cfg_func=cfg_func_then_change_bs128)


def r18_ep300_SGD_BNex(args):
    def cfg_func_then_change_bs128(cfg):
        cfg = ep300_funcs.ep300_SGD_BNex_cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 128
        return cfg
    return get_typical_ep300_params(
            args, 'r18_ep300_SGD_BNex', 
            cfg_func=cfg_func_then_change_bs128)


def r18_ep300_SGD_bs512(args):
    def cfg_func_then_change_lr(cfg):
        cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.optimizer['lr'] = 0.1
        return cfg
    return get_typical_params(
            args, 'r18_ep300_SGD_bs512', 
            cfg_func=cfg_func_then_change_lr)


def r18_ep300_SGD_bs512_s1(args):
    def cfg_func_then_change_lr(cfg):
        cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.optimizer['lr'] = 0.1
        return cfg
    return get_typical_params(
            args, 'r18_ep300_SGD_bs512_s1',
            cfg_func=cfg_func_then_change_lr,
            seed=1)


def r18_ep300_SGD_bs512_s2(args):
    def cfg_func_then_change_lr(cfg):
        cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.optimizer['lr'] = 0.1
        return cfg
    return get_typical_params(
            args, 'r18_ep300_SGD_bs512_s2',
            cfg_func=cfg_func_then_change_lr,
            seed=2)


def r18_ep300_more_hid_bn(args):
    def cfg_func_then_change_lr(cfg):
        cfg = simsiam_cfg_funcs.more_hid_bn(cfg)
        cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.optimizer['lr'] = 0.1
        return cfg
    return get_typical_params(
            args, 'r18_ep300_more_hid_bn', 
            cfg_func=cfg_func_then_change_lr)
