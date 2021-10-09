from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.moco.r18_funcs as r18_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def r18(args):
    param_builder = ParamsBuilder(
            args, 'r18', BASIC_SIMCLR_CFG, 
            add_svm_val=True)
    params = param_builder.build_params()
    return params


def r18_rdpd_sm(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = ParamsBuilder(
            args, 'r18_rdpd_sm', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=r18_funcs.rdpd_sm_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_img_face_rdpd_sm(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = ParamsBuilder(
            args, 'r18_img_face_rdpd_sm', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=r18_funcs.img_face_rdpd_sm_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_img_face_rdpd_sm_pos5prob8(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = ParamsBuilder(
            args, 'r18_img_face_rdpd_sm_pos5prob8', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=r18_funcs.img_face_rdpd_sm_pos5prob8_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def get_typical_ep300_params(
        args, exp_id, cfg_func,
        opt_update_interval=16,
        seed=None):
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_SIMCLR_CFG, 
            add_svm_val=True,
            col_name='simclr',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            opt_update_interval=opt_update_interval,
            seed=seed,
            )
    params = param_builder.build_params()
    return params


def r18_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ep300', 
            cfg_func=ep300_funcs.ep300_cfg_func)


def r18_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s1', 
            cfg_func=ep300_funcs.ep300_cfg_func,
            seed=1)


def r18_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s2', 
            cfg_func=ep300_funcs.ep300_cfg_func,
            seed=2)


def r18_mfn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_mfn_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                simclr_cfg_funcs.mneg_fn_num,
                ep300_funcs.ep300_cfg_func))


def r18_ep300_bs512(args):
    def _seq_cfg_func(cfg):
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = 128
        return cfg
    return get_typical_ep300_params(
            args, 'r18_ep300_bs512', 
            cfg_func=_seq_cfg_func,
            opt_update_interval=8)


def r18_ep300_bs128(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_bs128', 
            cfg_func=ep300_funcs.ep300_cfg_func,
            opt_update_interval=32)


def r18_ws_gn_ep300(args):
    def _seq_cfg_func(cfg):
        cfg = ws_gn_funcs.simclr_ws_gn_cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        return cfg
    return get_typical_ep300_params(
            args, 'r18_ws_gn_ep300', 
            cfg_func=_seq_cfg_func)
