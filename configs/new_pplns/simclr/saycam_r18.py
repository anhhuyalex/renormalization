from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.ewc_cfg_funcs as ewc_cfg_funcs
import openselfsup.config_related.vae_funcs as vae_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def r18_sy_ctl(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_sy_ctl', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=saycam_funcs.random_saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_order(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_sy_order_fx', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=saycam_funcs.saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_order_hp(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_sy_order_hp', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=saycam_funcs.hipp_pred_saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_ctl_two_img(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_sy_ctl_two_img', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=saycam_funcs.random_saycam_two_img_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_ctl_two_img_rd(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_sy_ctl_two_img_rd', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=saycam_funcs.random_saycam_two_img_rd_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_sy_cont(args):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamParamBuilder(
            args, 'r18_sy_cont', cfg_path, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=saycam_funcs.cont_saycam_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def get_typical_ep300_params(
        args, exp_id, cfg_func,
        batch_size=None, opt_update_interval=16,
        seed=None, **kwargs):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = SAYCamParamBuilder(
            args=args, exp_id=exp_id, cfg_path=BASIC_SIMCLR_CFG, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=_apply_ep300_func,
            opt_update_interval=opt_update_interval,
            col_name_in_work_dir=True,
            seed=seed, **kwargs)
    params = param_builder.build_params()
    return params


def r18_sy_cont_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300', 
            saycam_funcs.cont_saycam_ep300_cfg_func)


def r18_sy_cont_ewc_ll_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_ll_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_ll_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_l_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_l_ep300_2',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_l_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_ep300_3', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_s_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_s_ep300_3',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_s_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_l10_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_l10_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_l10_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_gd5_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_gd5_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_gd5_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_l300gd5_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_l300gd5_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_l300gd5_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


def r18_sy_cont_ewc_l30gd5_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ewc_l30gd5_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ewc_cfg_funcs.ewc_l30gd5_wrap),
            need_ewc_hook=True,
            opt_grad_clip={'max_norm': 1.0})


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


def r18_sy_cont_accu_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_accu_ep300',
            saycam_funcs.cont_accu_saycam_ep300_cfg_func)


def r18_sy_cont_mlp3_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp3_ep300', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                simclr_cfg_funcs.mlp_3layers_cfg_func),
            )


def r18_sy_cont_mlp4_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp4_ep300', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                simclr_cfg_funcs.mlp_4layers_cfg_func),
            )


def r18_sy_cont_mlp41bn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp41bn_ep300', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                simclr_cfg_funcs.mlp_4L1bn_cfg_func),
            )


def r18_sy_cont_mlp5_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_mlp5_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                simclr_cfg_funcs.mlp_5layers_cfg_func),
            )


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


def r18_sy_ctl_mlp3_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_mlp3_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                simclr_cfg_funcs.mlp_3layers_cfg_func),
            )


def r18_sy_ctl_ep300_is112_recon(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ep300_is112_recon', 
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                vae_funcs.add_recon_train_vae112,
                ))


def r18_sy_cont_ws_gn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ws_gn_ep300', 
            gnrl_funcs.sequential_func(
                saycam_funcs.cont_saycam_ep300_cfg_func,
                ws_gn_funcs.simclr_ws_gn_cfg_func))


def r18_sy_ctl_ws_gn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_sy_ctl_ws_gn_ep300', 
            gnrl_funcs.sequential_func(
                saycam_funcs.random_saycam_1M_cfg_func,
                ws_gn_funcs.simclr_ws_gn_cfg_func))


def r18_sy_cont_ep300_bs512(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_bs512', 
            saycam_funcs.cont_saycam_ep300_cfg_func,
            batch_size=128, opt_update_interval=8)


def r18_sy_cont_ep300_bs128(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_bs128',
            saycam_funcs.cont_saycam_ep300_cfg_func,
            opt_update_interval=32)


def r18_sy_cont_ep300_bs64(args):
    return get_typical_ep300_params(
            args, 'r18_sy_cont_ep300_bs64',
            saycam_funcs.cont_saycam_ep300_cfg_func,
            opt_update_interval=64)


def get_cotrain_params(
        args, exp_id, cfg_func,
        batch_size=None, opt_update_interval=16,
        mix_weight=1.0, **kwargs):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = CotrainSAYCamParamBuilder(
            mix_weight=mix_weight,
            args=args, exp_id=exp_id, cfg_path=BASIC_SIMCLR_CFG, 
            add_svm_val=True, col_name='simclr',
            cfg_change_func=_apply_ep300_func,
            opt_update_interval=opt_update_interval,
            col_name_in_work_dir=True, **kwargs)
    params = param_builder.build_params()
    return params


def r18_sy_cotrain_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_cotrain_ep300_2', 
            saycam_funcs.cotrain_saycam_ep300_cfg_func)


def r18_sy_half_cotrain_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300', 
            saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
            batch_size=32)


def r18_sy_hctr_is112_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112),
            batch_size=64)


def r18_sy_hctr_is112_mlp4_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_mlp4_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func),
            batch_size=64)


def r18_sy_hcatctr_is112_mlp4_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func),
            batch_size=64,
            concat_batches=True)


def r18_sy_hctr_is112_mlp4_max100cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_mlp4_max100cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max100_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hctr_is112_mlp4_max500cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_mlp4_max500cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max500_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hctr_is112_mlp4_max2000cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_mlp4_max2000cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max2000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hctr_is112_mlp4_max4000cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_mlp4_max4000cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max4000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hctr_is112_mlp4_max6000cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_is112_mlp4_max6000cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max6000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True)


def r18_sy_hcatctr_is112_mlp4_max6000cnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_max6000cnd_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max6000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_summax2000_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_summax2000_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.summax2000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_summax4000_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_summax4000_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.summax4000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_summax6000_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_summax6000_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.summax6000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_half_cotrain_ep300_md3(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300_md3', 
            saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
            batch_size=32,
            mix_weight=0.3)


def r18_sy_half_cotrain_ep300_m0(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300_m0', 
            saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
            batch_size=32,
            mix_weight=0.0)


def r18_sy_half_cotrain_ep300_m3(args):
    return get_cotrain_params(
            args, 'r18_sy_half_cotrain_ep300_m3', 
            saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
            batch_size=32,
            mix_weight=3)


def r18_sy_cotrain_ep300_md5(args):
    return get_cotrain_params(
            args, 'r18_sy_cotrain_ep300_md5', 
            saycam_funcs.cotrain_saycam_ep300_cfg_func,
            mix_weight=0.5)


def r18_sy_cotrain_ep300_m2(args):
    return get_cotrain_params(
            args, 'r18_sy_cotrain_ep300_m2', 
            saycam_funcs.cotrain_saycam_ep300_cfg_func,
            mix_weight=2.0)
