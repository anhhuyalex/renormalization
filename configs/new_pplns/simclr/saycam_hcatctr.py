from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.ewc_cfg_funcs as ewc_cfg_funcs
import openselfsup.config_related.vae_funcs as vae_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import torch
from .saycam_r18 import get_cotrain_params
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def r18_sy_hctr_typ_s2_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_typ_s2_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.scale_second_dataset(2),
                ),
            batch_size=64,
            concat_batches=True,
            scale_ratio=2,
            )


def r18_sy_hcatctr_is112_mlp4_max4k_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_max4k_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max4000_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_max10kcnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_max10k_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max10k_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_summax12k_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_summax12k_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.summax12k_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_max20kcnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_max20k_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max20k_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_is112_mlp4_max40kcnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_max40k_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max40k_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hcatctr_typ_max40kcnd_s2_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_typ_max40kcnd_s2_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max40k_cnd_storage_cfg_func,
                saycam_funcs.scale_second_dataset(2),
                ),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True,
            scale_ratio=2,
            )


def r18_sy_hcatctr_is112_mlp4_max80kcnd_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hcatctr_is112_mlp4_max80k_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max80k_cnd_storage_cfg_func),
            batch_size=64,
            use_cnd_hook=True,
            concat_batches=True)


def r18_sy_hctr_typ_thd7_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_typ_thd7_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                simclr_cfg_funcs.neg_th_value_d7,
                ),
            batch_size=64,
            concat_batches=True)


def r18_sy_hctr_typ_thd5_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_typ_thd5_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                simclr_cfg_funcs.neg_th_value_d5,
                ),
            batch_size=64,
            concat_batches=True)


def r18_sy_hctr_typ_thd9_ep300(args):
    return get_cotrain_params(
            args, 'r18_sy_hctr_typ_thd9_ep300',
            gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                simclr_cfg_funcs.neg_th_value_d9,
                ),
            batch_size=64,
            concat_batches=True)
