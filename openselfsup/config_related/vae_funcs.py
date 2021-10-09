import copy
from . import saycam_funcs


IMAGE_SIZE = 32
BASIC_VAE_CFG = dict(
        type='AutoEncoder',
        image_size=IMAGE_SIZE, 
        image_channels=3, classes=None,
        )


def set_adam_opt(cfg):
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    return cfg


def vae_with_img_size(cfg, image_size):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['pipeline'][0]['size'] = image_size
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(BASIC_VAE_CFG)
    cfg.model['image_size'] = image_size
    cfg = set_adam_opt(cfg)
    return cfg


def ctl_saycam_vae(cfg):
    return vae_with_img_size(cfg, 32)


def ctl_saycam_vae64(cfg):
    return vae_with_img_size(cfg, 64)


def ctl_saycam_vae112(cfg):
    return vae_with_img_size(cfg, 112)


def add_recon_train_vae112(cfg):
    cfg.data['train']['pipeline'][0]['size'] = 112
    old_model_cfg = copy.deepcopy(cfg.model)
    vae_cfg = copy.deepcopy(BASIC_VAE_CFG)
    vae_cfg['image_size'] = 112
    cfg.model = {
            'type': 'ReconTrain',
            'model_cfg': old_model_cfg,
            'vae_cfg': vae_cfg,
            'vae_pretrained': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/vae/default_vae112/epoch_7.pth',
            }
    return cfg


INTER_VAE_CFG = dict(
        type='InterAutoEncoder',
        backbone=dict(
            type='ResNet',
            depth=18,
            in_channels=3,
            out_indices=[2],  # 0: conv-1, x: stage-x
            norm_cfg=dict(type='SyncBN'),
            frozen_stages=4,
            num_stages=4,
            ),
        vae_cfg=dict(
            type='AutoEncoder',
            image_size=28, 
            image_channels=8, classes=None,
            start_channels=128,
            network_output="standard",
            ),
        padding_size=2,
        pretrained='/mnt/fs4/chengxuz/openselfsup_models/work_dirs/pretrains/ctl_SY/simclr_r18.pth',
        )
def saycam_inter_vae(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg = set_adam_opt(cfg)
    return cfg


def set_moreZ(cfg):
    cfg.model['vae_cfg']['z_dim'] = 500
    cfg.model['vae_cfg']['n_modes'] = 500
    return cfg


def saycam_inter_vae_moreZ(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg = set_moreZ(cfg)
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_inter_vae_c2(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg.model['vae_cfg']['image_channels'] = 2
    cfg.model['vae_cfg']['start_channels'] = 64
    cfg = set_adam_opt(cfg)
    return cfg


def set_lessZ(cfg):
    cfg.model['vae_cfg']['z_dim'] = 50
    cfg.model['vae_cfg']['n_modes'] = 50
    return cfg


def saycam_inter_vae_c2_lessZ(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg.model['vae_cfg']['image_channels'] = 2
    cfg.model['vae_cfg']['start_channels'] = 64
    cfg = set_lessZ(cfg)
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_inter_vae_c2_moreZ(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg.model['vae_cfg']['image_channels'] = 2
    cfg.model['vae_cfg']['start_channels'] = 64
    cfg = set_moreZ(cfg)
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_inter_vae_c1_moreZ(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg.model['vae_cfg']['image_channels'] = 1
    cfg.model['vae_cfg']['start_channels'] = 64
    cfg = set_moreZ(cfg)
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_inter_vae_c2_s112(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['pipeline'][0]['size'] = 112
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg.model['vae_cfg']['image_channels'] = 2
    cfg.model['vae_cfg']['image_size'] = 16
    cfg.model['padding_size'] = 1
    cfg.model['vae_cfg']['start_channels'] = 64
    cfg.model['pretrained'] = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/pretrains/ctl_SY_112/simclr_r18_ep10.pth'
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_inter_vae_c2_s224m112(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTER_VAE_CFG)
    cfg.model['vae_cfg']['image_channels'] = 2
    cfg.model['vae_cfg']['start_channels'] = 64
    cfg.model['vae_cfg']['image_size'] = 16
    cfg.model['padding_size'] = 1
    cfg.model['image_region'] = (112-56, 112+56)
    cfg = set_adam_opt(cfg)
    return cfg


INTERBN_VAE_CFG = dict(
        type='InterBNAutoEncoder',
        backbone=dict(
            type='ResNet',
            depth=18,
            in_channels=3,
            out_indices=[2],  # 0: conv-1, x: stage-x
            norm_cfg=dict(type='SyncBN'),
            frozen_stages=4,
            num_stages=4,
            ),
        vae_cfg=dict(
            type='AutoEncoder',
            image_size=112, 
            image_channels=3, classes=None,
            ),
        inter_channels=8,
        pretrained='/mnt/fs4/chengxuz/openselfsup_models/work_dirs/pretrains/ctl_SY_112/simclr_r18_ep10.pth',
        )
def saycam_interbn_vae_c2(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['pipeline'][0]['size'] = 112
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTERBN_VAE_CFG)
    cfg.model['inter_channels'] = 2
    cfg = set_adam_opt(cfg)
    return cfg


def test_saycam_interbn_vae_c2_w_vae112(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['pipeline'][0]['size'] = 112
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTERBN_VAE_CFG)
    cfg.model['inter_channels'] = 2
    cfg.model['vae_pretrained'] = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/vae/default_vae112/epoch_7.pth'
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_interbn_vae_c128(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['pipeline'][0]['size'] = 112
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.model = copy.deepcopy(INTERBN_VAE_CFG)
    cfg.model['inter_channels'] = 128
    cfg = set_adam_opt(cfg)
    return cfg


def saycam_interbn_vae_c2_s224m112(cfg):
    cfg = saycam_funcs.random_saycam_1M_cfg_func(cfg)
    cfg.data['train']['type'] = 'ExtractDataset'
    cfg.data['train']['pipeline'].insert(
                1, dict(type='CenterCrop', size=112))
    cfg.model = copy.deepcopy(INTERBN_VAE_CFG)
    cfg.model['inter_channels'] = 2
    cfg.model['pretrained'] = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    cfg = set_adam_opt(cfg)
    return cfg
