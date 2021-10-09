import copy
from .saycam_funcs import BASIC_DATA_SOURCE_DICT
MST_root = '/mnt/fs4/chengxuz/hippocampus_change/mst_related/MST'
BASIC_MST_DATA_SOURCE_DICT = {
        'type': 'MSTImageList',
        'root': MST_root,
        'oversample_len': 1281167,
        }


def set_syctl_train1(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(BASIC_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['all_random'] = True
    cfg.data['train']['data_source']['set_len'] = 1281167
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train1']['pipeline'].insert(
            -2, dict(type='RandomVerticalFlip', p=1))
    return cfg


def set_lrs(cfg, lr):
    cfg.lr_config = dict(
            policy='Step', step=200, 
            gamma=lr / cfg.optimizer['lr'])
    cfg.optimizer['lr'] = lr
    return cfg


def cotrain_syctl_mst_cfg_func(cfg, lr):
    cfg = set_syctl_train1(cfg)
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source'] = copy.deepcopy(
            BASIC_MST_DATA_SOURCE_DICT)

    cfg = set_lrs(cfg, lr)
    return cfg


def cotrain_syctl_mst_pair_cfg_func(cfg, lr):
    cfg = set_syctl_train1(cfg)
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source'] = copy.deepcopy(
            BASIC_MST_DATA_SOURCE_DICT)
    cfg.data['train2']['data_source']['type'] = 'MSTPairImageList'
    cfg.data['train2']['type'] = 'ContrastiveTwoImageDataset'

    cfg = set_lrs(cfg, lr)
    return cfg
