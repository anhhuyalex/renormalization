import copy
from . import saycam_funcs


def imgnt_cont_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetCont'
    cfg.data['train']['data_source']['num_epochs'] = 300
    cfg.data['train']['data_source']['keep_prev_epoch_num'] = 2
    return cfg


def cotr_imgnt_cont_cfg_func(cfg):
    cfg = imgnt_cont_cfg_func(cfg)
    cfg.data['train']['data_source']['data_len'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['accu'] = True
    return cfg


def imgnt_batchLD_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetBatchLD'
    cfg.data['train']['data_source']['batch_size'] = 256
    cfg.data['train']['data_source']['no_cate_per_batch'] = 10
    return cfg


def imgnt_batchLD512_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetBatchLD'
    cfg.data['train']['data_source']['batch_size'] = 512
    cfg.data['train']['data_source']['no_cate_per_batch'] = 10
    return cfg


def cotr_imgnt_cont_acsy_cfg_func(cfg, end_epoch=100):
    cfg = imgnt_cont_cfg_func(cfg)
    cfg.data['train']['data_source']['data_len'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source'].update(
            dict(
                type='ImageNetCntAccuSY',
                sy_root=saycam_funcs.SAYCam_root,
                sy_epoch_meta_path=saycam_funcs.SAYCam_list_file_cont_ep300,
                sy_file_num_meta_path=saycam_funcs.SAYCam_num_frames_meta_file,
                sy_end_epoch=end_epoch,
                ))
    return cfg
