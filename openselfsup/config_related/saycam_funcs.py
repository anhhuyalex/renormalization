import copy

SAYCam_root = '/data5/chengxuz/Dataset/infant_headcam/jpgs_extracted'
SAYCam_list_file = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_curr.txt'
BASIC_DATA_SOURCE_DICT = {
        'type': 'SAYCam',
        'list_file': SAYCam_list_file,
        'root': SAYCam_root,
        'batch_size': 256,
        }


def add_vertical_flip(cfg):
    if 'pipeline' in cfg.data['train']:
        cfg.data['train']['pipeline'].insert(
                -2, dict(type='RandomVerticalFlip', p=1))
    elif 'pipeline1' in cfg.data['train']:
        cfg.data['train']['pipeline1'].insert(
                -2, dict(type='RandomVerticalFlip', p=1))
        cfg.data['train']['pipeline2'].insert(
                -2, dict(type='RandomVerticalFlip', p=1))
    else:
        raise NotImplementedError
    return cfg


def random_saycam_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(BASIC_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['all_random'] = True
    cfg = add_vertical_flip(cfg)
    return cfg


def random_saycam_1M_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(BASIC_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['all_random'] = True
    cfg.data['train']['data_source']['set_len'] = 1281167
    cfg = add_vertical_flip(cfg)
    return cfg


def saycam_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(BASIC_DATA_SOURCE_DICT)
    cfg = add_vertical_flip(cfg)
    return cfg


def random_saycam_two_img_cfg_func(cfg):
    cfg = random_saycam_cfg_func(cfg)
    cfg.data['train']['data_source']['type'] = 'SAYCamTwoImage'
    cfg.data['train']['type'] = 'ContrastiveTwoImageDataset'
    return cfg


def random_saycam_two_img_rd_cfg_func(cfg):
    cfg = random_saycam_cfg_func(cfg)
    cfg.data['train']['data_source']['type'] = 'SAYCamTwoImageRandom'
    cfg.data['train']['type'] = 'ContrastiveTwoImageDataset'
    return cfg


SAYCam_list_file_cont = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_bench.txt'
SAYCam_list_file_cont_ep300 = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_bench_ep300.txt'
SAYCam_num_frames_meta_file = '/mnt/fs1/Dataset/infant_headcam/num_frames_meta.txt'
CONT_DATA_SOURCE_DICT = {
        'type': 'SAYCamCont',
        'list_file': SAYCam_list_file_cont,
        'num_frames_meta_file': SAYCam_num_frames_meta_file,
        'root': SAYCam_root,
        }
def cont_saycam_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg = add_vertical_flip(cfg)
    return cfg


def cont_saycam_ep300_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['list_file'] = SAYCam_list_file_cont_ep300
    cfg = add_vertical_flip(cfg)
    return cfg


def cont_accu_saycam_ep300_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['list_file'] = SAYCam_list_file_cont_ep300
    cfg.data['train']['data_source']['type'] = 'SAYCamContAccu'
    cfg = add_vertical_flip(cfg)
    return cfg


def cotrain_saycam_ep300_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['list_file'] = SAYCam_list_file_cont_ep300
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['type'] = 'SAYCamContAccu' 
    cfg = add_vertical_flip(cfg)
    return cfg


def cotrain_saycam_half_ep300_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['list_file'] = SAYCam_list_file_cont_ep300
    cfg.data['train']['data_source']['one_epoch_img_num'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['type'] = 'SAYCamContAccu' 
    cfg = add_vertical_flip(cfg)
    return cfg


def scale_second_dataset(scale_ratio):
    def _func(cfg):
        cfg.data['train2']['data_source']['one_epoch_img_num'] *= scale_ratio
        return cfg
    return _func


def cnd_storage_cfg_func(cfg):
    cfg.data['train2']['data_source']['type'] = 'SAYCamCndCont'

    img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Assuming first one is RandomResizedCrop
    if 'pipeline' in cfg.data['train1']:
        input_size = cfg.data['train1']['pipeline'][0]['size']
    elif 'pipeline1' in cfg.data['train1']:
        input_size = cfg.data['train1']['pipeline1'][0]['size']
    else:
        input_size = 224
    resize_size = int(256.0 * input_size / 224)
    test_pipeline = [
        dict(type='Resize', size=resize_size),
        dict(type='CenterCrop', size=input_size),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]
    cfg.data['train2']['data_source']['pipeline'] = test_pipeline
    return cfg


def min_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'min'
    return cfg


def max100_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_100'
    return cfg


def max500_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_500'
    return cfg


def max2000_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_2000'
    return cfg


def max4000_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_4000'
    return cfg


def max6000_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_6000'
    return cfg


def max10k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_10000'
    return cfg


def max20k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_20000'
    return cfg


def max40k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_40000'
    return cfg


def max80k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_80000'
    return cfg


def summax2000_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'sum_max_2000'
    return cfg


def summax4000_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'sum_max_4000'
    return cfg


def summax6000_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'sum_max_6000'
    return cfg


def summax12k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'sum_max_12000'
    return cfg


HIPP_PRED_DICT = {
        'type': 'HippHead',
        'hipp_mlp': dict(type='NonLinearNeckV1',
                         in_channels=256, hid_channels=256,
                         out_channels=256, with_avg_pool=False),
        'pred_mlp': dict(type='NonLinearNeckV1',
                         in_channels=384, hid_channels=384,
                         out_channels=128, with_avg_pool=False),
        }
def hipp_pred_saycam_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(BASIC_DATA_SOURCE_DICT)
    cfg = add_vertical_flip(cfg)
    cfg.model['hipp_head'] = HIPP_PRED_DICT
    return cfg
