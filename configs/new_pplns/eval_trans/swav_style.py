from ..trans_param_setter import TransParamBuilder
BASIC_TRANS_CFG = './configs/benchmarks/linear_classification/imagenet/r18_moco_np.py'


def get_typical_params(args, exp_id, ckpt_path):
    def _cfg_func(cfg):
        cfg.model.pretrained = ckpt_path
        cfg.optimizer['lr'] = 0.3
        cfg.optimizer['weight_decay'] = 1e-6
        cfg.lr_config = dict(policy='CosineAnealing', min_lr=0.)
        return cfg
    param_builder = TransParamBuilder(
            args, exp_id, BASIC_TRANS_CFG, 
            col_name='trans_swav_style',
            cfg_change_func=_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def byol_r18_ep300(args):
    ckpt_path = './work_dirs/pretrains/byol_r18_in_ep300.pth'
    return get_typical_params(
            args, 'byol_r18_ep300', 
            ckpt_path)


def moco_r18_ep300(args):
    ckpt_path = './work_dirs/pretrains/moco_r18_in_ep300_v2.pth'
    return get_typical_params(
            args, 'moco_r18_ep300', 
            ckpt_path)


def simclr_r18_ep300(args):
    ckpt_path = './work_dirs/pretrains/simclr_r18_ep300.pth'
    return get_typical_params(
            args, 'simclr_r18_ep300', 
            ckpt_path)


def simsiam_r18_ep300(args):
    ckpt_path = './work_dirs/pretrains/simsiam_r18_ep300.pth'
    return get_typical_params(
            args, 'simsiam_r18_ep300',
            ckpt_path)
