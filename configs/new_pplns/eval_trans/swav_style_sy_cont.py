from .swav_style import get_typical_params


def sy_cont_simclr_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_cont_simclr_r18', 
            ckpt_path)


def sy_cont_moco_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_cont_moco_r18', 
            ckpt_path)


def sy_cont_byol_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/byol_r18.pth'
    return get_typical_params(
            args, 'sy_cont_byol_r18',
            ckpt_path)
