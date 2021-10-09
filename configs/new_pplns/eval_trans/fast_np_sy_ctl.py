from .fast_np import get_typical_params


def sy_ctl_simclr_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18', 
            ckpt_path)


def sy_ctl_moco_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18', 
            ckpt_path)
