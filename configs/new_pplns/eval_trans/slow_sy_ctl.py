from .slow import get_typical_params


def sy_ctl_simclr_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18', 
            ckpt_path)


def sy_ctl_simclr_r18_ep290(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18_ep290.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18_ep290', 
            ckpt_path)


def sy_ctl_ep200_simclr_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY_ep200/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_ep200_simclr_r18', 
            ckpt_path)


def sy_ctl_moco_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18', 
            ckpt_path)


def sy_ctl_moco_r18_ep290(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18_ep290.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18_ep290', 
            ckpt_path)


def sy_ctl_ep200_moco_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY_ep200/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_ep200_moco_r18',
            ckpt_path)


def sy_ctl_simclr_r18_wd(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18_wd', 
            ckpt_path, wd=1e-4)


def sy_ctl_simclr_r18_lr5(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18_lr5',
            ckpt_path, lr=0.05)


def sy_ctl_simclr_r18_lr10(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18_lr10',
            ckpt_path, lr=0.1)


def sy_ctl_simclr_r18_lr50(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_simclr_r18_lr50',
            ckpt_path, lr=0.5)


def sy_ctl_moco_r18_wd(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18_wd', 
            ckpt_path, wd=1e-4)


def sy_ctl_moco_r18_lr5(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18_lr5',
            ckpt_path, lr=0.05)


def sy_ctl_moco_r18_lr10(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18_lr10',
            ckpt_path, lr=0.1)


def sy_ctl_moco_r18_lr50(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_moco_r18_lr50',
            ckpt_path, lr=0.5)


def sy_ctl_byol_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/byol_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_byol_r18',
            ckpt_path)


def sy_ctl_byolneg_r18(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/byolneg_r18.pth'
    return get_typical_params(
            args, 'sy_ctl_byolneg_r18',
            ckpt_path)


def sy_ctl_simsiam_r18_ep20(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simsiam_r18_ep20.pth'
    return get_typical_params(
            args, 'sy_ctl_simsiam_r18_ep20',
            ckpt_path)


def sy_ctl_simsiam_r18_ep110(args):
    ckpt_path = './work_dirs/pretrains/ctl_SY/simsiam_r18_ep110.pth'
    return get_typical_params(
            args, 'sy_ctl_simsiam_r18_ep110',
            ckpt_path)
