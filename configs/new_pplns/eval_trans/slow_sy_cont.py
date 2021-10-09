from .slow import get_typical_params


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


def sy_cont_simclr_r18_wd(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/simclr_r18.pth'
    return get_typical_params(
            args, 'sy_cont_simclr_r18_wd', 
            ckpt_path, wd=1e-4)


def sy_cont_moco_r18_wd(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/moco_r18.pth'
    return get_typical_params(
            args, 'sy_cont_moco_r18_wd', 
            ckpt_path, wd=1e-4)


def sy_cont_byol_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/byol_r18.pth'
    return get_typical_params(
            args, 'sy_cont_byol_r18',
            ckpt_path)


def sy_cont_byol_ep80_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/byol_r18_ep80.pth'
    return get_typical_params(
            args, 'sy_cont_byol_ep80_r18',
            ckpt_path)


def sy_cont_ep200_moco_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY_ep200/moco_r18.pth'
    return get_typical_params(
            args, 'sy_cont_ep200_moco_r18',
            ckpt_path)


def sy_cont_simsiam_ep10_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/simsiam_r18_ep10.pth'
    return get_typical_params(
            args, 'sy_cont_simsiam_ep10_r18',
            ckpt_path)


def sy_cont_simsiam_ep100_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/simsiam_r18_ep100.pth'
    return get_typical_params(
            args, 'sy_cont_simsiam_ep100_r18',
            ckpt_path)


def sy_cont_byolneg_ep260_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/byolneg_r18_ep260.pth'
    return get_typical_params(
            args, 'sy_cont_byolneg_ep260_r18',
            ckpt_path)


def sy_cont_byolneg_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_SY/byolneg_r18.pth'
    return get_typical_params(
            args, 'sy_cont_byolneg_r18',
            ckpt_path)
