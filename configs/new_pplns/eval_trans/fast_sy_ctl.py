from .fast import get_typical_params


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
