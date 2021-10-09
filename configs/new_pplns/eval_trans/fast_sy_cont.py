from .fast import get_typical_params


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
