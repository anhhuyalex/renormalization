from .slow import get_typical_params


def in_cont_simclr_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_IN/simclr_r18.pth'
    return get_typical_params(
            args, 'in_cont_simclr_r18', 
            ckpt_path)


def in_cont_moco_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_IN/moco_r18.pth'
    return get_typical_params(
            args, 'in_cont_moco_r18', 
            ckpt_path)


def in_cont_byol_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_IN/byol_r18.pth'
    return get_typical_params(
            args, 'in_cont_byol_r18', 
            ckpt_path)


def in_cont_byolneg_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_IN/byolneg_r18.pth'
    return get_typical_params(
            args, 'in_cont_byolneg_r18',
            ckpt_path)


def in_cont_simsiam_ep30_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_IN/simsiam_r18_ep30.pth'
    return get_typical_params(
            args, 'in_cont_simsiam_ep30_r18',
            ckpt_path)


def in_cont_simsiam_ep100_r18(args):
    ckpt_path = './work_dirs/pretrains/cont_IN/simsiam_r18_ep100.pth'
    return get_typical_params(
            args, 'in_cont_simsiam_ep100_r18',
            ckpt_path)
