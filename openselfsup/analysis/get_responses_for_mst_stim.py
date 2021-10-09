import os, sys
import pdb
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from openselfsup.analysis.local_paths import MODEL_KWARGS
from openselfsup.analysis.response_extractor import ResponseExtractor
RESULT_FOLDER = '/mnt/fs4/chengxuz/hippocampus_change/mst_related/mst_stim_embds'
STIM_FOLDER = '/mnt/fs4/chengxuz/hippocampus_change/mst_related/MST'


def get_eval_transforms():
    from torchvision import transforms
    norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg),
            ])
    return transform


def get_train_transforms():
    from openselfsup.utils import build_from_cfg
    from torchvision import transforms
    from openselfsup.datasets.registry import PIPELINES
    img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    train_transforms = [
        dict(type='RandomResizedCrop', size=224),
        dict(type='RandomHorizontalFlip'),
        dict(
            type='RandomAppliedTrans',
            transforms=[
                dict(
                    type='ColorJitter',
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.2)
            ],
            p=0.8),
        dict(type='RandomGrayscale', p=0.2),
        dict(
            type='RandomAppliedTrans',
            transforms=[
                dict(
                    type='GaussianBlur',
                    sigma_min=0.1,
                    sigma_max=2.0)
            ],
            p=0.5),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]
    train_transforms = [build_from_cfg(p, PIPELINES) for p in train_transforms]
    train_transform = transforms.Compose(train_transforms)
    return train_transform

AUG_TO_TRANS_FUNC = {
        'eval': get_eval_transforms,
        'train': get_train_transforms,
        }


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model',
                        type=str, default='simclr_sy_ctl',
                        action='store', choices=MODEL_KWARGS.keys())
    parser.add_argument('--stim_folder',
                        type=str, default='Set 1', action='store')
    parser.add_argument('--aug',
                        type=str, default='eval',
                        action='store', choices=['eval', 'train'])
    parser.add_argument('--suffix',
                        type=str, default=None,
                        action='store', help='Name suffix')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get responses for MST stimuli')
    parser = add_general_argument(parser)
    return parser


def get_imgs(args):
    stim_folder = os.path.join(STIM_FOLDER, args.stim_folder)
    all_imgs = os.listdir(stim_folder)
    all_imgs.sort()
    all_imgs = [
            os.path.join(stim_folder, _img)
            for _img in all_imgs]
    transforms = AUG_TO_TRANS_FUNC[args.aug]()
    all_imgs = [Image.open(_img).convert('RGB') for _img in all_imgs]
    all_imgs = [transforms(_img) for _img in all_imgs]
    return all_imgs


def main():
    parser = get_parser()
    args = parser.parse_args()

    response_extractor = ResponseExtractor(
            layers=[],
            **MODEL_KWARGS[args.which_model])
    all_imgs = get_imgs(args)

    embds = []
    batch_size = 64
    for sta_idx in range(0, len(all_imgs), batch_size):
        end_idx = min(len(all_imgs), sta_idx + batch_size)
        curr_embds = response_extractor.get_embds(all_imgs[sta_idx:end_idx])
        embds.append(curr_embds)
    embds = np.concatenate(embds, axis=0)

    result_folder = os.path.join(
            args.result_folder, args.which_model, 
            args.stim_folder.replace(' ', '_'))
    os.system('mkdir -p ' + result_folder)
    file_name = 'aug_'  + args.aug
    if args.suffix is not None:
        file_name += '_' + args.suffix
    np.save(os.path.join(result_folder, file_name), embds)
    print(result_folder)

    
if __name__ == '__main__':
    main()
