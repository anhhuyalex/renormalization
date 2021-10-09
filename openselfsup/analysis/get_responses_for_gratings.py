import os, sys
import pdb
import numpy as np
import argparse
from PIL import Image
from openselfsup.analysis.local_paths import MODEL_KWARGS
from openselfsup.analysis.response_extractor \
        import ResponseExtractor, get_transforms
RESULT_FOLDER = '/mnt/fs4/chengxuz/rodent_dev/responses/'
ALXNT_LAYERS = [
        'backbone.features.2',
        'backbone.features.6',
        'backbone.features.10',
        'backbone.features.13',
        'backbone.features.16']


def add_general_argument(parser):
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--which_model', 
                        type=str, required=True,
                        action='store', choices=MODEL_KWARGS.keys())
    parser.add_argument('--grating_folder', 
                        type=str, required=True,
                        action='store')
    parser.add_argument('--resize_size', 
                        type=int, default=64,
                        action='store', help='Length of shorter edge in resize')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Get responses for gratings')
    parser = add_general_argument(parser)
    return parser


def get_imgs(args):
    all_imgs = os.listdir(args.grating_folder)
    all_imgs.sort()
    all_imgs = [
            os.path.join(args.grating_folder, _img)
            for _img in all_imgs]
    transforms = get_transforms(args.resize_size)
    all_imgs = [Image.open(_img).convert('RGB') for _img in all_imgs]
    all_imgs = [transforms(_img) for _img in all_imgs]
    return all_imgs


def main():
    parser = get_parser()
    args = parser.parse_args()

    response_extractor = ResponseExtractor(
            layers=ALXNT_LAYERS,
            **MODEL_KWARGS[args.which_model])
    all_imgs = get_imgs(args)
    act_dict = response_extractor.get_activations(all_imgs)

    result_folder = os.path.join(
            args.result_folder, args.which_model, 
            os.path.basename(args.grating_folder), 
            'resize_%i' % args.resize_size)
    os.system('mkdir -p ' + result_folder)

    for key, value in act_dict.items():
        np.save(os.path.join(result_folder, key), value)
    print(result_folder)

    
if __name__ == '__main__':
    main()
