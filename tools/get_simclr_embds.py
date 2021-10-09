import os, sys
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import argparse
import pdb
import tensorflow as tf
from torch.utils.data import RandomSampler
import torch
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm


SIMCLR_MODEL_DIR = '/mnt/fs4/chengxuz/simclr_pretrained_models'
RESULT_SAVE_FOLDER = os.path.join(SIMCLR_MODEL_DIR, 'embd_results')
CKPT_DICTS = {
        'in_r18': '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748',
        'epic_r18': '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_epic/model.ckpt-311748',
        'in_r50': '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res50_full/model.ckpt-311748',
        'epic_r50': '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res50_epic/model.ckpt-311748',
        }
DATASET_FOLDERS = {
        'epic': '/data5/chengxuz/Dataset/epic/rgb_frames',
        'imagenet': '/data5/chengxuz/Dataset/imagenet_raw/train/',
        }
RESNET_DEPTH = 18


def get_parser():
    parser = argparse.ArgumentParser(
            description='To get SimCLR embd')
    parser.add_argument(
            '--gpu', 
            default='0', type=str, 
            action='store', help='GPU index')
    parser.add_argument(
            '--network', 
            default='in_r18', type=str, 
            action='store', help='Which network to use',
            choices=CKPT_DICTS.keys())
    parser.add_argument(
            '--dataset', 
            default='imagenet', type=str, 
            action='store', help='Which dataset to use',
            choices=DATASET_FOLDERS.keys())
    parser.add_argument(
            '--exp', 
            default='aug', type=str, 
            action='store', help='Which exp to run',
            choices=['aug', 'sanity'])
    return parser


def get_simclr_ending_points(inputs):
    sys.path.append(
            os.path.abspath('../simclr'))
    import resnet, run
    import model_util
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(['none'])
    with tf.variable_scope('base_model'):
        model = resnet.resnet_v1(
                resnet_depth=RESNET_DEPTH,
                width_multiplier=1,
                cifar_stem=False,
                drop_final_pool=True)
        output = model(
                inputs['images'], # 0-1, float32
                is_training=False)
    ending_points = resnet.ENDING_POINTS

    # 7*7 embedding
    hiddens_proj = model_util.projection_head(
            tf.reshape(
                output, 
                [-1, output.get_shape().as_list()[-1]]), 
            is_training=False)
    hiddens_proj = tf.reshape(
            hiddens_proj, 
            output.get_shape().as_list()[:-1] + [-1])
    ending_points.append(hiddens_proj)

    # Final embedding
    hiddens_proj = model_util.projection_head(
            tf.reduce_mean(output, axis=[1,2]), 
            is_training=False)
    ending_points.append(hiddens_proj)
    return ending_points


def get_one_image_augs(
        dataset_folder='/data5/chengxuz/Dataset/imagenet_raw/train/',
        img_num=50):
    one_folder = os.path.join(
            dataset_folder,
            np.random.choice(os.listdir(dataset_folder)))
    one_image = os.path.join(
            one_folder,
            np.random.choice(os.listdir(one_folder)))
    img = Image.open(one_image)
    random_resized_crop = transforms.RandomResizedCrop(224)
    totensor = transforms.ToTensor()

    all_imgs = []
    all_params = []
    for _ in range(img_num):
        _param = {}
        _crop_params = random_resized_crop.get_params(
                img, 
                random_resized_crop.scale,
                random_resized_crop.ratio)
        _param['crop'] = _crop_params
        _img = transforms.functional.resized_crop(
                img, 
                _crop_params[0], _crop_params[1], 
                _crop_params[2], _crop_params[3], 
                random_resized_crop.size, 
                random_resized_crop.interpolation)
        _img = totensor(_img)
        if _img.shape[0] == 1:
            _img = _img.repeat(3, 1, 1)
        all_params.append(_param)
        all_imgs.append(_img)
    imgs = torch.stack(all_imgs, axis=0)
    return imgs, all_params, one_image


def create_sess():
    gpu_options = tf.GPUOptions(allow_growth=True)
    SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            ))
    return SESS


def sanity_check(
        ckpt_path='/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748',
        save_path = '/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_sanity.pkl',
        ):
    num_batches = 1000
    batch_size = 50

    input_image = tf.placeholder(
            shape=[batch_size, 224, 224, 3], 
            dtype=tf.float32)
    ending_points = get_simclr_ending_points(
            {'images': input_image})
    penult_ly, embd = ending_points[-2:]

    SESS = create_sess()
    saver = tf.train.Saver()
    saver.restore(SESS, ckpt_path)

    all_params = []
    all_images = []
    all_embds = []

    for _ in tqdm(range(num_batches)):
        curr_imgs = []
        for _ in range(batch_size):
            imgs, _params, _image_path = get_one_image_augs(img_num=1)
            imgs = imgs.numpy()
            while imgs.shape[1] != 3:
                imgs, _params, _image_path = get_one_image_augs(img_num=1)
                imgs = imgs.numpy()
            curr_imgs.append(imgs)
            all_params.append(_params)
            all_images.append(_image_path)
        curr_imgs = np.concatenate(curr_imgs, axis=0)
        curr_imgs = np.transpose(curr_imgs, [0, 2, 3, 1])
        _penult_ly, _embd = SESS.run(
                [penult_ly, embd],
                feed_dict={input_image: curr_imgs})
        all_embds.append([_penult_ly, _embd])

    pickle.dump(
            {
                'params': all_params, 
                'embds': all_embds,
                'images': all_images},
            open(save_path, 'wb'))


def aug_embd(
        dataset_folder='/data5/chengxuz/Dataset/imagenet_raw/train/',
        save_path='/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_embd.pkl',
        ckpt_path='/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18/model.ckpt-311748',
        ):
    num_batches = 200
    batch_size = 50

    input_image = tf.placeholder(
            shape=[batch_size, 224, 224, 3], 
            dtype=tf.float32)
    ending_points = get_simclr_ending_points(
            {'images': input_image})
    penult_ly, embd = ending_points[-2:]

    SESS = create_sess()
    saver = tf.train.Saver()
    saver.restore(SESS, ckpt_path)

    all_params = []
    all_images = []
    all_embds = []

    for _ in tqdm(range(num_batches)):
        curr_imgs, _params, _image_path = get_one_image_augs(dataset_folder=dataset_folder)
        curr_imgs = curr_imgs.numpy()
        all_params.append(_params)
        all_images.append(_image_path)
        curr_imgs = np.transpose(curr_imgs, [0, 2, 3, 1])
        _penult_ly, _embd = SESS.run(
                [penult_ly, embd],
                feed_dict={input_image: curr_imgs})
        all_embds.append([_penult_ly, _embd])

    pickle.dump(
            {
                'params': all_params, 
                'embds': all_embds,
                'images': all_images},
            open(save_path, 'wb'))


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ckpt_path = CKPT_DICTS[args.network]
    global RESNET_DEPTH
    RESNET_DEPTH = int(args.network[-2:])

    if args.exp == 'aug':
        dataset_folder = DATASET_FOLDERS[args.dataset]
        save_path = os.path.join(
                RESULT_SAVE_FOLDER, 
                '{}_{}.pkl'.format(args.network, args.dataset))
        aug_embd(
                ckpt_path=ckpt_path,
                dataset_folder=dataset_folder,
                save_path=save_path,
                )
    else:
        save_path = os.path.join(
                RESULT_SAVE_FOLDER, 
                'sanity_{}_{}.pkl'.format(args.network, args.dataset))
        sanity_check(
                ckpt_path=ckpt_path,
                save_path=save_path,
                )


if __name__ == '__main__':
    #sanity_check()

    #aug_embd()

    #aug_embd(
    #        dataset_folder='/data5/chengxuz/Dataset/epic/rgb_frames',
    #        save_path='/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_epic_embd.pkl',
    #        )

    #aug_embd(
    #        ckpt_path='/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_epic/model.ckpt-311748',
    #        save_path='/mnt/fs4/chengxuz/simclr_pretrained_models/epic_simclr_res18_embd.pkl',
    #        )

    #aug_embd(
    #        ckpt_path='/mnt/fs4/chengxuz/simclr_pretrained_models/simclr_res18_epic/model.ckpt-311748',
    #        dataset_folder='/data5/chengxuz/Dataset/epic/rgb_frames',
    #        save_path='/mnt/fs4/chengxuz/simclr_pretrained_models/epic_simclr_res18_epic_embd.pkl',
    #        )
    main()
