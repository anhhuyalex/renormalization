from mmcv import Config
from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pdb
from torch.utils.data import RandomSampler
import torch
import torchvision.transforms as transforms
import pickle


MOCO_CFG_PATH = os.path.expanduser('~/openselfsup/configs/selfsup/moco/r18_v2.py')
LINEAR_CFG_PATH = os.path.expanduser('~/openselfsup/configs/benchmarks/linear_classification/imagenet/r18_moco_sl.py')
MOCO_MODEL_PATH = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/selfsup/moco/r18_v2/epoch_200.pth'


def get_data_loader_from_cfg(
        cfg_path=LINEAR_CFG_PATH):
    cfg = Config.fromfile(cfg_path)
    cfg.data.train.data_source.memcached = False
    dataset = build_dataset(cfg.data.train)
    data_loader = DataLoader(
            dataset,
            batch_size=50,
            sampler=RandomSampler(dataset),
            pin_memory=False)
    return data_loader


def get_model(
        model_cfg_path=MOCO_CFG_PATH,
        ckpt_path=MOCO_MODEL_PATH):
    model = build_model(
            Config.fromfile(model_cfg_path).model)
    model_dict = torch.load(ckpt_path)
    model.load_state_dict(model_dict['state_dict'])
    model = model.encoder_q.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def sanity_check_embds():
    data_loader = get_data_loader_from_cfg()
    model = get_model()

    all_labels = []
    all_embds = []

    for idx, data in enumerate(data_loader):
        all_labels.append(data['gt_label'])
        embd = model(data['img'].cuda())
        all_embds.append(embd[0].cpu().numpy())
        if idx % 100 == 0:
            print(idx)
        if idx >= 1000:
            break

    SAVE_PATH = '/mnt/fs4/chengxuz/openselfsup_models/moco_r18_in_embds.pkl'
    pickle.dump(
            {'lbl': all_labels, 'embds': all_embds},
            open(SAVE_PATH, 'wb'))


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
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

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
        _img = normalize(_img)

        all_params.append(_param)
        all_imgs.append(_img)
    imgs = torch.stack(all_imgs, axis=0)
    return imgs, all_params, one_image


def get_img_aug_embds(
        dataset_folder='/data5/chengxuz/Dataset/imagenet_raw/train/',
        save_path='/mnt/fs4/chengxuz/openselfsup_models/moco_r18_in_aug_embds.pkl'):
    model = get_model()

    all_params = []
    all_images = []
    all_embds = []

    for idx in range(200):
        imgs, _params, _image_path = get_one_image_augs()
        embd = model(imgs.cuda())

        all_embds.append(embd[0].cpu().numpy())
        all_params.append(_params)
        all_images.append(_image_path)
        if idx % 50 == 0:
            print(idx)

    pickle.dump(
            {
                'params': all_params, 
                'embds': all_embds,
                'images': all_images},
            open(save_path, 'wb'))


def get_epic_img_aug_embds():
    model = get_model()

    all_params = []
    all_images = []
    all_embds = []

    for idx in range(200):
        imgs, _params, _image_path = get_one_image_augs(
                dataset_folder='/data5/chengxuz/Dataset/epic/rgb_frames',
                )
        embd = model(imgs.cuda())

        all_embds.append(embd[0].cpu().numpy())
        all_params.append(_params)
        all_images.append(_image_path)
        if idx % 50 == 0:
            print(idx)

    SAVE_PATH = '/mnt/fs4/chengxuz/openselfsup_models/moco_r18_in_epic_aug_embds.pkl'
    pickle.dump(
            {
                'params': all_params, 
                'embds': all_embds,
                'images': all_images},
            open(SAVE_PATH, 'wb'))


if __name__ == '__main__':
    #sanity_check_embds()
    #get_one_image_augs()

    #get_img_aug_embds()

    #get_img_aug_embds(
    #        dataset_folder='/data5/chengxuz/Dataset/epic/rgb_frames',
    #        save_path='/mnt/fs4/chengxuz/openselfsup_models/moco_r18_in_epic_aug_embds.pkl',
    #        )

    get_img_aug_embds(
            dataset_folder='/data5/chengxuz/Dataset/yfcc/imagenet_size_jpgs_from_tfr/split_00/',
            save_path='/mnt/fs4/chengxuz/openselfsup_models/moco_r18_in_yfcc_aug_embds.pkl',
            )
