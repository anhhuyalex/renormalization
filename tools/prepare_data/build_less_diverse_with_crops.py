from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import os


parser = argparse.ArgumentParser(
    description='Sample images and repeat them')
parser.add_argument('num', help='Number of images to be sampled.', type=int)
args = parser.parse_args()

raw_folder = '/data5/chengxuz/Dataset/imagenet_raw/train/'
output_dir = '/data5/chengxuz/Dataset/resampled_imagenet/'
train_meta = 'data/imagenet/meta/train_%irep.txt' % args.num
new_train_meta = 'data/imagenet/meta/train_%irep_with_crop.txt' % args.num
os.system('mkdir -p ' + output_dir)

with open(train_meta, 'r') as f:
    paths = f.readlines()

np.random.seed(0)
sampled_imgs = np.random.choice(paths, args.num, replace=False)
resampled_imgs = np.random.choice(sampled_imgs, len(paths), replace=True)

random_resized_crop = transforms.RandomResizedCrop(300)
totensor = transforms.ToTensor()
new_paths = []

for _idx, _img_path in tqdm(enumerate(resampled_imgs)):
    img = Image.open(raw_folder + _img_path.strip())
    _img = random_resized_crop(img)
    _img = totensor(_img)
    _img = _img.numpy()
    _img = (_img * 255).astype(np.uint8)
    _img = np.transpose(_img, [1, 2, 0])
    img = Image.fromarray(_img)
    _new_path = output_dir + 'img_%i.jpg' % _idx
    new_paths.append(_new_path + '\n')
    img.save(_new_path)

with open(new_train_meta, 'w') as f:
    f.writelines(new_paths)
