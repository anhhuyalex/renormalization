from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Sample images and repeat them')
parser.add_argument('num', help='Number of images to be sampled.', type=int)
parser.add_argument('--labeled', help='Whether labeled or not', action='store_true')
args = parser.parse_args()

if not args.labeled:
    train_meta = 'data/imagenet/meta/train.txt'
    name_pattern = 'data/imagenet/meta/train_%irep.txt'
else:
    train_meta = 'data/imagenet/meta/train_labeled.txt'
    name_pattern = 'data/imagenet/meta/train_labeled_%irep.txt'

with open(train_meta, 'r') as f:
    paths = f.readlines()

np.random.seed(0)
sampled_imgs = np.random.choice(paths, args.num, replace=False)
resampled_imgs = np.random.choice(sampled_imgs, len(paths), replace=True)
with open(name_pattern % args.num, 'w') as f:
    f.writelines(resampled_imgs)
