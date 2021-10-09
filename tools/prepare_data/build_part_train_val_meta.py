from tqdm import tqdm
import numpy as np


def get_mapping(meta):
    with open(meta, 'r') as f:
        lines = f.readlines()
    paths = [l.strip().split()[0] for l in lines]
    labels = [int(l.strip().split()[1]) for l in lines]
    mapping = {}
    for p,l in tqdm(zip(paths, labels)):
        if l not in mapping:
            mapping[l] = [p]
        else:
            mapping[l].append(p)
    return mapping


train_meta = 'data/imagenet/meta/train_labeled.txt'
val_meta = 'data/imagenet/meta/val_labeled_new.txt'

train_mapping = get_mapping(train_meta)
val_mapping = get_mapping(val_meta)

num_classes = 100
num_imgs = 200

np.random.seed(0)
sampled_classes = np.random.permutation(1000)[:num_classes]
all_lines = []
for _class in sampled_classes:
    train_imgs = np.random.permutation(train_mapping[_class])[:num_imgs]
    for _img in train_imgs:
        all_lines.append('train/' + _img + ' ' + str(_class) + '\n')

for _class in sampled_classes:
    for _img in val_mapping[_class]:
        all_lines.append('val/' + _img + ' ' + str(_class) + '\n')
with open('data/imagenet/meta/part_train_val_labeled.txt', 'w') as f:
    f.writelines(all_lines)
