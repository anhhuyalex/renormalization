import pdb
import numpy as np

num_frames_meta_path = '/mnt/fs1/Dataset/infant_headcam/num_frames_meta.txt'
with open(num_frames_meta_path, 'r') as fin:
    all_lines = fin.readlines()
    all_video_names = [_line.split(',')[0] for _line in all_lines]

np.random.seed(0)
train_meta_path = '/mnt/fs1/Dataset/infant_headcam/embd_train_meta.txt'
num_train_vds = int(len(all_video_names) / 21.0 * 20)
train_video_names = all_video_names[:num_train_vds]
with open(train_meta_path, 'w') as fout:
    for line in train_video_names:
        fout.write(line + '\n')

val_meta_path = '/mnt/fs1/Dataset/infant_headcam/embd_val_meta.txt'
val_video_names = all_video_names[num_train_vds:]
with open(val_meta_path, 'w') as fout:
    for line in val_video_names:
        fout.write(line + '\n')
