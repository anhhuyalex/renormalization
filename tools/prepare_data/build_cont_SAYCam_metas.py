import os
import numpy as np
from tqdm import tqdm
import pdb
import copy

SAYCAM_jpg_dir = '/data5/chengxuz/Dataset/infant_headcam/jpgs_extracted'
#output_meta_path = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_bench.txt'
output_meta_path = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_bench_ep300.txt'

video_dirs = []
sup_dirs = os.listdir(SAYCAM_jpg_dir)
for _sup_dir in sup_dirs:
    curr_video_dirs = os.listdir(os.path.join(SAYCAM_jpg_dir, _sup_dir))
    curr_video_dirs = [
            os.path.join(_sup_dir, _video_dir) 
            for _video_dir in curr_video_dirs]
    curr_video_dirs = sorted(curr_video_dirs)
    video_dirs.append(curr_video_dirs)

video_dirs = sorted(
        video_dirs, key=lambda x: len(x), 
        reverse=True)
video_list = copy.copy(video_dirs[0])
for _video_dir in video_dirs[1:]:
    new_video_list = []
    curr_len = len(video_list)
    org_idx = 0
    curr_idx = 0
    while curr_idx < len(_video_dir):
        added_len = (curr_len - org_idx) // (len(_video_dir) - curr_idx)
        new_video_list.extend(video_list[org_idx:(org_idx+added_len)])
        org_idx += added_len
        new_video_list.append(_video_dir[curr_idx])
        curr_idx += 1
    if org_idx < curr_len:
        new_video_list.extend(video_list[org_idx:curr_len])
    video_list = new_video_list

all_frame_nums = []
for _video_dir in tqdm(video_list):
    curr_frames = os.listdir(os.path.join(SAYCAM_jpg_dir, _video_dir))
    num_frames = len(curr_frames)
    all_frame_nums.append(num_frames)

#num_epochs = 200
num_epochs = 300
all_epoch_meta = []
video_idx = 0
all_num_frames = sum(all_frame_nums)
curr_num_frames = 0
while curr_num_frames < all_num_frames:
    delta_vd_idx = 0
    delta_num_frames = 0
    wanted_delta_num_frames = (all_num_frames - curr_num_frames) \
                              // (num_epochs - len(all_epoch_meta))
    while delta_num_frames < wanted_delta_num_frames:
        delta_num_frames += all_frame_nums[video_idx + delta_vd_idx]
        delta_vd_idx += 1
    all_epoch_meta.append(
            ','.join(video_list[video_idx:(video_idx+delta_vd_idx)]) + '\n')
    curr_num_frames += delta_num_frames
    video_idx += delta_vd_idx

with open(output_meta_path, 'w') as fout:
    fout.writelines(all_epoch_meta)

'''
output_meta_path = '/mnt/fs1/Dataset/infant_headcam/num_frames_meta.txt'

with open(output_meta_path, 'w') as fout:
    for _video_dir, num_frames in zip(video_list, all_frame_nums):
        fout.write('%s, %i\n' % (_video_dir, num_frames))
'''
