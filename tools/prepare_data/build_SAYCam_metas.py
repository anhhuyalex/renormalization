import os
import numpy as np
from tqdm import tqdm
import pdb


SAYCAM_jpg_dir = '/data5/chengxuz/Dataset/infant_headcam/jpgs_extracted'
output_meta_path = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_curr.txt'

video_dirs = []
sup_dirs = os.listdir(SAYCAM_jpg_dir)
for _sup_dir in sup_dirs:
    curr_video_dirs = os.listdir(os.path.join(SAYCAM_jpg_dir, _sup_dir))
    curr_video_dirs = [
            os.path.join(_sup_dir, _video_dir) 
            for _video_dir in curr_video_dirs]
    video_dirs.extend(curr_video_dirs)

sample_length = 25 * 64 * 4
sample_rate = 25 * 64 * 2
total_len = 0
all_start_pos = []
for _video_dir in tqdm(video_dirs):
    curr_frames = os.listdir(os.path.join(SAYCAM_jpg_dir, _video_dir))
    curr_frames = sorted(curr_frames)
    num_frames = len(curr_frames)
    curr_samples = range(0, num_frames-sample_length, sample_rate)
    curr_start_pos = [
            os.path.join(
                _video_dir, 
                curr_frames[_sample]) + '\n'
            for _sample in curr_samples]
    all_start_pos.extend(curr_start_pos)
    total_len += len(curr_samples)

print(len(all_start_pos))
with open(output_meta_path, 'w') as fout:
    fout.writelines(all_start_pos)
