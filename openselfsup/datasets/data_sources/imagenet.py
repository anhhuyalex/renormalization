from ..registry import DATASOURCES
from .image_list import ImageList
import copy
import numpy as np
import os


@DATASOURCES.register_module
class ImageNet(ImageList):

    def __init__(self, root, list_file, memcached, mclient_path):
        super(ImageNet, self).__init__(
            root, list_file, memcached, mclient_path)


@DATASOURCES.register_module
class ImageNetCont(ImageList):
    def __init__(
            self, num_epochs, keep_prev_epoch_num, 
            root, list_file, memcached, mclient_path, 
            data_len=None, accu=False):
        super().__init__(
            root, list_file, memcached, mclient_path)
        self.raw_fns = copy.copy(self.fns)
        self.num_epochs = num_epochs
        self.keep_prev_epoch_num = keep_prev_epoch_num
        if data_len is None:
            self.data_len = len(self.raw_fns)
        else:
            self.data_len = data_len
        self.accu = accu
        self.set_epoch(0)

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        len_data = len(self.raw_fns)
        div_num = self.num_epochs + self.keep_prev_epoch_num
        if not self.accu:
            start_idx = int(len_data * (epoch * 1.0 / div_num))
        else:
            start_idx = 0
        end_idx = int(len_data * ((epoch+self.keep_prev_epoch_num) \
                                  * 1.0 / div_num))
        new_fns = self.raw_fns[start_idx : end_idx]
        self.fns = np.random.choice(
                new_fns, self.data_len, replace=True)


@DATASOURCES.register_module
class ImageNetCntAccuSY(ImageList):
    def __init__(
            self, num_epochs, keep_prev_epoch_num, 
            root, list_file, memcached, mclient_path, 
            sy_root, sy_epoch_meta_path, sy_file_num_meta_path, sy_end_epoch,
            data_len=None,
            ):
        super().__init__(
            root, list_file, memcached, mclient_path)
        self.raw_fns = copy.copy(self.fns)
        self.num_epochs = num_epochs
        self.keep_prev_epoch_num = keep_prev_epoch_num
        if data_len is None:
            self.data_len = len(self.raw_fns)
        else:
            self.data_len = data_len
        self.sy_root = sy_root
        self.sy_epoch_meta_path = sy_epoch_meta_path
        self.sy_file_num_meta_path = sy_file_num_meta_path
        self.sy_end_epoch = sy_end_epoch
        self.load_sy_metas()

    def load_sy_metas(self):
        self.sy_num_frames_meta = {}
        with open(self.sy_file_num_meta_path, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.sy_num_frames_meta[video_name] = num_frames

        with open(self.sy_epoch_meta_path, 'r') as fin:
            all_lines = fin.readlines()
        self.sy_epoch_meta = [l.strip() for l in all_lines]

        sy_fns_up_to_end = []
        for _ep in range(self.sy_end_epoch):
            video_dirs = self.sy_epoch_meta[_ep].split(',')
            for _dir in video_dirs:
                frames = [
                        os.path.join(
                            self.sy_root, _dir, '%06i.jpg' % (_frame+1))
                        for _frame in range(self.sy_num_frames_meta[_dir])]
                sy_fns_up_to_end.extend(frames)
        np.random.seed(self.sy_end_epoch)
        self.sy_fns = np.random.choice(
                sy_fns_up_to_end, self.data_len, replace=True)

    def set_epoch(self, epoch):
        assert epoch >= self.sy_end_epoch
        np.random.seed(epoch)
        len_data = len(self.raw_fns)
        div_num = self.num_epochs + self.keep_prev_epoch_num
        start_idx = int(len_data * (self.sy_end_epoch * 1.0 / div_num))
        end_idx = int(len_data * ((epoch+self.keep_prev_epoch_num) \
                                  * 1.0 / div_num))
        if start_idx < end_idx:
            new_fns = self.raw_fns[start_idx : end_idx]
        else:
            new_fns = []
        new_fns = np.concatenate([new_fns, self.sy_fns])
        self.fns = np.random.choice(
                new_fns, self.data_len, replace=True)


@DATASOURCES.register_module
class ImageNetBatchLD(ImageList):
    def __init__(
            self, batch_size, no_cate_per_batch,
            root, list_file, memcached, mclient_path):
        super().__init__(
            root, list_file, memcached, mclient_path)
        self.raw_fns = np.asarray(copy.copy(self.fns))
        self.batch_size = batch_size
        self.no_cate_per_batch = no_cate_per_batch
        self.get_class_metas()
        self.set_epoch(0)

    def get_class_metas(self):
        class_raw_names = [
                os.path.basename(os.path.dirname(_fn)) 
                for _fn in self.raw_fns]
        class_raw_names = np.asarray(class_raw_names)
        unique_class_names = np.unique(class_raw_names)
        class_indexes = {}
        class_no_imgs = {}
        for each_class in unique_class_names:
            class_indexes[each_class] = set(
                    np.where(class_raw_names==each_class)[0])
            class_no_imgs[each_class] = len(class_indexes[each_class])

        self.class_indexes = class_indexes
        self.class_no_imgs = class_no_imgs

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        len_data = len(self.raw_fns)
        left_imgs = copy.deepcopy(self.class_no_imgs)
        left_indexes = copy.deepcopy(self.class_indexes)
        new_fns = []

        class_not_enough = 0
        img_not_enough = 0
        for _ in range(len_data // self.batch_size):
            all_possible_classes = list(left_imgs.keys())
            cls_replace = len(all_possible_classes) < self.no_cate_per_batch
            batch_classes = np.random.choice(
                    all_possible_classes, self.no_cate_per_batch,
                    replace=cls_replace)
            img_index = set([])
            for each_class in batch_classes:
                img_index = img_index.union(left_indexes[each_class])
            img_replace = len(img_index) < self.batch_size
            img_index = np.random.choice(
                    np.asarray(list(img_index)), 
                    self.batch_size, replace=img_replace)
            for each_class in batch_classes:
                left_indexes[each_class] \
                        = left_indexes[each_class].difference(img_index)
                left_imgs[each_class] = len(left_indexes[each_class])
                if left_imgs[each_class] == 0:
                    left_imgs.pop(each_class)

            if cls_replace:
                class_not_enough += 1
            if img_replace:
                img_not_enough += 1
            new_fns.append(img_index)
        new_fns = np.asarray(new_fns)
        new_fns = new_fns[np.random.permutation(new_fns.shape[0])]
        new_fns = new_fns.reshape([-1])
        self.fns = self.raw_fns[new_fns]
