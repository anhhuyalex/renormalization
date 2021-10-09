from ..registry import DATASOURCES
from .image_list import ImageList
import copy
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm

from openselfsup.framework.dist_utils import get_dist_info, gather_tensors_batch

FPS_RATE = 25

@DATASOURCES.register_module
class SAYCam(ImageList):
    def __init__(
            self, root, list_file, batch_size, all_random=False,
            set_len=None,
            **kwargs):
        super().__init__(
            root, list_file, 
            **kwargs)
        self.base_fns = copy.deepcopy(self.fns)
        self.batch_size = batch_size
        self.all_random = all_random
        self.set_len = set_len
        self.set_epoch(0)

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        _, world_size = get_dist_info()
        assert self.batch_size % world_size == 0
        sub_batch_size = self.batch_size // world_size
        fns = []
        curr_order = np.random.permutation(self.base_fns)

        for _curr_start_frame in curr_order:
            frame_idx = os.path.basename(_curr_start_frame)
            frame_idx = frame_idx.split('.')[0]
            sta_frame_idx = int(frame_idx) + np.random.randint(FPS_RATE)
            all_frame_idxs = [
                    sta_frame_idx + _batch_idx*FPS_RATE \
                    for _batch_idx in range(self.batch_size)]
            frame_dir = os.path.dirname(_curr_start_frame)
            all_frame_path = [
                    os.path.join(frame_dir, '%06i.jpg' % _frame_idx) \
                    for _frame_idx in all_frame_idxs]
            all_frame_path = np.asarray(all_frame_path)
            # To address the distributed sampler
            all_frame_path = np.transpose(
                    all_frame_path.reshape([world_size, sub_batch_size]),
                    [1, 0]).reshape(-1)
            fns.append(all_frame_path)
        self.fns = np.concatenate(fns)
        if self.all_random:
            self.fns = np.random.permutation(self.fns)
        if self.set_len is not None:
            self.fns = self.fns[:self.set_len]


@DATASOURCES.register_module
class SAYCamTwoImage(SAYCam):
    def set_epoch(self, epoch):
        super().set_epoch(epoch)

        self.fns1 = self.fns
        fns2 = []
        for each_jpg in self.fns1:
            frame_dir = os.path.dirname(each_jpg)
            frame_idx = int(os.path.basename(each_jpg).split('.')[0])
            frame_idx += 2
            new_jpg = os.path.join(frame_dir, '%06i.jpg' % frame_idx)
            fns2.append(new_jpg)
        self.fns2 = np.asarray(fns2)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img1 = self.mc_loader(self.fns1[idx])
            img2 = self.mc_loader(self.fns2[idx])
        else:
            img1 = Image.open(self.fns1[idx])
            img2 = Image.open(self.fns2[idx])
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        assert not self.has_labels
        return img1, img2


@DATASOURCES.register_module
class SAYCamTwoImageRandom(SAYCamTwoImage):
    def set_epoch(self, epoch):
        SAYCam.set_epoch(self, epoch)

        if getattr(self, 'cache_dir_content', None) is None:
            self.cache_dir_content = {}

        self.fns1 = self.fns
        fns2 = []
        for each_jpg in self.fns1:
            frame_dir = os.path.dirname(each_jpg)
            if frame_dir not in self.cache_dir_content:
                self.cache_dir_content[frame_dir] = len(os.listdir(frame_dir))
            jpg_len = self.cache_dir_content[frame_dir]
            new_jpg = os.path.join(
                    frame_dir, 
                    '%06i.jpg' % (np.random.randint(jpg_len)+1))
            fns2.append(new_jpg)
        self.fns2 = np.asarray(fns2)


@DATASOURCES.register_module
class SAYCamCont(ImageList):
    def __init__(
            self, root, list_file, num_frames_meta_file,
            one_epoch_img_num=1281167,
            **kwargs):
        super().__init__(
            '', list_file, 
            **kwargs)
        self.video_root = root
        self.epoch_meta = copy.deepcopy(self.fns)
        self.num_frames_meta_file = num_frames_meta_file
        self.one_epoch_img_num = one_epoch_img_num
        self.load_num_frames_meta()
        self.set_epoch(0)

    def load_num_frames_meta(self):
        self.num_frames_meta = {}
        with open(self.num_frames_meta_file, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.num_frames_meta[video_name] = num_frames

    def get_base_fns(self, epoch):
        base_fns = []
        video_dirs = self.epoch_meta[epoch].split(',')
        for _dir in video_dirs:
            frames = [
                    os.path.join(self.video_root, _dir, '%06i.jpg' % (_frame+1))
                    for _frame in range(self.num_frames_meta[_dir])]
            base_fns.extend(frames)
        return base_fns

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        base_fns = self.get_base_fns(epoch)
        self.fns = np.random.choice(
                base_fns, self.one_epoch_img_num, replace=True)


@DATASOURCES.register_module
class SAYCamContAccu(SAYCamCont):
    def __init__(self, **kwargs):
        self.all_fns = {}
        super().__init__(**kwargs)

    def get_base_fns(self, epoch):
        base_fns = []
        for _tmp_epoch in range(epoch+1):
            if _tmp_epoch not in self.all_fns:
                self.all_fns[_tmp_epoch] = SAYCamCont.get_base_fns(
                        self, _tmp_epoch)
            base_fns.append(self.all_fns[_tmp_epoch])
        base_fns = np.concatenate(base_fns)
        return base_fns


@DATASOURCES.register_module
class SAYCamCndCont(SAYCamContAccu):
    def __init__(
            self, pipeline, batch_size=128,
            cond_method='max',
            *args, **kwargs):
        self.cnd_loader = None
        self.cont_loader = None
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.cond_method = cond_method
        super().__init__(*args, **kwargs)

    def load_num_frames_meta(self):
        self.num_frames_meta = {}
        with open(self.num_frames_meta_file, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.num_frames_meta[video_name] = num_frames

    def build_loaders(self, cont_data_source):
        from ..extraction import ExtractDatasetWidx
        from openselfsup.datasets.loader.sampler import DistributedSampler
        cnd_dataset = ExtractDatasetWidx(self, self.pipeline)
        cont_dataset = ExtractDatasetWidx(cont_data_source, self.pipeline)

        rank, world_size = get_dist_info()
        cnd_sampler = DistributedSampler(
                cnd_dataset, world_size, rank, 
                shuffle=False)
        self.cnd_loader = torch.utils.data.DataLoader(
                cnd_dataset,
                batch_size=self.batch_size,
                num_workers=10,
                sampler=cnd_sampler,
                )
        cont_sampler = DistributedSampler(
                cont_dataset, world_size, rank, 
                shuffle=False)
        self.cont_loader = torch.utils.data.DataLoader(
                cont_dataset,
                batch_size=self.batch_size,
                num_workers=10,
                sampler=cont_sampler,
                )

        assert len(cnd_dataset) % len(cont_dataset) == 0
        self.scale_ratio = len(cnd_dataset) // len(cont_dataset)

    def cross_device_gather(self, arr, data_len):
        _, world_size = get_dist_info()
        if world_size > 1:
            arr = np.concatenate(arr, axis=0)
            arr = gather_tensors_batch(arr, part_size=20)
        arr = np.concatenate(arr, axis=0)
        assert len(arr) >= data_len
        arr = arr[:data_len]
        return arr

    def get_storage_embds(self, model):
        rank, world_size = get_dist_info()
        if rank == 0:
            to_enum = tqdm(self.cnd_loader, desc='Get Storage Embds')
        else:
            to_enum = self.cnd_loader
        all_embds = []
        all_idxs = []
        model.eval()
        with torch.no_grad():
            for _idx, storage_batch in enumerate(to_enum):
                all_idxs.append(storage_batch['idx'].cpu().numpy())
                img = storage_batch['img'].cuda()
                all_embds.append(
                        model(
                            img=img, mode='test'
                            )['embd'].detach().cpu().numpy())
                del img
        model.train()
        data_len = len(self.cnd_loader.dataset)
        all_idxs = self.cross_device_gather(all_idxs, data_len)
        all_embds = self.cross_device_gather(all_embds, data_len)
        assert np.allclose(all_idxs, np.arange(data_len))
        return all_embds

    def get_cond_idx(self, _storage_sim):
        if self.cond_method == 'max':
            assert self.scale_ratio == 1
            _fn_idx = torch.argmax(_storage_sim, dim=-1)
        elif self.cond_method == 'min':
            assert self.scale_ratio == 1
            _fn_idx = torch.argmax(-_storage_sim, dim=-1)
        elif self.cond_method.startswith('max_'):
            _, _fn_idx = torch.topk(
                    _storage_sim, k=int(self.cond_method[4:]), 
                    largest=True, sorted=True,
                    dim=-1)
            if self.scale_ratio == 1:
                _fn_idx = _fn_idx[:, -1]
            else:
                _fn_idx = _fn_idx[:, -self.scale_ratio:]
                _fn_idx = _fn_idx.reshape(-1)
        elif self.cond_method.startswith('sum_max_'):
            _, _fn_idx = torch.topk(
                    torch.sum(_storage_sim, dim=0), 
                    k=int(self.cond_method[8:]), 
                    largest=True, sorted=True,
                    dim=-1)
            _fn_idx = _fn_idx[-_storage_sim.size(0)*self.scale_ratio:]
        else:
            raise NotImplementedError
        return _fn_idx

    def get_new_fns(self, model, storage_embds):
        storage_embds = torch.from_numpy(storage_embds).cuda()
        storage_embds = storage_embds.permute(1, 0)
        rank, world_size = get_dist_info()
        if rank == 0:
            to_enum = tqdm(self.cont_loader, desc='Get New Fns')
        else:
            to_enum = self.cont_loader
        fn_idxs = []
        all_idxs = []
        model.eval()
        with torch.no_grad():
            for _idx, cont_batch in enumerate(to_enum):
                all_idxs.append(cont_batch['idx'].cpu().numpy())
                img = cont_batch['img'].cuda()
                _embd = model(img=img, mode='test')['embd'].detach().cuda()
                _storage_sim = torch.matmul(_embd, storage_embds)
                _fn_idx = self.get_cond_idx(_storage_sim)
                fn_idxs.append(_fn_idx.detach().cpu().numpy())
                del img
        model.train()
        data_len = len(self.cont_loader.dataset)
        all_idxs = self.cross_device_gather(all_idxs, data_len)
        fn_idxs = self.cross_device_gather(fn_idxs, len(self.fns))
        assert np.allclose(all_idxs, np.arange(data_len))
        fns = self.fns[fn_idxs.astype(int)]
        return fns

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.set_epoch(epoch)

        if self.cont_loader is None:
            self.build_loaders(cont_data_source)

        storage_embds = self.get_storage_embds(model)
        self.fns = self.get_new_fns(model, storage_embds)
