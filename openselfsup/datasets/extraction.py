from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from torch.utils.data import Dataset

from openselfsup.utils import print_log, build_from_cfg

from torchvision.transforms import Compose



@DATASETS.register_module
class ExtractDataset(BaseDataset):
    """Dataset for feature extraction.
    """

    def __init__(self, data_source, pipeline):
        super(ExtractDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        return dict(img=img)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented


@DATASETS.register_module
class ExtractDatasetWidx(Dataset):
    """Dataset for feature extraction.
    """
    def __init__(self, data_source, pipeline):
        self.data_source = data_source
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        return dict(img=img, idx=idx)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
