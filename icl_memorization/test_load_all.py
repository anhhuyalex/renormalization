import os, sys, glob, pickle, math
from collections import defaultdict
import numpy as np
import pandas as pd

class FastParamUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_count = 0
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: None
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            def dummy_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
                numel = math.prod(size)
                self.param_count += numel
                return numel
            return dummy_tensor
        return super().find_class(module, name)

print("FastParamUnpickler ready")
