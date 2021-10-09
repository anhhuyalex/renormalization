from mmcv import Config
import argparse
from openselfsup.models import build_model
from openselfsup.datasets import build_dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import pdb
import torch
import pickle
from tqdm import tqdm


def get_transforms(resize_size=64):
    from torchvision import transforms
    norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg),
            ])
    return transform


def build_load_model(cfg, ckpt_path):
    model = build_model(cfg.model)
    if ckpt_path is not None:
        model_dict = torch.load(ckpt_path)
        model.load_state_dict(model_dict['state_dict'])
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class ResponseExtractor:
    def __init__(
            self, 
            cfg_path, cfg_func, ckpt_path,
            layers):
        self.cfg_path = cfg_path
        self.cfg_func = cfg_func
        self.ckpt_path = ckpt_path
        self.layers = layers

        self.get_cfg()
        self.get_model()

    def get_cfg(self):
        self.cfg = Config.fromfile(self.cfg_path)
        self.cfg = self.cfg_func(self.cfg)

    def get_model(self):
        self.model = build_load_model(
                self.cfg, self.ckpt_path)

    def get_layer(self, layer_name):
        module = self.model
        for part in layer_name.split('.'):
            module = module._modules.get(part)
            assert module is not None, \
                    f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def register_hooks(self):
        self.target_dict = {}
        self.hooks = []
        for layer_name in self.layers:
            layer = self.get_layer(layer_name)
            self.hooks.append(
                    self.register_one_hook(
                        layer, layer_name, self.target_dict))

    def register_one_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = output.cpu().data.numpy()

        hook = layer.register_forward_hook(hook_function)
        return hook

    def get_activations(self, images):
        import torch
        from torch.autograd import Variable
        #images = [torch.from_numpy(image) for image in images]
        images = Variable(torch.stack(images))
        images = images.cuda()
        self.model.eval()
        self.register_hooks()
        self.model(images, mode='test')
        for hook in self.hooks:
            hook.remove()
        return self.target_dict

    def get_embds(self, images):
        import torch
        from torch.autograd import Variable
        images = Variable(torch.stack(images))
        images = images.cuda()
        self.model.eval()
        embds = self.model(images, mode='test')
        return embds['embd'].detach().numpy()


def test():
    img_path = '/mnt/fs4/chengxuz/rodent_dev/HMAX/gratings/0100_grating_0_0.02_2/grating_0_frame_001.bmp'
    img = Image.open(img_path).convert('RGB')

    transform = get_transforms()
    inpt_img = transform(img)

    from openselfsup.analysis.local_paths import MODEL_KWARGS
    layers = [
            'backbone.features.2',
            'backbone.features.6',
            'backbone.features.10',
            'backbone.features.13',
            'backbone.features.16']
    response_extractor = ResponseExtractor(
            layers=layers,
            **MODEL_KWARGS['simclr_alxnt_ctl64'])
    act_dict = response_extractor.get_activations([inpt_img])
    pdb.set_trace()
    pass


if __name__ == '__main__':
    test()
