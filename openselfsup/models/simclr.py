import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import GatherLayer


@MODELS.register_module
class SimCLR(nn.Module):
    """SimCLR.

    Implementation of "A Simple Framework for Contrastive Learning
    of Visual Representations (https://arxiv.org/abs/2002.05709)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(
            self, backbone, neck=None, head=None, hipp_head=None,
            pretrained=None):
        super(SimCLR, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.hipp_head = None
        if hipp_head is not None:
            self.hipp_head = builder.build_head(hipp_head)
        self.init_weights(pretrained=pretrained)

    @staticmethod
    def _create_buffer(N, mode='normal'):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        if mode == 'normal':
            neg_mask[pos_ind] = 0
        elif mode == 'exposure_cotrain':
            neg_mask[: N, : N-1] = 0    # make sure face pairs are never treated as negatives
            neg_mask[pos_ind] = 0
            
        return mask, pos_ind, neg_mask

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')
        if self.hipp_head is not None:
            self.hipp_head.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def l2_normalize(self, z):
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        return z

    def get_embeddings(self, img):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        num_views = img.size(1)
        img = img.reshape(
            img.size(0) * num_views, img.size(2), img.size(3), img.size(4))
        x = self.forward_backbone(img)  # 2n
        z = self.neck(x)[0]  # (2n)xd
        z = self.l2_normalize(z)
        if num_views > 2:
            z = z.reshape(z.size(0) // num_views, num_views, z.size(1))
            z_0 = z[:, 0, :]
            z_others = z[:, 1:, :]
            z_others = torch.mean(z_others, dim=1)
            z_others = self.l2_normalize(z_others)
            z = torch.stack([z_0, z_others], dim=1)
            z = z.reshape(z.size(0)*2, z.size(2))
        return z

    def forward_train(
            self, img, mode='normal', within_batch_ctr=None, 
            add_noise=None,
            **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            mode (str): normal is used in pretraining for large dataset, exposure_cotrain 
                is used in exposure co-training to avoid same faces are treated as 
                negative pairs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        z = self.get_embeddings(img)
        if self.hipp_head is not None:
            hipp_loss = self.hipp_head(z)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        if add_noise is not None:
            noise = torch.randn(permu_z.shape[0], permu_z.shape[1]).cuda() * add_noise
            permu_z += noise
            permu_z = nn.functional.normalize(permu_z, dim=0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, mode=mode)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        if within_batch_ctr is not None:
            _t = self.head.temperature
            l_batch_ctr = torch.exp(negative / _t)
            l_batch_ctr = torch.sum(l_batch_ctr, dim=1)
            l_batch_ctr = torch.mean(torch.log(1.0 / l_batch_ctr))
            losses['loss'] -= within_batch_ctr * l_batch_ctr
        if self.hipp_head is not None:
            losses['loss'] += hipp_loss.pop('loss')
            for key in hipp_loss:
                assert key not in losses
            losses.update(hipp_loss)
        return losses

    def forward_train_asy(
            self, img, mode='normal', 
            **kwargs):
        z = self.get_embeddings(img)
        z = z.reshape(z.size(0)//2, 2, z.size(1))
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        z_0 = z[:, 0, :]
        z_1 = z[:, 1, :]
        positive = torch.sum(z_0 * z_1, dim=1, keepdim=True)
        negative = torch.matmul(z_1, z_1.permute(1, 0))
        nega_mask = 1 - torch.eye(z_1.size(0), dtype=torch.uint8).cuda()
        negative = torch.masked_select(
                negative, nega_mask==1).reshape(negative.size(0), -1)
        losses = self.head(positive, negative)
        return losses

    def forward_test(self, img, **kwargs):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        x = self.forward_backbone(img)
        z = self.neck(x)[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True))
        return {'embd': z.cpu()}
        
    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, mode='normal', **kwargs)
        elif mode == 'exposure_cotrain':
            return self.forward_train(img, mode='exposure_cotrain', **kwargs)
        elif mode == 'asy_train':
            return self.forward_train_asy(img, mode='normal', **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
