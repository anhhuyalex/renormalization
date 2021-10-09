import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class ContrastiveHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(
            self, temperature=0.1, pos_nn_num=None, 
            neg_fn_num=None, neg_th_value=None):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.pos_nn_num = pos_nn_num
        self.neg_fn_num = neg_fn_num
        self.neg_th_value = neg_th_value

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        if self.neg_fn_num is not None:
            neg, _ = torch.topk(-neg, self.neg_fn_num, dim=-1)
            neg = -neg
        if self.neg_th_value is not None:
            neg = neg.masked_fill(neg > self.neg_th_value, -float('inf'))
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        if self.pos_nn_num is None:
            losses['loss'] = self.criterion(logits, labels)
        else:
            logits = torch.exp(logits)
            pos_prob = torch.sum(logits[:, :(self.pos_nn_num+1)], dim=1)
            all_prob = torch.sum(logits, dim=1)
            losses['loss'] = -torch.mean(torch.log(pos_prob / all_prob))
        return losses
