"""PerforantHebb class."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .local_connection import LocalConnection
from .local_optimizer import LocalOptim

from openselfsup.models.registry import NECKS

@NECKS.register_module
class PerforantHebb_AHA(nn.Module):
    """
    A non-trainable module based on Dentate Gyrus (DG), produces sparse outputs and inhibits neurons after firing.
    """

    def __init__(self, ec_shape, dg_shape, ca3_shape, 
                 learning_rate,
                reset_params = True, reset_optim = True,
                 use_dg_ca3 = False, 
                ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.reset_params = reset_params 
        self.reset_optim = reset_optim
        self.use_dg_ca3 = use_dg_ca3
        

        # ec_size = np.prod(ec_shape[1:])
        # dg_size = np.prod(dg_shape[1:])
        # ca3_size = np.prod(ca3_shape[1:])

        self.dg_ca3 = LocalConnection(dg_shape, ca3_shape, bias=False)
        self.dg_ca3_optimizer = LocalOptim(self.dg_ca3.named_parameters(), lr=self.learning_rate)

        self.ec_ca3 = LocalConnection(ec_shape, ca3_shape, bias=False)
        self.ec_ca3_optimizer = LocalOptim(self.ec_ca3.named_parameters(), lr=self.learning_rate)
    
    def compute_dw(self, inputs, targets, weights):
        """
        Adapted from https://github.com/Cerenaut/pt-aha/blob/main/cls_module/cls_module/components/learning_rules.py#L19
        """
        inputs_tiled = inputs.unsqueeze(-1)
        inputs_tiled = inputs_tiled.repeat((1, 1, targets.size(1))).transpose(2, 1)

        targets_tiled = targets.unsqueeze(-1)
        targets_tiled = targets_tiled.repeat((1, 1, inputs.size(1)))

        # Expand dimension of weights and tile to match corresponding samples in
        # the given batch of inputs/targets
        weights_tiled = weights.unsqueeze(0)
        weights_tiled = weights_tiled.repeat((inputs.size(0), 1, 1))

        # This rule from Leabra is a variation on Oja's rule
        d_ws = targets_tiled * (inputs_tiled - weights_tiled)

        return torch.mean(d_ws, dim=0)
    
    def reset(self):
        if self.reset_params:
            self.dg_ca3.reset_parameters()
            self.ec_ca3.reset_parameters()

        # Reset the module optimizer
        if self.reset_optim:
            self.dg_ca3_optimizer.state = defaultdict(dict)
            self.ec_ca3_optimizer.state = defaultdict(dict)

    def forward(self, ec_inputs, dg_inputs, mode = "study"):
        with torch.no_grad():
            dg_ca3_in = dg_inputs

            if self.use_dg_ca3:
                pre_dg_ca3_out = self.dg_ca3(dg_ca3_in)
            else:
                pre_dg_ca3_out = dg_inputs

            ec_ca3_in = torch.flatten(ec_inputs, 1)
            pre_ec_ca3_out = self.ec_ca3(ec_ca3_in)

            pre_pc_cue = pre_dg_ca3_out + pre_ec_ca3_out
            pc_cue = pre_pc_cue

            if self.training:
                # Update DG:CA3 with respect to dg_ca3_in (i.e. outputs['ps'])
                if self.use_dg_ca3:
                    d_dg_ca3 = self.compute_dw(dg_ca3_in, pre_pc_cue, self.dg_ca3.weight)
                    d_dg_ca3 = d_dg_ca3.view(*self.dg_ca3.weight.size())
                    self.dg_ca3_optimizer.local_step(d_dg_ca3)

                # Update EC:CA3 with respect to ec_ca3_in (i.e. inputs)
                d_ec_ca3 = self.compute_dw(ec_ca3_in, pre_pc_cue, self.ec_ca3.weight)
                d_ec_ca3 = d_ec_ca3.view(*self.ec_ca3.weight.size())
                self.ec_ca3_optimizer.local_step(d_ec_ca3)

            # Compute the post synaptic activity for loss calculation
            if self.use_dg_ca3:
                post_dg_ca3_out = self.dg_ca3(dg_ca3_in)
            else:
                post_dg_ca3_out = dg_inputs

            post_ec_ca3_out = self.ec_ca3(ec_ca3_in)
            post_pc_cue = post_dg_ca3_out + post_ec_ca3_out

            dg_ca3_loss = F.mse_loss(pre_dg_ca3_out, post_dg_ca3_out)
            ec_ca3_loss = F.mse_loss(pre_ec_ca3_out, post_ec_ca3_out)
            pc_cue_loss = F.mse_loss(pre_pc_cue, post_pc_cue)

        return pc_cue, dg_ca3_loss, ec_ca3_loss, pc_cue_loss