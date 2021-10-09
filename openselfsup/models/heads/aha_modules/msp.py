"""MonosynapticPathway class."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from cerenaut_pt_core.utils import build_topk_mask
from .simple_autoencoder import SimpleAutoencoder

from openselfsup.models.registry import NECKS

@NECKS.register_module
class MonosynapticPathway_AHA(nn.Module):
    """
    An error-driven monosynaptic pathway implementation.
    """

    def __init__(self, ca3_shape, ec_shape, ca1_cfg, ca3_ca1_cfg,
                       ca1_reset_params = False,
                        ca1_reset_optim = False,
                        ca3_ca1_reset_params = True,
                        ca3_ca1_reset_optim = True,
                        ca3_recall = True):
        super().__init__()

        self.ca1_reset_params = ca1_reset_params
        self.ca1_reset_optim = ca1_reset_optim
        self.ca3_ca1_reset_params = ca3_ca1_reset_params
        self.ca3_ca1_reset_optim = ca3_ca1_reset_optim
        self.ca3_recall = ca3_recall
        
        # Build the CA1 sub-module, to reproduce the EC inputs
        self.ca1 = SimpleAutoencoder(input_size=ec_shape, output_size=ec_shape, **ca1_cfg)
        self.ca1_optimizer = optim.Adam(self.ca1.parameters(),
                                        lr=ca1_cfg['learning_rate'],
                                        weight_decay=ca1_cfg['weight_decay'])


        # Build the CA1 sub-module, to reproduce the EC inputs
        self.ca3_ca1 = SimpleAutoencoder(input_size=ca3_shape, output_size=ca1_cfg['hidden_size'], **ca3_ca1_cfg)
        self.ca3_ca1_optimizer = optim.Adam(self.ca3_ca1.parameters(),
                                            lr=ca3_ca1_cfg['learning_rate'],
                                            weight_decay=ca3_ca1_cfg['weight_decay'])

    def reset(self):
        if self.ca1_reset_params:
            self.ca1.reset_parameters()

        if self.ca3_ca1_reset_params:
            self.ca3_ca1.reset_parameters()

        # Reset the module optimizer
        if self.ca1_reset_optim:
            self.ca1_optimizer.state = defaultdict(dict)

        if self.ca3_ca1_reset_optim:
            self.ca3_ca1_optimizer.state = defaultdict(dict)

    def forward_ca1(self, inputs, targets):
        if self.training:
            self.ca1_optimizer.zero_grad()

        encoding, decoding = self.ca1(inputs)

        loss = F.mse_loss(decoding, targets)

        outputs = {
            'encoding': encoding.detach(),
            'decoding': decoding.detach(),

            'output': encoding.detach()  # Designated output for linked modules
        }

        if self.training:
            loss.backward()
            self.ca1_optimizer.step()
            self.ca1_optimizer.zero_grad()
            
        return loss, outputs

    def forward_ca3_ca1(self, inputs, targets):
        if self.training:
            self.ca3_ca1_optimizer.zero_grad()

        encoding, decoding = self.ca3_ca1(inputs)

        loss = F.mse_loss(decoding, targets)

        outputs = {
            'encoding': encoding.detach(),
            'decoding': decoding.detach(),

            'output': encoding.detach()  # Designated output for linked modules
        }

        if self.training:
            loss.backward()
            self.ca3_ca1_optimizer.step()
            self.ca3_ca1_optimizer.zero_grad()

        return loss, outputs

    def forward(self, ec_inputs, ca3_inputs, mode="study"):
        # During study, EC will drive the CA1
        ca1_loss, ca1_outputs = self.forward_ca1(inputs=ec_inputs, targets=ec_inputs)
        ca3_ca1_target = ca1_outputs['encoding']

        ca3_ca1_loss, ca3_ca1_outputs = self.forward_ca3_ca1(inputs=ca3_inputs, targets=ca3_ca1_target)

        # During recall, the CA3:CA1 will drive CA1 reconstruction
        if not self.training and self.ca3_recall:
            ca1_hidden_recon = ca3_ca1_outputs['decoding']
            ca1_decoding = self.ca1.decode(ca1_hidden_recon)
            ca1_loss = F.mse_loss(ca1_decoding, ec_inputs)

            ca1_outputs = {
            'encoding': None,
            'decoding': ca1_decoding,
            'output': None
            }

        return ca1_loss, ca1_outputs, ca3_ca1_loss, ca3_ca1_outputs