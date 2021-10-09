"""KNNBuffer class."""

import torch
import torch.nn as nn

from openselfsup.models.registry import NECKS

@NECKS.register_module
class KNNBuffer_AHA(nn.Module):
    """
    A simple buffer implementation with K-nearest neighbour lookup.
    """

    def __init__(self, input_shape, target_shape, shift_range):
        super().__init__()

        self.shift_range = shift_range

        self.reset()

    def set_buffer_mode(self, mode='override'):
        self.buffer_mode = mode

    def reset(self):
        self.buffer = None
        self.buffer_batch = None
        self.buffer_mode = 'override'

    def shift_inputs(self, tensor):
        """From sparse r[0,1] to b[-1,1]"""
        tensor = (tensor > 0).float()
        tensor[tensor == 0] = -1
        return tensor

    def forward(self, inputs, mode="study"):
        """
        During training, store the training pattern inputs into the buffer.
        Not applicable:
            Depending on the buffer_mode, the buffer can be either overriden every step
            or continously appended.
        At test time, use the test pattern inputs to lookup the matching patterns
        from the buffer using K-nearest neighbour with K=1.
        """
        if self.training:
            """
            During training, 
            """
          # Range shift from unit to signed unit
            if self.shift_range == True:
                inputs = self.shift_inputs(inputs)

            self.buffer_batch = inputs

            # Memorise inputs in buffer
            if self.buffer is None:
                self.buffer = inputs.unsqueeze(1)
            else:
                self.buffer = torch.cat((self.buffer, inputs.unsqueeze(1)), dim=1)
                # self.buffer.append(inputs)

            return self.buffer_batch
            
        else: # in recall mode
            recalled = torch.zeros_like(inputs)

            # print("recalled", recalled.shape)
            for i, test_input in enumerate(inputs):
                
                embds_bank = self.buffer[i]
                # print("test_input", test_input.shape, test_input)
                # print("embds_bank", embds_bank.shape)
                diff = embds_bank - test_input
                # print("diff", diff.shape, diff)
                # print("diff list", [(em - test_input)[:10] for em in embds_bank])
                dist = torch.argmax(torch.norm(diff, dim=1, p=None))
                # knn = dist.topk(k=1, largest=False)
                # print("i dist",i, dist)
                recalled[i] = embds_bank[dist]

            return recalled