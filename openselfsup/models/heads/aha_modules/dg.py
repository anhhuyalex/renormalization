"""DentateGyrus class."""

import torch
import torch.nn as nn

import numpy as np

from openselfsup.models.registry import NECKS

@NECKS.register_module
class AHA_linearDG_kSparseInhibition(nn.Module):
    """
    A non-trainable module based on Dentate Gyrus (DG), produces sparse outputs and inhibits neurons after firing.
    """
    def __init__(self, in_channels, out_channels, sparsity, 
                     knockout_rate, init_scale, inhibition_decay,
                     use_stub = False,
                     add_bias = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparsity = sparsity
        self.knockout_rate = knockout_rate
        self.init_scale = init_scale
        self.inhibition_decay = inhibition_decay
        self.use_stub = use_stub
    

        self.layer = nn.Linear(self.in_channels, self.out_channels, bias=add_bias)
        self.layer.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        """Custom initialization *does* make a big difference to orthogonality, even with inhibition"""

        num_weights = self.out_channels * self.in_channels
        random_values = torch.rand(num_weights, dtype=torch.float32)

        knockout_rate = self.knockout_rate
        keep_rate = 1.0 - knockout_rate

        initial_mask = np.random.choice(np.array([0, 1], dtype=np.float32),
                                        size=(num_weights),
                                        p=[knockout_rate, keep_rate])

        initial_values = random_values * initial_mask * self.init_scale
        initial_values = torch.reshape(initial_values, shape=(self.out_channels, self.in_channels))

        abs_sum = torch.sum(torch.abs(initial_values), dim=1, keepdim=True)
        norm = 1.0 / abs_sum

        initial_values = initial_values * norm
        self.layer.weight.data = initial_values

    def build_topk_mask(self, x, dim=1, k=2):
      """
      Simple functional version of KWinnersMask/KWinners since
      autograd function apparently not currently exportable by JIT
      Sourced from Jeremy's RSM code
      """
      res = torch.zeros_like(x)
      _, indices = torch.topk(x, k=k, dim=dim, sorted=False)
      return res.scatter(dim, indices, 1)

    def apply_sparse_filter(self, encoding):
        """Sparse filtering with inhibition."""
        batch_size = encoding.shape[0]

        k = int(self.sparsity)
        inhibition_decay = self.inhibition_decay

        cells_shape = [self.out_channels]
        batch_cells_shape = [batch_size, self.out_channels]

        device = encoding.device

        inhibition = torch.zeros(cells_shape, device=device)
        filtered = torch.zeros(batch_cells_shape, device=device)

        # Inhibit over time within a batch (because we don't bother having repeats for this).
        for i in range(0, batch_size):
            # Create a mask with a 1 for this batch only
            this_batch_mask = torch.zeros([batch_size, 1], device=device)
            this_batch_mask[i][0] = 1.0

            refraction = 1.0 - inhibition
            refraction_2d = refraction.unsqueeze(0)  # add batch dim
            refracted = torch.abs(encoding) * refraction_2d

            # Find the "winners". The top k elements in each batch sample
            # ---------------------------------------------------------------------
            top_k_mask = self.build_topk_mask(refracted, dim=-1, k=k)

            # Retrospectively add batch-sparsity per cell: pick the top-k
            # ---------------------------------------------------------------------
            batch_filtered = encoding * top_k_mask  # apply mask 3 to output 2
            this_batch_filtered = batch_filtered * this_batch_mask
            this_batch_topk = top_k_mask * this_batch_mask
            fired, _ = torch.max(this_batch_topk, dim=0)  # reduce over batch

            inhibition = inhibition * inhibition_decay + fired  # set to 1

            filtered = filtered + this_batch_filtered

        return filtered, inhibition

    def stub(self, x):
        """Returns a batch of non overlapping n-hot samples in range [0, 1]."""
        batch_size = x.shape[0]
        n = self.sparsity

        assert ((batch_size * n - 1) + n) < self.out_channels, "Can't produce batch_size {0} non-overlapping samples, " \
               "reduce n {1} or increase sample_size {2}".format(batch_size, n, self.out_channels)

        batch = torch.zeros(batch_size, self.out_channels)

        # Return the sample at given index
        for idx in range(batch_size):
            start_idx = idx * n
            end_idx = start_idx + n
            batch[idx][start_idx:end_idx] = 1

        return batch

    def compute_overlap(self, encoding):
        """ a, b = (0,1)
        Overlap is where tensors have 1's in the same position.
        Return number of bits that overlap """

        num_samples = encoding.shape[0]
        batch_overlap = torch.zeros(num_samples).to(encoding.device)

        for i in range(num_samples):
            a = encoding[i]

            for j in range(num_samples):
                # Avoid matching the same samples
                if i == j:
                    continue

            b = encoding[j]
            overlap = torch.sum(a * b)
            batch_overlap[i] += overlap

        return batch_overlap

    def forward(self, inputs):  # pylint: disable=arguments-differ
        inputs = torch.flatten(inputs, start_dim=1)
        if self.use_stub:
            top_k_mask = self.stub(inputs)

        else:
            with torch.no_grad():
                encoding = self.layer(inputs)
                filtered_encoding, _ = self.apply_sparse_filter(encoding)

            # Override encoding to become binary mask
            top_k_mask = self.build_topk_mask(filtered_encoding, dim=-1, k=self.sparsity)
            
        # overlap = self.compute_overlap(top_k_mask)
        # overlap_sum = overlap.sum().item()

        # assert overlap_sum == 0.0, 'Found overlap between samples in batch'

        # return top_k_mask.detach()
        return filtered_encoding.detach()
