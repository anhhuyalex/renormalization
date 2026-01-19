# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python (fmri)
#     language: python
#     name: fmri
# ---

# +
# We want to maximize the dot product
# Step 1: Add the position encoding (content + position)
# INFONCE: A and B are complete unconstrained, instead of optimizing over pi
# Check: Does the model localize the right memory

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import copy

class DAM(nn.Module):
    def __init__(self, N, H, M, eta=1.0, init_std=1e-2):
        """
        Args:
            N: Sequence length (bits).
            H: Hidden dimension (number of heads).
            M: Memory capacity (number of sequences).
            eta: Inverse temperature parameter.
        """
        super().__init__()
        self.N = N
        self.H = H
        self.M = M
        self.eta = eta
        self.init_std = init_std

        # A(n) parameters: (N, H, N). We will mask unused parts.
        # We need independent parameters for each length n.
        # For length n (predicting n-th bit, 0-indexed), we use inputs 0..n-1.
        # So we have separate weights for each n.
        self.A_logits = nn.Parameter(torch.randn(N, H, N) * self.init_std) # small positive weights

        # B parameters: (H, N).
        self.B_logits = nn.Parameter(torch.randn(H, N) * self.init_std) # small positive weights

        # Memory state
        # We store binary sequences
        # Register as a buffer so `model.to(device)` moves it with the module.
        # Note: we still reassign/resize it in `update_memory`, which is fine for buffers.
        self.register_buffer("memory", torch.zeros(0, N))
        self.is_memory_empty = True

    def update_memory(self, sequences):
        """
        Updates the DAM with new sequences.
        The memory is a queue so we only keep the last M sequences.
        Args:
            sequences: (Batch, N) tensor of binary sequences (-1, +1).
        """
        new_memory = torch.cat([self.memory, sequences.detach().to(self.memory.device)], dim=0)

        if new_memory.shape[0] > self.M:
            new_memory = new_memory[-self.M:]

        self.memory = new_memory
        self.is_memory_empty = False

    def clear_memory(self):
        """
        Clears the memory.
        """
        self.memory = torch.zeros(0, self.N, device=self.memory.device)
        self.is_memory_empty = True

    def get_A(self, n):
        """
        Returns normalized A(n) matrix of shape (H, n).
        """
        if n == 0:
            return torch.zeros(self.H, 0, device=self.A_logits.device)

        # Select params for predictng n+1 th bit (index n).
        # We use slice 0:n because we attend to first n bits.
        logits = self.A_logits[n, :, :n] # (H, n)
        return F.softmax(logits, dim=-1)

    def get_B(self):
        """
        Returns normalized B matrix of shape (H, N).
        """
        return F.softmax(self.B_logits, dim=-1)

    def forward_step(self, zeta, n, phi_mu):
        """
        Predict probability of (n+1)-th bit (index n) being +1.

        Args:
            zeta: (Batch, N) full sequences (we only peek up to n).
            n: int, current length (0 to N-1). We verify up to zeta[:, :n].
               We want to predict zeta[:, n].
            phi_mu: (M, H) Precomputed memory keys.

        Returns:
            probs: (Batch,) probability that next bit is +1.
        """
        B_val = zeta.shape[0] # batch size

        # handle empty memory
        if self.is_memory_empty: # shape: (M, N) where M = memory capacity, N = sequence length
            return torch.full((B_val,), 0.5, device=zeta.device)

        A_n = self.get_A(n) # shape: (H, n)
        # context is the input sequence up to position n (exclusive)
        context = zeta[:, :n] # shape: (Batch, n)
        # hat_phi = sum A_i * zeta_i.
        hat_phi = torch.einsum('bi,hi->bh', context, A_n)
        # if n == 5 and np.random.rand() < 0.1:
        #     print("A_n", A_n[0].flatten(), A_n.shape, self.A_logits[n, :, :n])
        #     print("context: ", context[0,:10], "hat_phi: ", hat_phi[0,:10])
            # print("B_mat", B_mat, B_mat.shape)
        # 3. Retrieval probabilities
        # score: (Batch, M)
        # score = eta * hat_phi^T phi_mu
        score = self.eta * torch.einsum('bh,mh->bm', hat_phi, phi_mu)
        pi = F.softmax(score, dim=1) # (Batch, M)
        
        # 4. Predict next bit
        # We want prob that (n)-th bit is +1.
        # memory_bits: (M,)
        memory_bits_at_n = self.memory[:, n] # +1 or -1

        # We want sum of pi where memory bit is +1.
        # Indicator: (memory_bits_at_n == 1).float()
        plus_one_mask = (memory_bits_at_n > 0).float()
        # prob = sum(pi * mask)
        prob_plus_one = torch.sum(pi * plus_one_mask.unsqueeze(0), dim=1)

        # print("prob_plus_one: ", prob_plus_one)
        return prob_plus_one

    def train_batch(self, sequences, optimizer):
        """
        Computes the BCE loss averaged over all positions.
        Args:
             sequences: (Batch, N)
        Returns:
             loss: scalar
        """
        total_loss = 0.0
        total_accuracy = 0.0
        
        # If memory is empty, we can't really predict based on history, 
        # but we returning 0.5 loss is appropriate and no grad update.
        if self.is_memory_empty:
            # Just return dummy values, no update
            bce = -torch.log(torch.tensor(0.5))
            return bce, torch.tensor(0.5)

        if self.training:
            optimizer.zero_grad() 
        loss_batch = 0.0
        
        # Precompute phi_mu for the batch (Constant for this batch update)
        # phi_mu = B (H, N) x memory^T (N, M) -> (H, M) -> transpose to (M, H)
        # einsum: hn, mn -> mh
        phi_mu = torch.einsum('hn,mn->mh', self.get_B(), self.memory)
        
        for n in range(1, self.N):
            # Predict n-th bit (0-indexed) using 0..n-1 history
            prob_plus_one = self.forward_step(sequences, n, phi_mu) # shape: (Batch,)
            # Target
            target = sequences[:, n] # -1 or +1, shape: (Batch,)
            # Convert target to 0/1 for BCE
            target_01 = (target > 0).float()

            # numeric stability
            prob_plus_one = torch.clamp(prob_plus_one, 1e-6, 1.0 - 1e-6)

            loss_n = F.binary_cross_entropy(prob_plus_one, target_01)
            loss_batch = loss_batch + loss_n

            # accuracy
            accuracy_n = ((prob_plus_one > 0.5) == (target_01 > 0.5))
            total_accuracy += accuracy_n.float().mean()
            
        # Normalize sum of losses by number of predictions (N-1)
        # Note: problem says 1/N sum_{n=0}^{N-1}, but code loop is range(1, N) -> n=1..N-1.
        # This misses n=0 prediction. But n=0 has 0 context.
        # forward_step(n=0) uses context zeta[:, :0] (empty).
        # We can include n=0 if we want, but let's stick to existing range but fix normalization.
        
        loss_final = loss_batch / (self.N - 1)
        if self.training:
            loss_final.backward()
            optimizer.step() 
         
        total_loss = loss_final.item()
        avg_accuracy = total_accuracy / (self.N - 1)

        return total_loss, avg_accuracy

def initialize_record(args):
    record = {
        "args": args,
        "logs": [],
    }
    return record


# +

if __name__ == "__main__":
    print("Running DAM...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=1e1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--INIT_STD", type=float, default=1e-1)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--M", type=int, default=3000)
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--BATCH_SIZE", type=int, default=50)
    parser.add_argument("--NUM_STEPS", type=int, default=50000)
    parser.add_argument("--NUM_ITERS_PER_LOG", type=int, default=100)
    parser.add_argument("--savedir", type=str, default="/scratch/qanguyen/gautam")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to cuda if available.",
    )
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    print(f"args: {args}")

    # Create savedir if it doesn't exist
    os.makedirs(f"{args.savedir}/{args.prefix}", exist_ok=True) 
    exp_name = f"{args.prefix}/dam_{time.time()}"
    model = DAM(args.N, args.H, args.M, eta=args.eta, init_std=args.INIT_STD)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    record = initialize_record(args)

    # Generate random sequences
    sequences = torch.sign(torch.randn(args.K, args.N, device=device))

    # Initialize memory
    # We start with empty memory as per standard online learning or just-in-time memorization.
    # Initializing with zeros is harmful because dot products with -1/+1 sequences are low,
    # effectively acting as noise or bias if not masked correctly.
    # init_mem = torch.zeros(args.M, args.N)
    # model.update_memory(init_mem)
    # print(f"Memory initialized with shape: {model.memory.shape}")
    print("Memory initialized as empty.")

    # Training Loop Simulation
    model.train()
    for step in range(args.NUM_STEPS):

        # Sample batch from sequences
        indices = torch.randint(0, args.K, (args.BATCH_SIZE,), device=device)
        batch = sequences[indices]

        loss, accuracy = model.train_batch(batch, optimizer)

        # Update memory
        model.update_memory(batch)
        # Compute stats
        with torch.no_grad():
            A_grad_norm = model.A_logits.grad.norm().item() if model.A_logits.grad is not None else 0.0
            B_grad_norm = model.B_logits.grad.norm().item() if model.B_logits.grad is not None else 0.0
            # A_entropy (approximate, averaged over N)
            # Just take mean max attention as proxy for "sharpness"
            A_sharpness = 0.0
            for n in range(1, args.N):
                A_n = model.get_A(n) # (H, n)
                A_sharpness += A_n.max(dim=1).values.mean().item()
            A_sharpness /= (args.N - 1)
            
            logs = {
                "step": step,
                "loss": loss,
                "accuracy": accuracy,
                "A_grad_norm": A_grad_norm,
                "B_grad_norm": B_grad_norm,
                "A_sharpness": A_sharpness,
                "phase": "train"
            }
            record["logs"].append(logs)

        if step % args.NUM_ITERS_PER_LOG == 0:
            print(f"Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
            
            print(f"Stats: A_grad={A_grad_norm:.4f}, B_grad={B_grad_norm:.4f}, A_sharp={A_sharpness:.4f}")

    print("Verification complete.")
    # save model.state_dict() 
    record["model_state_dict"] = copy.deepcopy(model.state_dict())

    # resample sequences 
    sequences = torch.sign(torch.randn(args.K, args.N, device=device))
    print(f"Sequences resampled with shape: {sequences.shape}")
    model.eval()
    model.clear_memory() # clear memory for evaluation

    for step in range(args.NUM_STEPS):

        # Sample batch from sequences
        indices = torch.randint(0, args.K, (args.BATCH_SIZE,), device=device)
        batch = sequences[indices]

        loss, accuracy = model.train_batch(batch, optimizer)

        # Update memory
        model.update_memory(batch)

        logs = {
                "step": step,
                "loss": loss,
                "accuracy": accuracy, 
                "phase": "eval"
            }
        record["logs"].append(logs)
    # save model.state_dict() as A_logits and B_logits
    torch.save(record, f"{args.savedir}/{exp_name}.pt")