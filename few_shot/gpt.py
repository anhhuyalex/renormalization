import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.sequence_block_size, config.sequence_block_size)).view(
                    1, 1, config.sequence_block_size, config.sequence_block_size
                ),
            )

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 64
    input_dim: int = 16
    output_dim: int = 1
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True

    @property
    def sequence_block_size(self):
        return self.block_size * 2


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim + config.output_dim, config.n_embd, bias=config.bias)
        self.transformer = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(config.sequence_block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.output_head = nn.Linear(config.n_embd, config.output_dim, bias=config.bias)
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def build_sequence(self, xs, ys):
        batch_size, seq_len, _ = xs.shape
        x_tokens = torch.cat([xs, torch.zeros(batch_size, seq_len, 1, device=xs.device, dtype=xs.dtype)], dim=-1)
        y_tokens = torch.cat([torch.zeros(batch_size, seq_len, xs.size(-1), device=xs.device, dtype=xs.dtype), ys], dim=-1)
        sequence = torch.stack([x_tokens, y_tokens], dim=2).reshape(batch_size, seq_len * 2, xs.size(-1) + 1)
        return sequence

    def forward(self, xs, ys):
        _, seq_len, _ = xs.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")
        sequence = self.build_sequence(xs, ys)
        positions = torch.arange(sequence.size(1), device=sequence.device, dtype=torch.long)
        x = self.input_projection(sequence) + self.transformer.wpe(positions)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x_positions = x[:, 0::2, :]
        return self.output_head(x_positions)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        decay_params = [parameter for parameter in param_dict.values() if parameter.dim() >= 2]
        nodecay_params = [parameter for parameter in param_dict.values() if parameter.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


class OneLayerAttention(nn.Module):
    def __init__(
        self,
        len_context,
        input_dim,
        output_dim,
        num_heads,
        num_hidden_features,
        num_mlp_layers,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.len_context = len_context
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.hidden_dim = num_heads * num_hidden_features
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(input_dim + output_dim, self.hidden_dim, bias=bias)
        self.position_embedding = nn.Embedding(len_context * 2, self.hidden_dim)
        self.pre_attn_ln = LayerNorm(self.hidden_dim, bias=bias)
        self.post_attn_ln = LayerNorm(self.hidden_dim, bias=bias)
        self.c_attn = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=bias)
        mlp_layers = [nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)]
        for _ in range(max(0, num_mlp_layers - 1)):
            mlp_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias))
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.output_head = nn.Linear(self.hidden_dim, output_dim, bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def build_sequence(self, xs, ys):
        batch_size, seq_len, _ = xs.shape
        x_tokens = torch.cat([xs, torch.zeros(batch_size, seq_len, 1, device=xs.device, dtype=xs.dtype)], dim=-1)
        y_tokens = torch.cat([torch.zeros(batch_size, seq_len, xs.size(-1), device=xs.device, dtype=xs.dtype), ys], dim=-1)
        return torch.stack([x_tokens, y_tokens], dim=2).reshape(batch_size, seq_len * 2, xs.size(-1) + 1)

    def get_qkv(self, hidden_states):
        batch_size, seq_len, channels = hidden_states.size()
        q, k, v = self.c_attn(hidden_states).split(self.hidden_dim, dim=2)
        q = q.view(batch_size, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)
        return q, k, v, channels

    def forward(self, xs, ys):
        _, seq_len, _ = xs.shape
        if seq_len > self.len_context:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.len_context}")
        sequence = self.build_sequence(xs, ys)
        positions = torch.arange(sequence.size(1), device=sequence.device, dtype=torch.long)
        hidden_states = self.input_projection(sequence) + self.position_embedding(positions)
        hidden_states = self.pre_attn_ln(self.dropout(hidden_states))
        q, k, v, channels = self.get_qkv(hidden_states)
        attended = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        hidden_states = attended.transpose(1, 2).contiguous().reshape(xs.size(0), seq_len * 2, channels)
        hidden_states = self.post_attn_ln(hidden_states)
        hidden_states = self.dropout(F.relu(self.mlp_layers[0](hidden_states)))
        for layer in self.mlp_layers[1:]:
            hidden_states = hidden_states + self.dropout(F.relu(layer(hidden_states)))
        x_positions = hidden_states[:, 0::2, :]
        return self.output_head(x_positions)
