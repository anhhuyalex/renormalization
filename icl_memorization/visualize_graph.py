import torch
from torchviz import make_dot
from gpt import HeadsMLPGPT

# Dummy parameters for initialization
len_context = 100
num_heads = 2
num_hidden_attn_features = 16 # input features per head
num_input_mlp_features = 32
num_hidden_mlp_features = 64
vocab_size = 2
num_layers = 6

# Instantiate the model
model = HeadsMLPGPT(
    len_context=len_context,
    num_heads=num_heads,
    num_hidden_attn_features=num_hidden_attn_features,
    num_input_mlp_features=num_input_mlp_features,
    num_hidden_mlp_features=num_hidden_mlp_features,
    vocab_size=vocab_size,
    num_layers=num_layers
)

# Dummy input
batch_size = 1
seq_len = 100
idx = torch.randint(0, vocab_size, (batch_size, seq_len))
targets = torch.randint(0, vocab_size, (batch_size, seq_len)) # Assuming targets has same shape

# Forward pass
output = model(idx, targets)

# Generate visualization
dot = make_dot(output, params=dict(model.named_parameters())+ [('idx', idx), ('targets', targets)])
dot.format = 'png'
output_filename = 'heads_mlp_gpt_architecture'
dot.render(output_filename)

print(f"Visualization saved to {output_filename}.png")
