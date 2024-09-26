import torch

from torch import nn
from src.models.base import NODE_MLP

input_tensor = torch.randn(10, 2001)

model = NODE_MLP(
    input_size=2001,
    hidden_size=64,
    vf_depth=5,
    vf_hidden_size=256,
    output_size=1,
    dropout=0.,
    lr=1e-3,
    lr_schedule="RLROP",
    lr_patience=50,
    weight_decay=1e-5,
    activation="SiLU",
    decoder_depth=2,
    plot_interval=500,
    metric=nn.L1Loss,
)

output = model(input_tensor)