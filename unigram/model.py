import torch
import torch.nn as nn

class SmallMLP(nn.Module):
    """Tiny time-conditioned MLP that predicts a hyperbolic endpoint."""

    def __init__(self, input_dim: int, hidden_size: int, depth: int, output_dim: int):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.Tanh())
            dim = hidden_size
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([z, t], dim=-1))

class MLPLM(nn.Module):
    def __init__(self, vocab_size: int, io_dim: int, hidden_size: int, depth: int):
        super().__init__()
        self.mlp = SmallMLP(
            input_dim=io_dim + 1,
            hidden_size=hidden_size,
            depth=depth,
            output_dim=io_dim,
        )

        self.lm_head = nn.Linear(io_dim, vocab_size, bias=False)

    def forward(self, z: torch.Tensor, t: torch.Tensor):
        return self.lm_head(self.mlp(z=z, t=t))
