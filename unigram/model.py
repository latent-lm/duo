import math
from typing import Union

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

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the time-conditioned MLP.

        Args:
            z (`torch.Tensor` of shape `(batch_size, io_dim)`):
                Input state in model coordinates.
            t (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, 1)`):
                Per-example time values.

        Returns:
            `torch.Tensor` of shape `(batch_size, output_dim)`:
                Predicted endpoint features.
        """
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([z, t], dim=-1))

class MLPLM(nn.Module):
    def __init__(self, vocab_size: int, input_dim: int, output_dim: int, hidden_size: int, depth: int):
        super().__init__()
        self.mlp = SmallMLP(
            input_dim=input_dim + 1,
            hidden_size=hidden_size,
            depth=depth,
            output_dim=output_dim,
        )

        self.lm_head = nn.Linear(output_dim, vocab_size, bias=False)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict vocabulary logits from a time-conditioned state.

        Args:
            z (`torch.Tensor` of shape `(batch_size, input_dim)`):
                Input state.
            t (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, 1)`):
                Per-example time values.

        Returns:
            `torch.Tensor` of shape `(batch_size, vocab_size)`:
                Vocabulary logits.
        """
        return self.lm_head(self.mlp(z=z, t=t))

def polar_to_cart(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Map polar disk coordinates to Cartesian disk coordinates.

    Args:
        rho (`torch.Tensor` of shape `(...,)`):
            Radial coordinates.
        theta (`torch.Tensor` of shape `(...,)`):
            Angular coordinates.

    Returns:
        `torch.Tensor` of shape `(..., 2)`:
            Cartesian coordinates. For example, if `rho` and `theta`
            have shape `(256,)`, the output has shape `(256, 2)`.
    """
    r = torch.tanh(rho / 2)
    return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

def vocab_points(V: int, device: Union[str, torch.device], dtype) -> torch.Tensor:
    """
    Construct evenly spaced vocabulary anchor points on the unit circle.

    Args:
        V (`int`):
            Vocabulary size.
        device (`str` or `torch.device`):
            Device for the returned tensor.
        dtype:
            Tensor dtype for the returned tensor.

    Returns:
        `torch.Tensor` of shape `(V, 2)`:
            Vocabulary anchor points in Cartesian coordinates. Internally,
            `phi` has shape `(V,)`. For example, when `V = 2`, `phi` has
            shape `(2,)` and the returned tensor has shape `(2, 2)`.
    """
    phi = (torch.arange(V, device=device, dtype=dtype) + 0.5) * (2 * math.pi / V)
    return torch.stack([phi.cos(), phi.sin()], dim=-1)

def poisson_posterior(
    x: torch.Tensor,
    v: torch.Tensor,
    log_ps: torch.Tensor,
    d: int = 2,
) -> torch.Tensor:
    """
    Compute the posterior expectation E_{y|x_t}[(y - x_t) / |y - x_t|^2].

    \mathbb{E}_{y | x_t} \left[ \frac{y - x_t}{||y-x_t ||^2} \right] 
        = \frac{1}{A_{d-1}} \left( 
            \sum_{y \in V} \left( 
                \frac{1 - \|x_t\|^2}{\|x_t - y\|^2} 
            \right)^{d-1} 
                \frac{y - x_t}{||y-x_t ||^2} 
        \right)

    Follow Bayes rule, q(y | x_t) = \frac{q(y) q(x_t | y)}{q(x_t)}
    The posterior is q(y | x_t) propto p(y) * |y - x_t|^{-2(d-1)}; the
    y-independent (1 - |x_t|^2)^{d-1} factor cancels in normalization.

    Args:
        x (`torch.Tensor` of shape `(..., d)`):
            Cartesian disk coordinates for the current state.
        v (`torch.Tensor` of shape `(V, d)`):
            Vocabulary anchor points.
        log_ps (`torch.Tensor` of shape `(V,)`):
            Log prior probabilities over the vocabulary.
        d (`int`, *optional*, defaults to `2`):
            Hyperbolic space dimension parameter in the Poisson kernel.

    Returns:
        `torch.Tensor` of shape `(..., d)`:
            Posterior expectation of (y - x_t) / |y - x_t|^2.
    """
    # x.unsqueeze(-2): (B, 1, d)
    # diff: (B, V, d), Broadcast x along with -2 dimension and compute difference
    diff = v - x.unsqueeze(-2)
    # sq: (B, V)
    sq = diff.square().sum(-1)
    # mu: (B, V)
    mu = (log_ps - (d - 1) * sq.log()).softmax(-1)
    # Weighted sum
    # (mu / sq).unsqueeze(-1): (B, V, 1)
    # (mu / sq).unsqueeze(-1).mul(diff): (B, V, d)
    # (mu / sq).unsqueeze(-1).mul(diff).sum(-2): (B, d)
    return (mu / sq).unsqueeze(-1).mul(diff).sum(-2)

def poisson_posterior_new(
    x: torch.Tensor,
    v: torch.Tensor,
    log_ps: torch.Tensor,
    d: int = 2,
) -> torch.Tensor:
    """
    Alternative Poisson-kernel posterior implementation.

    Args:
        x (`torch.Tensor` of shape `(..., 2)`):
            Cartesian disk coordinates for the current state.
        v (`torch.Tensor` of shape `(V, 2)`):
            Vocabulary anchor points.
        log_ps (`torch.Tensor` of shape `(V,)`):
            Log prior probabilities over the vocabulary.
        d (`int`, *optional*, defaults to `2`):
            Hyperbolic space dimension parameter in the Poisson kernel.

    Returns:
        `torch.Tensor` of shape `(..., V)`:
            Posterior probabilities over the vocabulary. As above,
            `sq = |v - x|^2` has shape `(..., V)`.
    """
    sq = (v - x.unsqueeze(-2)).square().sum(-1)
    return (log_ps - (d - 1) * sq.log()).softmax(-1)

def bridge_drift(
    x: torch.Tensor,
    expectation: torch.Tensor,
    d: int = 2,
) -> torch.Tensor:
    """
    Compute the Bayes-optimal bridge drift field.

    Args:
        x (`torch.Tensor` of shape `(..., d)`):
            Cartesian disk coordinates for the current state.
        expectation (`torch.Tensor` of shape `(..., d)`):
            Posterior expectation E_{y|x_t}[(y - x_t) / |y - x_t|^2], as
            returned by `poisson_posterior`.
        d (`int`, *optional*, defaults to `2`):
            Hyperbolic space dimension parameter.

    Returns:
        `torch.Tensor` of shape `(..., d)`:
            Drift vectors in Cartesian coordinates.
    """
    g2 = 1 - x.square().sum(-1, keepdim=True)
    return (d - 1) / 2 * g2.square() * expectation - d / 4 * g2 * x

class OptimalModel(nn.Module):
    """Bayes-optimal unigram model on D^2.

    Forward returns log-prior logits broadcast over the batch — these are the
    optimal logits for `binary_bridge_loss`, since
    `softmax(horosphere_dists + log_ps)` recovers the true posterior
    q(y | x_t) ∝ p(y) · |y - x_t|^{-2(d-1)} for d=2 (the (1-|x_t|^2) factor
    cancels in normalization). Use `optimal_drift` if you need the
    bridge-drift form instead (e.g., for the visualizer).
    """

    def __init__(self, ps):
        super().__init__()
        ps = torch.as_tensor(ps, dtype=torch.float64)
        self.register_buffer("log_ps", (ps / ps.sum()).log())
        print(f"self.ps: {self.log_ps.exp()}")
        print(f"self.log_ps: {self.log_ps}")

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Bayes-optimal logits.

        Args:
            z (`torch.Tensor` of shape `(..., 2)`):
                Polar disk coordinates `(rho, theta)`. Only the leading shape
                is used; coordinates themselves are ignored because the
                optimal logits are just `log p(y)` (the per-`x_t` Poisson
                kernel is supplied by the loss via horosphere distances).
            t (`torch.Tensor`):
                Time values. Ignored.

        Returns:
            `torch.Tensor` of shape `(..., V)`:
                Log-prior logits, broadcast to match the leading batch shape.
        """
        del t
        leading_shape = z.shape[:-1]
        return self.log_ps.to(dtype=torch.float32).expand(*leading_shape, -1)

    def optimal_drift(self, z: torch.Tensor) -> torch.Tensor:
        """
        Closed-form Bayes-optimal bridge drift in Cartesian coordinates.

        Args:
            z (`torch.Tensor` of shape `(..., 2)`):
                Polar disk coordinates `(rho, theta)`.

        Returns:
            `torch.Tensor` of shape `(..., 2)`:
                Bridge drift evaluated under the true posterior.
        """
        z = z.double()
        x = polar_to_cart(z[..., 0], z[..., 1])
        v = vocab_points(self.log_ps.numel(), z.device, z.dtype)
        expectation = poisson_posterior(x, v, self.log_ps)
        return bridge_drift(x, expectation).float()
