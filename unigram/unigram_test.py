"""Minimal unigram sanity test for the hyperbolic DLM in ``hyper_dm.md``.

This experiment is intentionally simple:
  - vocabulary size = 2
  - sequence length = 1
  - dataset distribution: P(A)=0.8, P(B)=0.2
  - model: a small MLP

It is designed to verify the inequality chain

    NELBO >= NLL >= H(Y)

in the regime where both KL gaps should shrink.  The script trains with one
of five objectives:
  - cross entropy surrogate
  - L2 surrogate
  - diffusion surrogate
  - full NELBO
  - discrete NELBO

Regardless of the training loss, the script always evaluates the NELBO
separately and visualizes it against the dataset entropy over optimization
steps.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _validate_hyper_embed_length(hyper_embed_length: float):
    if not 0.0 < hyper_embed_length <= 1.0:
        raise ValueError("hyper_embed_length must satisfy 0 < hyper_embed_length < 1")


@dataclass
class ExperimentConfig:
    loss: str = "nelbo"
    train_size: int = 20_000
    val_size: int = 4_000
    batch_size: int = 256
    max_steps: int = 2_000
    lr: float = 1e-3
    hidden_size: int = 64
    depth: int = 2
    hyper_dim: int = 2
    hyper_T: int = 1000
    hyper_dt: float = 0.01
    hyper_embed_length: float = 1.0
    max_ball_norm: float = 0.99
    p_a: float = 0.8
    seed: int = 0
    num_workers: int = 0
    history_every: int = 10
    proposal: str = "fixed_interval"
    proposal_start: float | None = None
    proposal_end: float | None = None
    proposal_exp_lambda: float = 1.0
    inference_dt: float | None = None
    eval_num_samples: int = 4096
    discrete_temperature: float = 1.0
    discrete_approximation: str = "diag_probit"
    discrete_gauss_hermite_order: int = 32
    discrete_train_mc_samples: int = 16
    discrete_eval_mc_samples: int = 128
    plot_path: str = "unigram_test.png"

    def __post_init__(self):
        _validate_hyper_embed_length(self.hyper_embed_length)
        if self.inference_dt is None:
            self.inference_dt = self.hyper_dt
        if self.inference_dt <= 0:
            raise ValueError("inference_dt must be > 0")
        if self.eval_num_samples < 1:
            raise ValueError("eval_num_samples must be >= 1")
        if self.discrete_temperature <= 0:
            raise ValueError("discrete_temperature must be > 0")
        if self.discrete_approximation not in {"diag_probit", "gauss_hermite"}:
            raise ValueError(
                "discrete_approximation must be one of "
                "{'diag_probit', 'gauss_hermite'}"
            )
        if self.discrete_gauss_hermite_order < 1:
            raise ValueError("discrete_gauss_hermite_order must be >= 1")
        if self.discrete_train_mc_samples < 1:
            raise ValueError("discrete_train_mc_samples must be >= 1")
        if self.discrete_eval_mc_samples < 1:
            raise ValueError("discrete_eval_mc_samples must be >= 1")


@dataclass
class ProposalSchedule:
    tau: torch.Tensor
    delta_tau: torch.Tensor


class ProposalGenerator:
    """Build the reverse-time schedule used by training and inference."""

    def __init__(
        self,
        num_steps: int,
        mode: str,
        start: float,
        end: float,
        exp_lambda: float = 1.0,
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if mode not in {"fixed_interval", "uniform", "exponential"}:
            raise ValueError(f"Unsupported proposal mode: {mode}")
        if start <= 0:
            raise ValueError("Proposal start must be > 0")
        if end < start:
            raise ValueError("Proposal end must be >= proposal start")
        if exp_lambda <= 0:
            raise ValueError("Exponential proposal rate must be > 0")
        self.num_steps = num_steps
        self.mode = mode
        self.start = start
        self.end = end
        self.exp_lambda = exp_lambda

    @classmethod
    def from_config(cls, config: ExperimentConfig):
        start = (
            config.hyper_dt if config.proposal_start is None else config.proposal_start
        )
        end = (
            config.hyper_T * config.hyper_dt
            if config.proposal_end is None
            else config.proposal_end
        )
        return cls(
            num_steps=config.hyper_T,
            mode=config.proposal,
            start=start,
            end=end,
            exp_lambda=config.proposal_exp_lambda,
        )

    @classmethod
    def fixed_step(cls, num_steps: int, delta_tau: float):
        return cls(
            num_steps=num_steps,
            mode="fixed_interval",
            start=delta_tau,
            end=num_steps * delta_tau,
        )

    def _quantile_grid(self, device: torch.device, dtype: torch.dtype):
        return (
            torch.arange(1, self.num_steps + 1, device=device, dtype=dtype)
            / self.num_steps
        )

    def _generate_tau(self, device: torch.device, dtype: torch.dtype):
        if self.mode == "fixed_interval":
            tau = torch.linspace(
                self.start, self.end, steps=self.num_steps, device=device, dtype=dtype
            )
        elif self.mode == "uniform":
            quantiles = self._quantile_grid(device, dtype)
            tau = self.start + (self.end - self.start) * quantiles
        else:
            quantiles = self._quantile_grid(device, dtype).clamp(max=1 - 1e-6)
            tau = -torch.log1p(-quantiles) / self.exp_lambda
        return tau.sort().values

    def generate(
        self, device: torch.device | None = None, dtype: torch.dtype = torch.float32
    ):
        if device is None:
            device = torch.device("cpu")
        tau = self._generate_tau(device, dtype)
        tau0 = torch.cat([tau.new_zeros(1), tau], dim=0)
        delta_tau = tau0[1:] - tau0[:-1]
        return ProposalSchedule(tau=tau, delta_tau=delta_tau)


class HyperbolicBridge(nn.Module):
    """Hyperbolic bridge utilities on the Poincare disk."""

    def __init__(
        self,
        hyper_dim: int,
        hyper_embed_length: float = 1.0,
        max_ball_norm: float = 0.9999,
        discrete_temperature: float = 1.0,
        discrete_approximation: str = "diag_probit",
        discrete_gauss_hermite_order: int = 32,
    ):
        super().__init__()
        assert hyper_dim >= 2
        _validate_hyper_embed_length(hyper_embed_length)
        if discrete_temperature <= 0:
            raise ValueError("discrete_temperature must be > 0")
        if discrete_approximation not in {"diag_probit", "gauss_hermite"}:
            raise ValueError(
                "discrete_approximation must be one of "
                "{'diag_probit', 'gauss_hermite'}"
            )
        if discrete_gauss_hermite_order < 1:
            raise ValueError("discrete_gauss_hermite_order must be >= 1")
        self.hyper_dim = hyper_dim
        self.hyper_embed_length = hyper_embed_length
        self.max_ball_norm = max_ball_norm
        self.discrete_temperature = discrete_temperature
        self.discrete_approximation = discrete_approximation

        endpoints = torch.zeros(2, hyper_dim, dtype=torch.float32)
        endpoints[0, 0] = self.hyper_embed_length
        endpoints[1, 0] = -self.hyper_embed_length
        self.register_buffer("endpoints", endpoints, persistent=False)
        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(
            discrete_gauss_hermite_order
        )
        self.register_buffer(
            "gh_nodes", torch.tensor(gh_nodes, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "gh_weights",
            torch.tensor(gh_weights, dtype=torch.float64),
            persistent=False,
        )

    def normalize_endpoint(self, endpoint: torch.Tensor):
        return (
            endpoint
            / endpoint.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            * self.hyper_embed_length
        )

    def endpoint_posterior(self, endpoint: torch.Tensor):
        endpoint = self.normalize_endpoint(endpoint)
        return self.discrete_logits(endpoint).softmax(dim=-1)

    def discrete_logits(self, state: torch.Tensor):
        return state @ self.endpoints.t() / self.discrete_temperature

    def discrete_posterior(self, state: torch.Tensor):
        return self.discrete_logits(state).softmax(dim=-1)

    def project_to_discrete(self, endpoint: torch.Tensor):
        return (endpoint @ self.endpoints.t()).argmax(dim=-1)

    def clamp_to_ball(self, z: torch.Tensor):
        znorm = z.norm(dim=-1, keepdim=True).clamp(min=1e-32)
        return z * (znorm.clamp(max=self.max_ball_norm) / znorm)

    def project_state(self, z: torch.Tensor):
        return self.clamp_to_ball(z)

    def sample_prior_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ):
        return torch.zeros(batch_size, self.hyper_dim, device=device, dtype=dtype)

    def _expand_like_state(self, value: torch.Tensor | float, z: torch.Tensor):
        value = torch.as_tensor(value, device=z.device, dtype=z.dtype)
        while value.ndim < z.ndim:
            value = value.unsqueeze(-1)
        return value

    def drift(self, z: torch.Tensor, y: torch.Tensor):
        conf = (1 - z.square().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
        diff = y - z
        sq_diff = diff.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        bridge = ((self.hyper_dim - 1) / 2) * conf.square() / sq_diff * diff
        ito = -(self.hyper_dim / 4) * conf * z
        return bridge + ito

    def diffusion(self, z: torch.Tensor):
        conf = (1 - z.square().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
        return conf / 2

    def bridge_step(
        self, z: torch.Tensor, y: torch.Tensor, delta_tau: torch.Tensor | float
    ):
        delta_tau = self._expand_like_state(delta_tau, z)
        drift = self.drift(z, y)
        noise = self.diffusion(z) * delta_tau.sqrt() * torch.randn_like(z)
        return self.clamp_to_ball(z + drift * delta_tau + noise)

    def sample_reverse_path(self, y: torch.Tensor, schedule: ProposalSchedule):
        """Simulate q(x_{tau(i)} | y) marginals with x_{tau(T)} = 0."""
        y64 = self.normalize_endpoint(y).to(torch.float64)
        z = torch.zeros_like(y64)
        states = [None] * (schedule.delta_tau.numel() + 1)
        states[-1] = z.to(torch.float32)
        for idx in range(schedule.delta_tau.numel() - 1, -1, -1):
            z = self.bridge_step(z, y64, schedule.delta_tau[idx])
            states[idx] = z.to(torch.float32)
        return torch.stack(states, dim=1)

    def diffusion_term(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
    ):
        pred_n = self.normalize_endpoint(pred)
        diff_y = y - z
        diff_pred = pred_n - z
        dir_y = diff_y / diff_y.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        dir_pred = diff_pred / diff_pred.square().sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        conf = (1 - z.square().sum(dim=-1)).clamp(min=1e-8)
        error = (dir_y - dir_pred).square().sum(dim=-1)
        delta_tau = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype)
        return 0.5 * delta_tau * (self.hyper_dim - 1) ** 2 * conf.square() * error

    def reconstruction_term(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
    ):
        pred_n = self.normalize_endpoint(pred)
        conf = (1 - z.square().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
        diff_pred = pred_n - z
        sq_diff = diff_pred.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        drift = ((self.hyper_dim - 1) / 2) * conf.square() / sq_diff * diff_pred - (
            self.hyper_dim / 4
        ) * conf * z
        delta_tau_state = self._expand_like_state(delta_tau, z)
        residual = y - z - drift * delta_tau_state

        log_dt = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype).log()
        log_2pi = z.new_tensor(2 * math.pi).log()
        normalizer = (
            0.5 * self.hyper_dim * log_2pi
            + self.hyper_dim * (conf / 2).log().squeeze(-1)
            + 0.5 * self.hyper_dim * log_dt
        )
        quad = (
            2.0
            / (conf.square() * delta_tau_state)
            * residual.square().sum(dim=-1, keepdim=True)
        )
        return normalizer + quad.squeeze(-1)

    def prior_term(self, batch_size: int, device: torch.device):
        # The discrete derivation fixes q(x_{tau(T)} | y) = p(x_{tau(T)}) = delta_0.
        return torch.zeros(batch_size, device=device, dtype=torch.float32)

    def _expand_transition_step(self, delta_tau: torch.Tensor | float, z: torch.Tensor):
        delta_tau = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype)
        while delta_tau.ndim < z.ndim - 1:
            delta_tau = delta_tau.unsqueeze(0)
        return delta_tau.unsqueeze(-1)

    def _local_chart_transition_moments(
        self, z: torch.Tensor, y: torch.Tensor, delta_tau: torch.Tensor | float
    ):
        delta_tau_state = self._expand_transition_step(delta_tau, z)
        y = self.normalize_endpoint(y)
        mean_state = z + self.drift(z, y) * delta_tau_state
        state_var = self.diffusion(z).square() * delta_tau_state
        return mean_state, state_var

    def _logit_moments_from_state_moments(
        self, mean_state: torch.Tensor, state_var: torch.Tensor
    ):
        logits_mean = self.discrete_logits(mean_state)
        endpoint_sq_norm = self.endpoints.square().sum(dim=-1).to(
            device=mean_state.device, dtype=mean_state.dtype
        )
        view_shape = (1,) * (mean_state.ndim - 1) + (endpoint_sq_norm.numel(),)
        logits_var = (
            state_var
            * endpoint_sq_norm.view(view_shape)
            / (self.discrete_temperature**2)
        )
        return logits_mean, logits_var

    def _sample_transition(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
    ):
        del num_samples
        # The local-chart discrete NELBO uses analytic Gaussian moments
        # of the one-step transition instead of Monte Carlo samples.
        return self._local_chart_transition_moments(z, y, delta_tau)

    def _monte_carlo_transition_logits(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
    ):
        mean_state, state_var = self._sample_transition(
            z, y, delta_tau, num_samples=num_samples
        )
        logits_mean, logits_var = self._logit_moments_from_state_moments(
            mean_state, state_var
        )
        return logits_mean, logits_var, state_var.squeeze(-1)

    def _diagonal_probit_probs(
        self, logits_mean: torch.Tensor, logits_var: torch.Tensor
    ):
        logits_var = logits_var.clamp(min=0.0)
        scale = torch.sqrt(1.0 + (math.pi / 8.0) * logits_var).clamp(min=1e-8)
        probs = (logits_mean / scale).softmax(dim=-1).clamp(min=1e-8)
        return probs / probs.sum(dim=-1, keepdim=True)

    def _monte_carlo_transition_logits_stats(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
    ):
        logits_mean, logits_var, _ = self._monte_carlo_transition_logits(
            z, y, delta_tau, num_samples=num_samples
        )
        return logits_mean, logits_var

    def _binary_gauss_hermite_probs(
        self,
        logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
    ):
        if isinstance(logits, tuple):
            logits_mean, _, state_var = logits
        else:
            logits_mean = logits.mean(dim=0)
            state_var = None
        if logits_mean.shape[-1] != 2:
            raise ValueError(
                "gauss_hermite discrete approximation currently only supports K=2"
            )
        gap_mean = logits_mean[..., 0] - logits_mean[..., 1]
        if state_var is None:
            gap = logits[..., 0] - logits[..., 1]
            gap_var = (gap.square().mean(dim=0) - gap_mean.square()).clamp(min=0.0)
        else:
            endpoint_gap = (self.endpoints[0] - self.endpoints[1]).to(
                device=gap_mean.device, dtype=gap_mean.dtype
            )
            gap_var = (
                state_var
                * endpoint_gap.square().sum()
                / (self.discrete_temperature**2)
            ).clamp(min=0.0)
        gap_std = gap_var.sqrt()

        nodes = self.gh_nodes.to(device=gap_mean.device, dtype=gap_mean.dtype)
        weights = self.gh_weights.to(device=gap_mean.device, dtype=gap_mean.dtype)
        shape = (nodes.shape[0],) + (1,) * gap_mean.ndim
        quadrature_points = gap_mean.unsqueeze(0) + math.sqrt(2.0) * gap_std.unsqueeze(
            0
        ) * nodes.view(shape)
        prob_0 = (weights.view(shape) * torch.sigmoid(quadrature_points)).sum(
            dim=0
        ) / math.sqrt(math.pi)
        prob_0 = prob_0.clamp(min=1e-8, max=1 - 1e-8)
        return torch.stack([prob_0, 1 - prob_0], dim=-1)

    def _approximate_discrete_transition_probs(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
        method: str | None = None,
    ):
        del num_samples
        if method is None:
            method = self.discrete_approximation
        if method == "diag_probit":
            logits_mean, logits_var = self._monte_carlo_transition_logits_stats(
                z, y, delta_tau, num_samples=0
            )
            return self._diagonal_probit_probs(logits_mean, logits_var)
        if method == "gauss_hermite":
            logits = self._monte_carlo_transition_logits(
                z, y, delta_tau, num_samples=0
            )
            return self._binary_gauss_hermite_probs(logits)
        raise ValueError(f"Approximation method {method} is not supported.")

    def discrete_diffusion_term(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
    ):
        pred_n = self.normalize_endpoint(pred)
        # This is L_Discrete in hyper_dm.md: the categorical KL between
        # local-chart one-step bridge approximations under q and p_theta.
        q_probs = self._approximate_discrete_transition_probs(
            z, y, delta_tau, num_samples=num_samples
        )
        p_probs = self._approximate_discrete_transition_probs(
            z, pred_n, delta_tau, num_samples=num_samples
        )
        return (q_probs * (q_probs.log() - p_probs.log())).sum(dim=-1)

    def discrete_reconstruction_term(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
    ):
        pred_n = self.normalize_endpoint(pred)
        # Approximate p_theta(w_0 | x_1) by projecting the local-chart
        # Gaussian reverse step through the discrete emission model.
        probs = self._approximate_discrete_transition_probs(
            z, pred_n, delta_tau, num_samples=num_samples
        )
        targets = self.project_to_discrete(y).unsqueeze(-1)
        return -probs.log().gather(-1, targets).squeeze(-1)

    def discrete_refinement_term(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
    ):
        pred_n = self.normalize_endpoint(pred)
        mean_q, state_var = self._local_chart_transition_moments(z, y, delta_tau)
        mean_p, _ = self._local_chart_transition_moments(z, pred_n, delta_tau)
        # Use the same approximate E[softmax(Ex / T)] moments as the discrete
        # denoising term inside the Stein mean shift.
        pi_q = self._approximate_discrete_transition_probs(
            z, y, delta_tau, num_samples=0
        )
        pi_p = self._approximate_discrete_transition_probs(
            z, pred_n, delta_tau, num_samples=0
        )
        endpoints = self.endpoints.to(device=z.device, dtype=z.dtype)
        feedback = (
            state_var / self.discrete_temperature
        ) * ((pi_q - pi_p) @ endpoints)
        delta = (mean_p - mean_q) + feedback
        return delta.square().sum(dim=-1) / (
            2.0 * state_var.squeeze(-1).clamp(min=1e-8)
        )

    def discrete_initial_term(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
        num_samples: int,
    ):
        pred_n = self.normalize_endpoint(pred)
        # In the unigram setup x_0 is the deterministic endpoint selected by
        # w_0, so L_initial reduces to -log p_theta(x_0 = y | w_0, x_1).
        # Using Bayes:
        #   p_theta(y | w_0, x_1)
        #   = p_theta(w_0 | y) p_theta(y | x_1) / p_theta(w_0 | x_1).
        gaussian_nll = self.reconstruction_term(z, y, pred_n, delta_tau)
        discrete_nll = self.discrete_reconstruction_term(
            z, y, pred_n, delta_tau, num_samples=num_samples
        )
        targets = self.project_to_discrete(y).unsqueeze(-1)
        endpoint_emission_nll = -self.endpoint_posterior(y).clamp(min=1e-8).log().gather(
            -1, targets
        ).squeeze(-1)
        return gaussian_nll + endpoint_emission_nll - discrete_nll


class EulerScheduler:
    def __init__(self, bridge: nn.Module, model: nn.Module):
        self.bridge = bridge
        self.model = model

    def _device(self):
        if hasattr(self.bridge, "endpoints"):
            return self.bridge.endpoints.device
        return next(self.model.parameters()).device

    def _expand_step(self, delta_tau: torch.Tensor | float, z: torch.Tensor):
        delta_tau = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype)
        while delta_tau.ndim < z.ndim:
            delta_tau = delta_tau.unsqueeze(-1)
        return delta_tau

    def _project_state(self, z: torch.Tensor):
        if hasattr(self.bridge, "project_state"):
            return self.bridge.project_state(z)
        return z

    def _noise(self, z: torch.Tensor):
        return torch.randn_like(z)

    def _predict_target(self, z: torch.Tensor, t: torch.Tensor):
        pred = self.model(
            z.to(dtype=torch.float32), t.to(device=z.device, dtype=torch.float32)
        )
        pred = pred.to(dtype=z.dtype)
        if hasattr(self.bridge, "normalize_endpoint"):
            return self.bridge.normalize_endpoint(pred)
        return pred

    def step(self, z: torch.Tensor, t: torch.Tensor, delta_tau: torch.Tensor | float):
        pred = self._predict_target(z=z, t=t)
        delta_tau_state = self._expand_step(delta_tau, z)
        drift = self.bridge.drift(z=z, y=pred)
        noise = self.bridge.diffusion(z=z) * delta_tau_state.sqrt() * self._noise(z)
        z = z + drift * delta_tau_state + noise
        return self._project_state(z)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        schedule: ProposalSchedule | None = None,
        num_inference_steps: int | None = None,
        delta_tau: float | None = None,
        init: torch.Tensor | None = None,
        return_path: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        if device is None:
            device = self._device()
        if schedule is None:
            if num_inference_steps is None or delta_tau is None:
                raise ValueError(
                    "Either schedule or both num_inference_steps and delta_tau are "
                    "required."
                )
            schedule = ProposalGenerator.fixed_step(
                num_steps=num_inference_steps, delta_tau=delta_tau
            ).generate(device=device)
        if init is None:
            if not hasattr(self.bridge, "sample_prior_state"):
                raise ValueError("bridge must implement sample_prior_state for init.")
            z = self.bridge.sample_prior_state(
                batch_size=batch_size, device=device, dtype=dtype
            )
        else:
            z = init.to(device=device, dtype=dtype)
            batch_size = z.shape[0]

        path = [z] if return_path else None
        for idx in range(schedule.delta_tau.numel() - 1, -1, -1):
            tau_t = schedule.tau[idx].expand(batch_size).to(device=device)
            z = self.step(z=z, t=tau_t, delta_tau=schedule.delta_tau[idx])
            if path is not None:
                path.append(z)
        if return_path:
            return z, torch.stack(path, dim=1)
        return z


class Evaluator:
    def _normalize_dist(self, dist: torch.Tensor):
        dist = dist.to(dtype=torch.float32)
        return dist / dist.sum().clamp(min=1e-8)

    def _vocab_size(
        self,
        bridge: nn.Module,
        label_dist: torch.Tensor | None,
        labels: torch.Tensor | None,
    ):
        if label_dist is not None:
            return label_dist.numel()
        if hasattr(bridge, "endpoints"):
            return bridge.endpoints.shape[0]
        if labels is None:
            raise ValueError("Unable to infer vocab size.")
        return int(labels.max().item()) + 1

    def unigram_distribution(self, tokens: torch.Tensor, vocab_size: int):
        tokens = tokens.reshape(-1).to(dtype=torch.long)
        counts = torch.bincount(tokens, minlength=vocab_size).to(dtype=torch.float32)
        return self._normalize_dist(counts)

    def unigram_entropy(self, dist: torch.Tensor):
        dist = self._normalize_dist(dist).clamp(min=1e-8)
        return -(dist * dist.log()).sum()

    def unigram_crossentropy(self, pred_dist: torch.Tensor, label_dist: torch.Tensor):
        pred_dist = self._normalize_dist(pred_dist).clamp(min=1e-8)
        label_dist = self._normalize_dist(label_dist)
        return -(label_dist * pred_dist.log()).sum()

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        bridge: nn.Module,
        scheduler: EulerScheduler,
        num_inference_steps: int | None = None,
        batch_size: int = 1,
        labels: torch.Tensor | None = None,
        label_dist: torch.Tensor | None = None,
        schedule: ProposalSchedule | None = None,
        delta_tau: float = 0.03,
    ):
        vocab_size = self._vocab_size(bridge, label_dist, labels)
        if label_dist is None:
            if labels is None:
                raise ValueError("Either labels or label_dist must be provided.")
            label_dist = self.unigram_distribution(labels, vocab_size=vocab_size)
        else:
            label_dist = self._normalize_dist(label_dist)

        was_training = model.training
        model.eval()
        states = scheduler.generate(
            batch_size=batch_size,
            schedule=schedule,
            num_inference_steps=num_inference_steps,
            delta_tau=delta_tau,
        )
        if was_training:
            model.train()
        if hasattr(bridge, "normalize_endpoint"):
            states = bridge.normalize_endpoint(states.to(dtype=torch.float32))
        tokens = bridge.project_to_discrete(states)
        pred_dist = self.unigram_distribution(tokens, vocab_size=vocab_size)
        print(f"pred_dist: {pred_dist}")
        print(f"label_dist: {label_dist}")
        return {
            "pred_dist": pred_dist,
            "label_dist": label_dist,
            "pred_entropy": self.unigram_entropy(pred_dist),
            "label_entropy": self.unigram_entropy(label_dist),
            "unigram_crossentropy": self.unigram_crossentropy(
                pred_dist=pred_dist, label_dist=label_dist
            ),
            "tokens": tokens,
        }


class Visualizer:
    def plot_poincare_disk_bridge(
        self,
        bridge: HyperbolicBridge,
        scheduler: EulerScheduler,
        schedule: ProposalSchedule,
        output_path: str | Path,
        labels: torch.Tensor | None = None,
        num_paths_per_label: int = 4,
    ):
        if getattr(bridge, "hyper_dim", None) != 2:
            raise ValueError("plot_poincare_disk_bridge requires hyper_dim=2.")
        if not hasattr(bridge, "endpoints"):
            raise ValueError("bridge must expose endpoints for label visualization.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = bridge.endpoints.device
        if labels is None:
            labels = torch.arange(bridge.endpoints.shape[0], device=device)
        labels = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)
        targets = bridge.normalize_endpoint(bridge.endpoints[labels]).to(torch.float32)
        prior = bridge.sample_prior_state(
            batch_size=1, device=device, dtype=targets.dtype
        )[0].cpu()
        batch_size = labels.numel() * num_paths_per_label

        was_training = scheduler.model.training
        scheduler.model.eval()
        with torch.no_grad():
            final_state, path = scheduler.generate(
                batch_size=batch_size,
                schedule=schedule,
                device=device,
                dtype=torch.float64,
                return_path=True,
            )
        if was_training:
            scheduler.model.train()

        final_boundary = bridge.normalize_endpoint(final_state.to(dtype=torch.float32))
        pred_tokens = bridge.project_to_discrete(final_boundary).cpu()
        path = path.to(dtype=torch.float32).cpu()

        theta = torch.linspace(0, 2 * math.pi, steps=512)
        circle_x = theta.cos().cpu().numpy()
        circle_y = theta.sin().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.fill(circle_x, circle_y, color="#f7f7f7", zorder=0)
        ax.plot(circle_x, circle_y, "k-", linewidth=2)
        for radius in (0.2, 0.4, 0.6, 0.8):
            ax.plot(
                radius * circle_x, radius * circle_y, "k-", alpha=0.06, linewidth=0.5
            )

        cmap = plt.get_cmap("tab10")
        for idx, label in enumerate(labels.tolist()):
            color = cmap(idx % 10)
            label_paths = path[pred_tokens == label]
            for path_idx, label_path in enumerate(label_paths):
                ax.plot(
                    label_path[:, 0],
                    label_path[:, 1],
                    "-",
                    color=color,
                    alpha=0.9 if path_idx == 0 else 0.2,
                    linewidth=2.0 if path_idx == 0 else 0.8,
                )
            ax.plot(
                targets[idx, 0].cpu(),
                targets[idx, 1].cpu(),
                "*",
                color=color,
                markersize=16,
                zorder=5,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=f"Label {label}",
            )
            if label_paths.numel() > 0:
                ends = label_paths[:, -1, :]
                ax.scatter(
                    ends[:, 0],
                    ends[:, 1],
                    color=color,
                    s=36,
                    alpha=0.9,
                    zorder=6,
                    edgecolors="k",
                    linewidths=0.4,
                )

        ax.plot(
            prior[0],
            prior[1],
            "o",
            color="limegreen",
            markersize=10,
            zorder=7,
            markeredgecolor="k",
            markeredgewidth=0.8,
            label="Origin (prior)",
        )
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(r"Model-Generated Paths on Poincar\'e Disk $\mathbb{D}^2$")
        ax.grid(True, alpha=0.15)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class UnigramHyperbolicDLM(L.LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.config = config
        self.bridge = HyperbolicBridge(
            hyper_dim=config.hyper_dim,
            hyper_embed_length=config.hyper_embed_length,
            max_ball_norm=config.max_ball_norm,
            discrete_temperature=config.discrete_temperature,
            discrete_approximation=config.discrete_approximation,
            discrete_gauss_hermite_order=config.discrete_gauss_hermite_order,
        )
        self.proposal_generator = ProposalGenerator.from_config(config)
        total_time = config.hyper_T * config.hyper_dt
        num_inference_steps = max(1, round(total_time / config.inference_dt))
        self.inference_generator = ProposalGenerator.fixed_step(
            num_steps=num_inference_steps, delta_tau=config.inference_dt
        )
        self.model = SmallMLP(
            input_dim=config.hyper_dim + 1,
            hidden_size=config.hidden_size,
            depth=config.depth,
            output_dim=config.hyper_dim,
        )
        self.euler_scheduler = EulerScheduler(bridge=self.bridge, model=self.model)
        self.evaluator = Evaluator()
        self._test_label_counts: torch.Tensor | None = None
        self.entropy_nats = -(
            config.p_a * math.log(config.p_a)
            + (1 - config.p_a) * math.log(1 - config.p_a)
        )
        self.history = {
            "train_step": [],
            "train_loss": [],
            "train_nelbo": [],
            "train_discrete_nelbo": [],
            "train_prior": [],
            "train_diffusion": [],
            "train_discrete_diffusion": [],
            "train_discrete_refinement": [],
            "train_reconstruction": [],
            "train_discrete_reconstruction": [],
            "train_discrete_initial": [],
            "train_recon_conf_sq": [],
            "train_recon_last_step": [],
            "train_horocycle_scale": [],
            "train_last_step_diffusion": [],
            "val_step": [],
            "val_loss": [],
            "val_nelbo": [],
            "val_discrete_nelbo": [],
            "val_prior": [],
            "val_diffusion": [],
            "val_discrete_diffusion": [],
            "val_discrete_refinement": [],
            "val_reconstruction": [],
            "val_discrete_reconstruction": [],
            "val_discrete_initial": [],
            "val_recon_conf_sq": [],
            "val_recon_last_step": [],
            "val_horocycle_scale": [],
            "val_last_step_diffusion": [],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def _time_from_tau(self, tau: torch.Tensor):
        return tau.to(dtype=torch.float32, device=self.device)

    def _reconstruction_residual_terms(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        delta_tau: torch.Tensor | float,
    ):
        pred_n = self.bridge.normalize_endpoint(pred)
        conf = (1 - z.square().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
        diff = pred_n - z
        sq_diff = diff.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        drift = ((self.config.hyper_dim - 1) / 2) * conf.square() / sq_diff * diff - (
            self.config.hyper_dim / 4
        ) * conf * z
        delta_tau = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype)
        residual = y - z - drift * delta_tau
        return pred_n, conf, diff, residual

    def _proposal_schedule(self):
        return self.proposal_generator.generate(device=self.device)

    def _inference_schedule(self):
        return self.inference_generator.generate(device=self.device)

    def _discrete_mc_samples(self):
        if self.training:
            return self.config.discrete_train_mc_samples
        return self.config.discrete_eval_mc_samples

    def _vocab_size(self):
        return int(self.bridge.endpoints.shape[0])

    def _token_counts(self, tokens: torch.Tensor):
        return torch.bincount(
            tokens.reshape(-1).to(dtype=torch.long).cpu(),
            minlength=self._vocab_size(),
        ).to(dtype=torch.float32)

    def _evaluate_unigram_batch(self, labels: torch.Tensor):
        labels = labels.to(dtype=torch.long, device=self.device)
        return self.evaluator.evaluate(
            model=self.model,
            bridge=self.bridge,
            scheduler=self.euler_scheduler,
            schedule=self._inference_schedule(),
            batch_size=labels.shape[0],
            labels=labels,
        )

    def _predict_boundary(self, z: torch.Tensor, tau: torch.Tensor):
        flat_z = z.reshape(-1, z.shape[-1])
        flat_tau = self._time_from_tau(tau).reshape(-1)
        pred = self.model(flat_z, flat_tau)
        return pred.reshape(*z.shape[:-1], -1)

    def _prepare_targets(self, tokens: torch.Tensor):
        y = self.bridge.normalize_endpoint(self.bridge.endpoints[tokens])
        discrete = self.bridge.project_to_discrete(y)
        return y, discrete

    def _sample_training_states(self, y: torch.Tensor):
        schedule = self._proposal_schedule()
        with torch.no_grad():
            states = self.bridge.sample_reverse_path(y, schedule)
        return schedule, states

    def _compute_surrogate_losses(
        self,
        z_diff: torch.Tensor,
        y_diff: torch.Tensor,
        pred_diff: torch.Tensor,
        discrete_tokens: torch.Tensor,
    ):
        posterior_diff = self.bridge.endpoint_posterior(pred_diff).clamp(min=1e-8)
        ce = (
            -posterior_diff.log()
            .gather(
                -1,
                discrete_tokens[:, None, None].expand(-1, z_diff.shape[1], 1),
            )
            .squeeze(-1)
            .sum(dim=-1)
        )
        l2 = (pred_diff - y_diff).square().sum(dim=-1).sum(dim=-1)
        return {"ce": ce, "l2": l2}

    def _compute_diffusion_losses(
        self,
        states: torch.Tensor,
        schedule: ProposalSchedule,
        y: torch.Tensor,
        discrete_tokens: torch.Tensor,
    ):
        batch_size = y.shape[0]
        discrete_mc_samples = self._discrete_mc_samples()
        z_diff = states[:, 2:, :]
        tau_diff = schedule.tau[1:].expand(batch_size, -1)
        y_diff = y[:, None, :].expand_as(z_diff)
        pred_diff = self._predict_boundary(z_diff, tau_diff)
        surrogate_losses = self._compute_surrogate_losses(
            z_diff, y_diff, pred_diff, discrete_tokens
        )
        diffusion = self.bridge.diffusion_term(
            z_diff, y_diff, pred_diff, schedule.delta_tau[1:]
        ).sum(dim=-1)
        discrete_diffusion = self.bridge.discrete_diffusion_term(
            z_diff,
            y_diff,
            pred_diff,
            schedule.delta_tau[1:],
            num_samples=discrete_mc_samples,
        ).sum(dim=-1)
        discrete_refinement = self.bridge.discrete_refinement_term(
            z_diff, y_diff, pred_diff, schedule.delta_tau[1:]
        ).sum(dim=-1)
        return {
            **surrogate_losses,
            "diffusion": diffusion,
            "discrete_diffusion": discrete_diffusion,
            "discrete_refinement": discrete_refinement,
        }

    def _compute_reconstruction_losses(
        self, states: torch.Tensor, schedule: ProposalSchedule, y: torch.Tensor
    ):
        batch_size = y.shape[0]
        discrete_mc_samples = self._discrete_mc_samples()
        z_recon = states[:, 1, :]
        tau_recon = schedule.tau[0].expand(batch_size)
        delta_tau_recon = schedule.delta_tau[0]
        pred_recon = self._predict_boundary(z_recon, tau_recon)
        recon = self.bridge.reconstruction_term(z_recon, y, pred_recon, delta_tau_recon)
        discrete_recon = self.bridge.discrete_reconstruction_term(
            z_recon, y, pred_recon, delta_tau_recon, num_samples=discrete_mc_samples
        )
        discrete_initial = self.bridge.discrete_initial_term(
            z_recon,
            y,
            pred_recon,
            delta_tau_recon,
            num_samples=discrete_mc_samples,
        )
        _, recon_conf, recon_diff, recon_residual = self._reconstruction_residual_terms(
            z_recon, y, pred_recon, delta_tau_recon
        )
        recon_conf_sq = recon_conf.squeeze(-1).square()
        recon_last_step = recon_diff.square().sum(dim=-1)
        horocycle_scale = 2.0 / (recon_conf_sq * delta_tau_recon)
        last_step_diffusion = horocycle_scale * recon_residual.square().sum(dim=-1)
        return {
            "reconstruction": recon,
            "discrete_reconstruction": discrete_recon,
            "discrete_initial": discrete_initial,
            "recon_conf_sq": recon_conf_sq,
            "recon_last_step": recon_last_step,
            "horocycle_scale": horocycle_scale,
            "last_step_diffusion": last_step_diffusion,
        }

    def _compute_losses(self, tokens: torch.Tensor):
        tokens = tokens.to(dtype=torch.long, device=self.device)
        y, discrete_tokens = self._prepare_targets(tokens)
        batch_size = tokens.shape[0]
        schedule, states = self._sample_training_states(y)

        diffusion_losses = self._compute_diffusion_losses(
            states, schedule, y, discrete_tokens
        )
        reconstruction_losses = self._compute_reconstruction_losses(states, schedule, y)
        prior = self.bridge.prior_term(batch_size, self.device)
        nelbo = (
            prior
            + diffusion_losses["diffusion"]
            + reconstruction_losses["reconstruction"]
        )
        discrete_nelbo = (
            prior
            + diffusion_losses["discrete_diffusion"]
            + diffusion_losses["discrete_refinement"]
            + reconstruction_losses["discrete_reconstruction"]
            + reconstruction_losses["discrete_initial"]
        )

        return {
            **diffusion_losses,
            "prior": prior,
            "nelbo": nelbo,
            "dnelbo": discrete_nelbo,
            "discrete_nelbo": discrete_nelbo,
            **reconstruction_losses,
        }

    @torch.no_grad()
    def generate_samples(self, num_samples: int):
        schedule = self._inference_schedule()
        z = self.euler_scheduler.generate(
            batch_size=num_samples,
            schedule=schedule,
            device=self.device,
            dtype=torch.float64,
        )
        final_boundary = self.bridge.normalize_endpoint(z.to(dtype=torch.float32))
        return {
            "boundary": final_boundary,
            "tokens": self.bridge.project_to_discrete(final_boundary),
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        train_loss = losses[self.config.loss].mean()
        train_nelbo = losses["nelbo"].mean().detach()
        train_discrete_nelbo = losses["discrete_nelbo"].mean().detach()
        train_prior = losses["prior"].mean().detach()
        train_diffusion = losses["diffusion"].mean().detach()
        train_discrete_diffusion = losses["discrete_diffusion"].mean().detach()
        train_discrete_refinement = losses["discrete_refinement"].mean().detach()
        train_reconstruction = losses["reconstruction"].mean().detach()
        train_discrete_reconstruction = (
            losses["discrete_reconstruction"].mean().detach()
        )
        train_discrete_initial = losses["discrete_initial"].mean().detach()
        train_recon_conf_sq = losses["recon_conf_sq"].mean().detach()
        train_recon_last_step = losses["recon_last_step"].mean().detach()
        train_horocycle_scale = losses["horocycle_scale"].mean().detach()
        train_last_step_diffusion = losses["last_step_diffusion"].mean().detach()
        batch_size = batch.shape[0]

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_nelbo",
            train_nelbo,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_discrete_nelbo",
            train_discrete_nelbo,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_prior",
            train_prior,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_diffusion",
            train_diffusion,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_discrete_diffusion",
            train_discrete_diffusion,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_discrete_refinement",
            train_discrete_refinement,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_reconstruction",
            train_reconstruction,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_discrete_reconstruction",
            train_discrete_reconstruction,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_discrete_initial",
            train_discrete_initial,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_recon_conf_sq",
            train_recon_conf_sq,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_recon_last_step",
            train_recon_last_step,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_horocycle_scale",
            train_horocycle_scale,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_last_step_diffusion",
            train_last_step_diffusion,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "entropy",
            self.entropy_nats,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )

        if self.global_step % self.config.history_every == 0:
            self.history["train_step"].append(int(self.global_step))
            self.history["train_loss"].append(float(train_loss.detach().cpu()))
            self.history["train_nelbo"].append(float(train_nelbo.cpu()))
            self.history["train_discrete_nelbo"].append(
                float(train_discrete_nelbo.cpu())
            )
            self.history["train_prior"].append(float(train_prior.cpu()))
            self.history["train_diffusion"].append(float(train_diffusion.cpu()))
            self.history["train_discrete_diffusion"].append(
                float(train_discrete_diffusion.cpu())
            )
            self.history["train_discrete_refinement"].append(
                float(train_discrete_refinement.cpu())
            )
            self.history["train_reconstruction"].append(
                float(train_reconstruction.cpu())
            )
            self.history["train_discrete_reconstruction"].append(
                float(train_discrete_reconstruction.cpu())
            )
            self.history["train_discrete_initial"].append(
                float(train_discrete_initial.cpu())
            )
            self.history["train_recon_conf_sq"].append(float(train_recon_conf_sq.cpu()))
            self.history["train_recon_last_step"].append(
                float(train_recon_last_step.cpu())
            )
            self.history["train_horocycle_scale"].append(
                float(train_horocycle_scale.cpu())
            )
            self.history["train_last_step_diffusion"].append(
                float(train_last_step_diffusion.cpu())
            )
        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        val_loss = losses[self.config.loss].mean()
        val_nelbo = losses["nelbo"].mean()
        val_discrete_nelbo = losses["discrete_nelbo"].mean()
        val_prior = losses["prior"].mean()
        val_diffusion = losses["diffusion"].mean()
        val_discrete_diffusion = losses["discrete_diffusion"].mean()
        val_discrete_refinement = losses["discrete_refinement"].mean()
        val_reconstruction = losses["reconstruction"].mean()
        val_discrete_reconstruction = losses["discrete_reconstruction"].mean()
        val_discrete_initial = losses["discrete_initial"].mean()
        val_recon_conf_sq = losses["recon_conf_sq"].mean()
        val_recon_last_step = losses["recon_last_step"].mean()
        val_horocycle_scale = losses["horocycle_scale"].mean()
        val_last_step_diffusion = losses["last_step_diffusion"].mean()
        batch_size = batch.shape[0]

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val_nelbo",
            val_nelbo,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val_discrete_nelbo",
            val_discrete_nelbo,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_prior",
            val_prior,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_diffusion",
            val_diffusion,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_discrete_diffusion",
            val_discrete_diffusion,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_discrete_refinement",
            val_discrete_refinement,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_reconstruction",
            val_reconstruction,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_discrete_reconstruction",
            val_discrete_reconstruction,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_discrete_initial",
            val_discrete_initial,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_recon_conf_sq",
            val_recon_conf_sq,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_recon_last_step",
            val_recon_last_step,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_horocycle_scale",
            val_horocycle_scale,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_last_step_diffusion",
            val_last_step_diffusion,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return val_loss

    def on_test_epoch_start(self):
        self._test_label_counts = torch.zeros(self._vocab_size(), dtype=torch.float32)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        self._test_label_counts += self._token_counts(batch)

    def on_test_epoch_end(self):
        if self._test_label_counts is None:
            return
        label_count = int(self._test_label_counts.sum().item())
        batch_size = max(label_count, self.config.eval_num_samples)
        label_dist = self._test_label_counts / self._test_label_counts.sum().clamp(
            min=1e-8
        )
        metrics = self.evaluator.evaluate(
            model=self.model,
            bridge=self.bridge,
            scheduler=self.euler_scheduler,
            schedule=self._inference_schedule(),
            batch_size=batch_size,
            label_dist=label_dist.to(device=self.device),
        )
        self.log(
            "test_label_entropy",
            metrics["label_entropy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "test_pred_entropy",
            metrics["pred_entropy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "test_unigram_crossentropy",
            metrics["unigram_crossentropy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self._test_label_counts = None

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        required = {
            "val_loss",
            "val_nelbo",
            "val_discrete_nelbo",
            "val_prior",
            "val_diffusion",
            "val_discrete_diffusion",
            "val_discrete_refinement",
            "val_reconstruction",
            "val_discrete_reconstruction",
            "val_discrete_initial",
            "val_recon_conf_sq",
            "val_recon_last_step",
            "val_horocycle_scale",
            "val_last_step_diffusion",
        }
        if not required.issubset(metrics):
            return

        self.history["val_step"].append(int(self.global_step))
        self.history["val_loss"].append(float(metrics["val_loss"].detach().cpu()))
        self.history["val_nelbo"].append(float(metrics["val_nelbo"].detach().cpu()))
        self.history["val_discrete_nelbo"].append(
            float(metrics["val_discrete_nelbo"].detach().cpu())
        )
        self.history["val_prior"].append(float(metrics["val_prior"].detach().cpu()))
        self.history["val_diffusion"].append(
            float(metrics["val_diffusion"].detach().cpu())
        )
        self.history["val_discrete_diffusion"].append(
            float(metrics["val_discrete_diffusion"].detach().cpu())
        )
        self.history["val_discrete_refinement"].append(
            float(metrics["val_discrete_refinement"].detach().cpu())
        )
        self.history["val_reconstruction"].append(
            float(metrics["val_reconstruction"].detach().cpu())
        )
        self.history["val_discrete_reconstruction"].append(
            float(metrics["val_discrete_reconstruction"].detach().cpu())
        )
        self.history["val_discrete_initial"].append(
            float(metrics["val_discrete_initial"].detach().cpu())
        )
        self.history["val_recon_conf_sq"].append(
            float(metrics["val_recon_conf_sq"].detach().cpu())
        )
        self.history["val_recon_last_step"].append(
            float(metrics["val_recon_last_step"].detach().cpu())
        )
        self.history["val_horocycle_scale"].append(
            float(metrics["val_horocycle_scale"].detach().cpu())
        )
        self.history["val_last_step_diffusion"].append(
            float(metrics["val_last_step_diffusion"].detach().cpu())
        )


def plot_history(model: UnigramHyperbolicDLM, output_path: Path):
    if not model.history["train_step"] and not model.history["val_step"]:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)

    loss_ax = axes[0, 0]
    nelbo_ax = axes[0, 1]
    discrete_nelbo_ax = axes[0, 2]
    diff_ax = axes[1, 0]
    discrete_diff_ax = axes[1, 1]
    recon_ax = axes[1, 2]
    discrete_recon_ax = axes[2, 0]
    geom_ax = axes[2, 1]
    scale_ax = axes[2, 2]
    last_step_ax = axes[3, 0]
    empty_ax_1 = axes[3, 1]
    empty_ax_2 = axes[3, 2]

    if model.history["train_step"]:
        loss_ax.plot(
            model.history["train_step"],
            model.history["train_loss"],
            label="Train Loss",
            color="tab:orange",
            linewidth=1.8,
        )
        nelbo_ax.plot(
            model.history["train_step"],
            model.history["train_nelbo"],
            label="Train NELBO",
            color="tab:purple",
            linewidth=1.8,
        )
        discrete_nelbo_ax.plot(
            model.history["train_step"],
            model.history["train_discrete_nelbo"],
            label="Train Discrete NELBO",
            color="tab:purple",
            linewidth=1.8,
        )
        nelbo_ax.plot(
            model.history["train_step"],
            model.history["train_prior"],
            label="Train Prior",
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )

    if model.history["train_step"]:
        diff_ax.plot(
            model.history["train_step"],
            model.history["train_diffusion"],
            label="Train Diffusion",
            color="tab:blue",
            linewidth=1.8,
        )
        discrete_diff_ax.plot(
            model.history["train_step"],
            model.history["train_discrete_diffusion"],
            label="Train Discrete Denoising",
            color="tab:blue",
            linewidth=1.8,
        )
        recon_ax.plot(
            model.history["train_step"],
            model.history["train_reconstruction"],
            label="Train Reconstruction",
            color="tab:orange",
            linewidth=1.8,
        )
        discrete_recon_ax.plot(
            model.history["train_step"],
            model.history["train_discrete_reconstruction"],
            label="Train Discrete Reconstruction",
            color="tab:orange",
            linewidth=1.8,
        )
        empty_ax_1.plot(
            model.history["train_step"],
            model.history["train_discrete_refinement"],
            label="Train Discrete Refinement",
            color="tab:blue",
            linewidth=1.8,
        )
        empty_ax_2.plot(
            model.history["train_step"],
            model.history["train_discrete_initial"],
            label="Train Discrete Initial",
            color="tab:orange",
            linewidth=1.8,
        )

    if model.history["val_step"]:
        loss_ax.plot(
            model.history["val_step"],
            model.history["val_loss"],
            label="Val Loss",
            color="tab:green",
            linewidth=1.8,
        )
        nelbo_ax.plot(
            model.history["val_step"],
            model.history["val_nelbo"],
            label="Val NELBO",
            color="tab:red",
            linewidth=1.8,
        )
        discrete_nelbo_ax.plot(
            model.history["val_step"],
            model.history["val_discrete_nelbo"],
            label="Val Discrete NELBO",
            color="tab:red",
            linewidth=1.8,
        )
        nelbo_ax.plot(
            model.history["val_step"],
            model.history["val_prior"],
            label="Val Prior",
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
        )
        diff_ax.plot(
            model.history["val_step"],
            model.history["val_diffusion"],
            label="Val Diffusion",
            color="tab:green",
            linewidth=1.8,
        )
        discrete_diff_ax.plot(
            model.history["val_step"],
            model.history["val_discrete_diffusion"],
            label="Val Discrete Denoising",
            color="tab:green",
            linewidth=1.8,
        )
        recon_ax.plot(
            model.history["val_step"],
            model.history["val_reconstruction"],
            label="Val Reconstruction",
            color="tab:red",
            linewidth=1.8,
        )
        discrete_recon_ax.plot(
            model.history["val_step"],
            model.history["val_discrete_reconstruction"],
            label="Val Discrete Reconstruction",
            color="tab:red",
            linewidth=1.8,
        )
        empty_ax_1.plot(
            model.history["val_step"],
            model.history["val_discrete_refinement"],
            label="Val Discrete Refinement",
            color="tab:green",
            linewidth=1.8,
        )
        empty_ax_2.plot(
            model.history["val_step"],
            model.history["val_discrete_initial"],
            label="Val Discrete Initial",
            color="tab:red",
            linewidth=1.8,
        )
        scale_ax.plot(
            model.history["val_step"],
            model.history["val_horocycle_scale"],
            label="Val Horocycle Scale",
            color="tab:red",
            linewidth=1.8,
        )
        last_step_ax.plot(
            model.history["val_step"],
            model.history["val_last_step_diffusion"],
            label="Val Last-Step Diffusion",
            color="tab:red",
            linewidth=1.8,
        )

    if model.history["train_step"]:
        geom_ax.plot(
            model.history["train_step"],
            model.history["train_recon_conf_sq"],
            label=r"Train $(1-\|x_{\tau(1)}\|^2)^2$",
            color="tab:blue",
            linewidth=1.8,
        )
        geom_ax.plot(
            model.history["train_step"],
            model.history["train_recon_last_step"],
            label=r"Train $\|\hat{y}_\theta(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2$",
            color="tab:orange",
            linewidth=1.8,
        )
        scale_ax.plot(
            model.history["train_step"],
            model.history["train_horocycle_scale"],
            label="Train Horocycle Scale",
            color="tab:purple",
            linewidth=1.8,
        )
        last_step_ax.plot(
            model.history["train_step"],
            model.history["train_last_step_diffusion"],
            label="Train Last-Step Diffusion",
            color="tab:purple",
            linewidth=1.8,
        )

    nelbo_ax.axhline(
        model.entropy_nats,
        color="tab:brown",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        label="Train Entropy",
    )
    discrete_nelbo_ax.axhline(
        model.entropy_nats,
        color="tab:brown",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        label="Train Entropy",
    )
    nelbo_ax.axhline(
        model.entropy_nats,
        color="tab:pink",
        linestyle=":",
        linewidth=1.4,
        alpha=0.9,
        label="Val Entropy",
    )
    discrete_nelbo_ax.axhline(
        model.entropy_nats,
        color="tab:pink",
        linestyle=":",
        linewidth=1.4,
        alpha=0.9,
        label="Val Entropy",
    )

    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training and Validation Loss")
    loss_ax.grid(alpha=0.2)
    loss_ax.legend()

    nelbo_ax.set_ylabel("Nats")
    nelbo_ax.set_title("Full NELBO = Prior + Diffusion + Reconstruction")
    nelbo_ax.grid(alpha=0.2)
    nelbo_ax.legend()

    discrete_nelbo_ax.set_ylabel("Nats")
    discrete_nelbo_ax.set_title(
        "Full Discrete NELBO = Denoising + Refinement + Reconstruction + Initial"
    )
    discrete_nelbo_ax.grid(alpha=0.2)
    discrete_nelbo_ax.legend()

    diff_ax.set_ylabel("Nats")
    diff_ax.set_title("NELBO Diffusion Term")
    diff_ax.grid(alpha=0.2)
    diff_ax.legend()

    discrete_diff_ax.set_ylabel("Nats")
    discrete_diff_ax.set_title(r"Discrete Denoising $L_{Discrete}$")
    discrete_diff_ax.grid(alpha=0.2)
    discrete_diff_ax.legend()

    recon_ax.set_ylabel("Nats")
    recon_ax.set_title("NELBO Reconstruction Term")
    recon_ax.grid(alpha=0.2)
    recon_ax.legend()

    discrete_recon_ax.set_ylabel("Nats")
    discrete_recon_ax.set_title("Discrete NELBO Reconstruction Term")
    discrete_recon_ax.grid(alpha=0.2)
    discrete_recon_ax.legend()

    geom_ax.set_ylabel("Value")
    geom_ax.set_title(r"Reconstruction Geometry at $x_{\tau(1)}$")
    geom_ax.set_yscale("log")
    geom_ax.grid(alpha=0.2)
    geom_ax.legend()

    scale_ax.set_ylabel("Value")
    scale_ax.set_title(
        r"Horocycle Scale $\frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta \tau_1}$"
    )
    scale_ax.grid(alpha=0.2)
    scale_ax.legend()

    last_step_ax.set_xlabel("Optimization step")
    last_step_ax.set_ylabel("Value")
    last_step_ax.set_title(
        r"Last-Step Diffusion $\frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta \tau_1}\|\mathrm{residual}\|^2$"
    )
    last_step_ax.grid(alpha=0.2)
    last_step_ax.legend()

    discrete_recon_ax.set_xlabel("Optimization step")
    geom_ax.set_xlabel("Optimization step")
    scale_ax.set_xlabel("Optimization step")
    empty_ax_1.set_xlabel("Optimization step")
    empty_ax_1.set_ylabel("Nats")
    empty_ax_1.set_title(r"Continuous Refinement $L_{Refinement}$")
    empty_ax_1.grid(alpha=0.2)
    empty_ax_1.legend()
    empty_ax_2.set_xlabel("Optimization step")
    empty_ax_2.set_ylabel("Nats")
    empty_ax_2.set_title(r"Initial $L_{Initial}$")
    empty_ax_2.grid(alpha=0.2)
    empty_ax_2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss",
        choices=["ce", "l2", "diffusion", "nelbo", "dnelbo"],
        default="nelbo",
    )
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hyper-dim", type=int, default=2)
    parser.add_argument("--hyper-T", type=int, default=1000)
    parser.add_argument("--hyper-dt", type=float, default=0.01)
    parser.add_argument(
        "--proposal",
        choices=["fixed_interval", "uniform", "exponential"],
        default="fixed_interval",
    )
    parser.add_argument("--proposal-start", type=float, default=None)
    parser.add_argument("--proposal-end", type=float, default=None)
    parser.add_argument("--proposal-exp-lambda", type=float, default=1.0)
    parser.add_argument("--inference-dt", type=float, default=None)
    parser.add_argument("--discrete-temperature", type=float, default=1.0)
    parser.add_argument(
        "--discrete-approximation",
        choices=["diag_probit", "gauss_hermite"],
        default="diag_probit",
    )
    parser.add_argument("--discrete-gauss-hermite-order", type=int, default=32)
    parser.add_argument("--discrete-train-mc-samples", type=int, default=16)
    parser.add_argument("--discrete-eval-mc-samples", type=int, default=128)
    parser.add_argument("--p-a", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--bridge-plot-path", type=str, default="unigram_bridge.jpg")
    parser.add_argument("--bridge-plot-num-paths", type=int, default=4)
    parser.add_argument("--plot-path", type=str, default="unigram_test.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        loss=args.loss,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        hidden_size=args.hidden_size,
        depth=args.depth,
        hyper_dim=args.hyper_dim,
        hyper_T=args.hyper_T,
        hyper_dt=args.hyper_dt,
        # hyper_embed_length=args.hyper_embed_length,
        proposal=args.proposal,
        proposal_start=args.proposal_start,
        proposal_end=args.proposal_end,
        proposal_exp_lambda=args.proposal_exp_lambda,
        inference_dt=args.inference_dt,
        discrete_temperature=args.discrete_temperature,
        discrete_approximation=args.discrete_approximation,
        discrete_gauss_hermite_order=args.discrete_gauss_hermite_order,
        discrete_train_mc_samples=args.discrete_train_mc_samples,
        discrete_eval_mc_samples=args.discrete_eval_mc_samples,
        p_a=args.p_a,
        seed=args.seed,
        num_workers=args.num_workers,
        history_every=args.history_every,
        plot_path=args.plot_path,
    )

    L.seed_everything(config.seed, workers=True)

    datamodule = UnigramDataModule(config)
    model = UnigramHyperbolicDLM(config)
    train_schedule = model.proposal_generator.generate()

    print("Running hyperbolic unigram test")
    print(f"  Loss: {config.loss}")
    print(f"  Dataset entropy: {model.entropy_nats:.6f} nats")
    print(f"  P(A)={config.p_a:.2f}, P(B)={1 - config.p_a:.2f}")
    print(f"  hyper_T={config.hyper_T}, base hyper_dt={config.hyper_dt}")
    print(f"  hyper_embed_length={config.hyper_embed_length:.4f}")
    print(f"  max_ball_norm={config.max_ball_norm:.4f}")
    print(
        f"  proposal={config.proposal}, "
        f"tau(1)={train_schedule.tau[0].item():.4f}, "
        f"tau(T)={train_schedule.tau[-1].item():.4f}"
    )
    print(f"  inference_dt={config.inference_dt}")
    print(f"  discrete_temperature={config.discrete_temperature:.4f}")
    print(f"  discrete_approximation={config.discrete_approximation}")
    print(
        f"  discrete_mc_samples(train={config.discrete_train_mc_samples}, "
        f"eval={config.discrete_eval_mc_samples})"
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=config.max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=False)
    if test_metrics:
        print("Test metrics:")
        for name, value in test_metrics[0].items():
            print(f"  {name}: {value:.6f}")

    plot_path = Path(config.plot_path)
    plot_history(model, plot_path)
    print(f"Saved plot -> {plot_path}")

    if args.bridge_plot_path is not None:
        visualizer = Visualizer()
        visualizer.plot_poincare_disk_bridge(
            bridge=model.bridge,
            scheduler=model.euler_scheduler,
            schedule=model._inference_schedule(),
            output_path=args.bridge_plot_path,
            num_paths_per_label=args.bridge_plot_num_paths,
        )
        print(f"Saved bridge plot -> {args.bridge_plot_path}")


if __name__ == "__main__":
    main()
