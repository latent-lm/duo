"""Minimal unigram sanity test for the hyperbolic DLM in ``hyper_dm.md``.

This experiment is intentionally simple:
  - vocabulary size = 2
  - sequence length = 1
  - dataset distribution: P(A)=0.8, P(B)=0.2
  - model: a small MLP

It is designed to verify the inequality chain

    NELBO >= NLL >= H(Y)

in the regime where both KL gaps should shrink.  The script trains with one
of four objectives:
  - cross entropy surrogate
  - L2 surrogate
  - diffusion surrogate
  - full NELBO

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
  hyper_embed_length: float = 0.99
  p_a: float = 0.8
  seed: int = 0
  num_workers: int = 0
  history_every: int = 10
  proposal: str = "fixed_interval"
  proposal_start: float | None = None
  proposal_end: float | None = None
  proposal_exp_lambda: float = 1.0
  inference_dt: float = 0.01
  plot_path: str = "unigram_test.png"


@dataclass
class ProposalSchedule:
  tau: torch.Tensor
  delta_tau: torch.Tensor


class ProposalGenerator:
  """Build the reverse-time schedule used by training and inference."""

  def __init__(self, num_steps: int, mode: str,
               start: float, end: float, exp_lambda: float = 1.0):
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
    start = config.hyper_dt if config.proposal_start is None else config.proposal_start
    end = (config.hyper_T * config.hyper_dt
           if config.proposal_end is None else config.proposal_end)
    return cls(
      num_steps=config.hyper_T,
      mode=config.proposal,
      start=start,
      end=end,
      exp_lambda=config.proposal_exp_lambda)

  @classmethod
  def fixed_step(cls, num_steps: int, delta_tau: float):
    return cls(
      num_steps=num_steps,
      mode="fixed_interval",
      start=delta_tau,
      end=num_steps * delta_tau)

  def _quantile_grid(self, device: torch.device, dtype: torch.dtype):
    return (torch.arange(1, self.num_steps + 1, device=device, dtype=dtype)
            / self.num_steps)

  def _generate_tau(self, device: torch.device, dtype: torch.dtype):
    if self.mode == "fixed_interval":
      tau = torch.linspace(
        self.start, self.end, steps=self.num_steps,
        device=device, dtype=dtype)
    elif self.mode == "uniform":
      quantiles = self._quantile_grid(device, dtype)
      tau = self.start + (self.end - self.start) * quantiles
    else:
      quantiles = self._quantile_grid(device, dtype).clamp(max=1 - 1e-6)
      tau = -torch.log1p(-quantiles) / self.exp_lambda
    return tau.sort().values

  def generate(self, device: torch.device | None = None,
               dtype: torch.dtype = torch.float32):
    if device is None:
      device = torch.device("cpu")
    tau = self._generate_tau(device, dtype)
    tau0 = torch.cat([tau.new_zeros(1), tau], dim=0)
    delta_tau = tau0[1:] - tau0[:-1]
    return ProposalSchedule(tau=tau, delta_tau=delta_tau)


class UnigramDataset(Dataset):
  """Exact unigram dataset with sequence length 1."""

  def __init__(self, size: int, p_a: float, seed: int):
    super().__init__()
    n_a = int(round(size * p_a))
    n_b = size - n_a
    tokens = torch.cat([
      torch.zeros(n_a, dtype=torch.long),
      torch.ones(n_b, dtype=torch.long),
    ])
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(tokens.numel(), generator=generator)
    self.tokens = tokens[permutation]

  def __len__(self):
    return self.tokens.numel()

  def __getitem__(self, index: int):
    return self.tokens[index]


class UnigramDataModule(L.LightningDataModule):
  def __init__(self, config: ExperimentConfig):
    super().__init__()
    self.config = config
    self.train_dataset = None
    self.val_dataset = None

  def setup(self, stage: str | None = None):
    del stage
    self.train_dataset = UnigramDataset(
      size=self.config.train_size,
      p_a=self.config.p_a,
      seed=self.config.seed)
    self.val_dataset = UnigramDataset(
      size=self.config.val_size,
      p_a=self.config.p_a,
      seed=self.config.seed + 1)

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.config.batch_size,
      shuffle=True,
      num_workers=self.config.num_workers)

  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.config.batch_size,
      shuffle=False,
      num_workers=self.config.num_workers)


class SmallMLP(nn.Module):
  """Tiny time-conditioned MLP that predicts a hyperbolic endpoint."""

  def __init__(self, input_dim: int, hidden_size: int,
               depth: int, output_dim: int):
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


class HyperbolicBridge(nn.Module):
  """Hyperbolic bridge utilities on the Poincare disk."""

  def __init__(self, hyper_dim: int, hyper_embed_length: float=1.0):
    super().__init__()
    assert hyper_dim >= 2
    self.hyper_dim = hyper_dim
    self.hyper_embed_length = hyper_embed_length

    endpoints = torch.zeros(2, hyper_dim, dtype=torch.float32)
    endpoints[0, 0] = self.hyper_embed_length
    endpoints[1, 0] = - self.hyper_embed_length
    self.register_buffer("endpoints", endpoints, persistent=False)

  def normalize_endpoint(self, endpoint: torch.Tensor):
    return endpoint / endpoint.norm(dim=-1, keepdim=True).clamp(min=1e-8) * self.hyper_embed_length

  def endpoint_posterior(self, endpoint: torch.Tensor):
    endpoint = self.normalize_endpoint(endpoint)
    return (endpoint @ self.endpoints.t()).softmax(dim=-1)

  def project_to_discrete(self, endpoint: torch.Tensor):
    return (endpoint @ self.endpoints.t()).argmax(dim=-1)

  def clamp_to_ball(self, z: torch.Tensor):
    znorm = z.norm(dim=-1, keepdim=True).clamp(min=1e-32)
    return z * (znorm.clamp(max=self.hyper_embed_length) / znorm)

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

  def bridge_step(self, z: torch.Tensor, y: torch.Tensor,
                  delta_tau: torch.Tensor | float):
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

  def diffusion_term(self, z: torch.Tensor, y: torch.Tensor,
                     pred: torch.Tensor, delta_tau: torch.Tensor | float):
    pred_n = self.normalize_endpoint(pred)
    diff_y = y - z
    diff_pred = pred_n - z
    dir_y = diff_y / diff_y.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    dir_pred = (diff_pred
                / diff_pred.square().sum(dim=-1, keepdim=True).clamp(min=1e-8))
    conf = (1 - z.square().sum(dim=-1)).clamp(min=1e-8)
    error = (dir_y - dir_pred).square().sum(dim=-1)
    delta_tau = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype)
    return (0.5 * delta_tau * (self.hyper_dim - 1) ** 2
            * conf.square() * error)

  def reconstruction_term(self, z: torch.Tensor, y: torch.Tensor,
                          pred: torch.Tensor,
                          delta_tau: torch.Tensor | float):
    pred_n = self.normalize_endpoint(pred)
    conf = (1 - z.square().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    diff_pred = pred_n - z
    sq_diff = diff_pred.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    drift = (((self.hyper_dim - 1) / 2) * conf.square() / sq_diff * diff_pred
             - (self.hyper_dim / 4) * conf * z)
    delta_tau_state = self._expand_like_state(delta_tau, z)
    residual = y - z - drift * delta_tau_state

    log_dt = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype).log()
    log_2pi = z.new_tensor(2 * math.pi).log()
    normalizer = (
      0.5 * self.hyper_dim * log_2pi
      + self.hyper_dim * (conf / 2).log().squeeze(-1)
      + 0.5 * self.hyper_dim * log_dt)
    quad = (2.0 / (conf.square() * delta_tau_state)
            * residual.square().sum(dim=-1, keepdim=True))
    return normalizer + quad.squeeze(-1)

  def prior_term(self, batch_size: int, device: torch.device):
    # The discrete derivation fixes q(x_{tau(T)} | y) = p(x_{tau(T)}) = delta_0.
    return torch.zeros(batch_size, device=device, dtype=torch.float32)


class UnigramHyperbolicDLM(L.LightningModule):
  def __init__(self, config: ExperimentConfig):
    super().__init__()
    self.save_hyperparameters(asdict(config))
    self.config = config
    self.bridge = HyperbolicBridge(hyper_dim=config.hyper_dim, hyper_embed_length=config.hyper_embed_length)
    self.proposal_generator = ProposalGenerator.from_config(config)
    self.inference_generator = ProposalGenerator.fixed_step(
      num_steps=config.hyper_T,
      delta_tau=config.inference_dt)
    self.model = SmallMLP(
      input_dim=config.hyper_dim + 1,
      hidden_size=config.hidden_size,
      depth=config.depth,
      output_dim=config.hyper_dim)
    self.entropy_nats = -(
      config.p_a * math.log(config.p_a)
      + (1 - config.p_a) * math.log(1 - config.p_a))
    self.history = {
      "train_step": [],
      "train_loss": [],
      "train_nelbo": [],
      "train_prior": [],
      "train_diffusion": [],
      "train_reconstruction": [],
      "train_recon_conf_sq": [],
      "train_recon_last_step": [],
      "train_horocycle_scale": [],
      "train_last_step_diffusion": [],
      "val_step": [],
      "val_loss": [],
      "val_nelbo": [],
      "val_prior": [],
      "val_diffusion": [],
      "val_reconstruction": [],
      "val_recon_conf_sq": [],
      "val_recon_last_step": [],
      "val_horocycle_scale": [],
      "val_last_step_diffusion": [],
    }

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.config.lr)

  def _time_from_tau(self, tau: torch.Tensor):
    return tau.to(dtype=torch.float32, device=self.device)

  def _reconstruction_residual_terms(self, z: torch.Tensor, y: torch.Tensor,
                                     pred: torch.Tensor,
                                     delta_tau: torch.Tensor | float):
    pred_n = self.bridge.normalize_endpoint(pred)
    conf = (1 - z.square().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    diff = pred_n - z
    sq_diff = diff.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    drift = (((self.config.hyper_dim - 1) / 2)
             * conf.square() / sq_diff * diff
             - (self.config.hyper_dim / 4) * conf * z)
    delta_tau = torch.as_tensor(delta_tau, device=z.device, dtype=z.dtype)
    residual = y - z - drift * delta_tau
    return pred_n, conf, diff, residual

  def _proposal_schedule(self):
    return self.proposal_generator.generate(device=self.device)

  def _inference_schedule(self):
    return self.inference_generator.generate(device=self.device)

  def _predict_boundary(self, z: torch.Tensor, tau: torch.Tensor):
    flat_z = z.reshape(-1, z.shape[-1])
    flat_tau = self._time_from_tau(tau).reshape(-1)
    pred = self.model(flat_z, flat_tau)
    return pred.reshape(*z.shape[:-1], -1)

  def _prepare_targets(self, tokens: torch.Tensor):
    y = self.bridge.normalize_endpoint(self.bridge.endpoints[tokens])
    discrete = self.bridge.project_to_discrete(y)
    return y, discrete

  def _compute_losses(self, tokens: torch.Tensor):
    tokens = tokens.to(dtype=torch.long, device=self.device)
    y, discrete_tokens = self._prepare_targets(tokens)
    batch_size = tokens.shape[0]
    schedule = self._proposal_schedule()
    with torch.no_grad():
      states = self.bridge.sample_reverse_path(y, schedule)

    z_diff = states[:, 2:, :]
    tau_diff = schedule.tau[1:].expand(batch_size, -1)
    y_diff = y[:, None, :].expand_as(z_diff)
    pred_diff = self._predict_boundary(z_diff, tau_diff)
    posterior_diff = self.bridge.endpoint_posterior(pred_diff).clamp(min=1e-8)
    ce = -posterior_diff.log().gather(
      -1,
      discrete_tokens[:, None, None].expand(-1, z_diff.shape[1], 1),
    ).squeeze(-1).sum(dim=-1)
    l2 = (pred_diff - y_diff).square().sum(dim=-1).sum(dim=-1)
    diffusion = self.bridge.diffusion_term(
      z_diff, y_diff, pred_diff, schedule.delta_tau[1:]).sum(dim=-1)

    z_recon = states[:, 1, :]
    tau_recon = schedule.tau[0].expand(batch_size)
    delta_tau_recon = schedule.delta_tau[0]
    pred_recon = self._predict_boundary(z_recon, tau_recon)
    recon = self.bridge.reconstruction_term(
      z_recon, y, pred_recon, delta_tau_recon)
    _, recon_conf, recon_diff, recon_residual = (
      self._reconstruction_residual_terms(
        z_recon, y, pred_recon, delta_tau_recon))
    recon_conf_sq = recon_conf.squeeze(-1).square()
    recon_last_step = recon_diff.square().sum(dim=-1)
    horocycle_scale = 2.0 / (recon_conf_sq * delta_tau_recon)
    last_step_diffusion = horocycle_scale * recon_residual.square().sum(dim=-1)
    prior = self.bridge.prior_term(batch_size, self.device)
    nelbo = prior + diffusion + recon

    return {
      "ce": ce,
      "l2": l2,
      "prior": prior,
      "diffusion": diffusion,
      "nelbo": nelbo,
      "reconstruction": recon,
      "recon_conf_sq": recon_conf_sq,
      "recon_last_step": recon_last_step,
      "horocycle_scale": horocycle_scale,
      "last_step_diffusion": last_step_diffusion,
    }

  @torch.no_grad()
  def generate_samples(self, num_samples: int):
    schedule = self._inference_schedule()
    z = torch.zeros(
      num_samples, self.config.hyper_dim,
      device=self.device, dtype=torch.float64)

    for idx in range(self.config.hyper_T - 1, -1, -1):
      tau_t = schedule.tau[idx].expand(num_samples)
      pred = self._predict_boundary(z.float(), tau_t)
      boundary = self.bridge.normalize_endpoint(pred).to(torch.float64)
      z = self.bridge.bridge_step(z, boundary, schedule.delta_tau[idx])

    final_boundary = self.bridge.normalize_endpoint(z.float())
    return {
      "boundary": final_boundary,
      "tokens": self.bridge.project_to_discrete(final_boundary),
    }

  def training_step(self, batch: torch.Tensor, batch_idx: int):
    del batch_idx
    losses = self._compute_losses(batch)
    train_loss = losses[self.config.loss].mean()
    train_nelbo = losses["nelbo"].mean().detach()
    train_prior = losses["prior"].mean().detach()
    train_diffusion = losses["diffusion"].mean().detach()
    train_reconstruction = losses["reconstruction"].mean().detach()
    train_recon_conf_sq = losses["recon_conf_sq"].mean().detach()
    train_recon_last_step = losses["recon_last_step"].mean().detach()
    train_horocycle_scale = losses["horocycle_scale"].mean().detach()
    train_last_step_diffusion = losses["last_step_diffusion"].mean().detach()
    batch_size = batch.shape[0]

    self.log("train_loss", train_loss, on_step=True, on_epoch=True,
             prog_bar=True, batch_size=batch_size)
    self.log("train_nelbo", train_nelbo, on_step=True, on_epoch=True,
             prog_bar=True, batch_size=batch_size)
    self.log("train_prior", train_prior, on_step=True, on_epoch=True,
             prog_bar=False, batch_size=batch_size)
    self.log("train_diffusion", train_diffusion, on_step=True, on_epoch=True,
             prog_bar=False, batch_size=batch_size)
    self.log("train_reconstruction", train_reconstruction,
             on_step=True, on_epoch=True, prog_bar=False,
             batch_size=batch_size)
    self.log("train_recon_conf_sq", train_recon_conf_sq,
             on_step=True, on_epoch=True, prog_bar=False,
             batch_size=batch_size)
    self.log("train_recon_last_step", train_recon_last_step,
             on_step=True, on_epoch=True, prog_bar=False,
             batch_size=batch_size)
    self.log("train_horocycle_scale", train_horocycle_scale,
             on_step=True, on_epoch=True, prog_bar=False,
             batch_size=batch_size)
    self.log("train_last_step_diffusion", train_last_step_diffusion,
             on_step=True, on_epoch=True, prog_bar=False,
             batch_size=batch_size)
    self.log("entropy", self.entropy_nats, on_step=True, on_epoch=False,
             prog_bar=True, batch_size=batch_size)

    if self.global_step % self.config.history_every == 0:
      self.history["train_step"].append(int(self.global_step))
      self.history["train_loss"].append(float(train_loss.detach().cpu()))
      self.history["train_nelbo"].append(float(train_nelbo.cpu()))
      self.history["train_prior"].append(float(train_prior.cpu()))
      self.history["train_diffusion"].append(float(train_diffusion.cpu()))
      self.history["train_reconstruction"].append(
        float(train_reconstruction.cpu()))
      self.history["train_recon_conf_sq"].append(
        float(train_recon_conf_sq.cpu()))
      self.history["train_recon_last_step"].append(
        float(train_recon_last_step.cpu()))
      self.history["train_horocycle_scale"].append(
        float(train_horocycle_scale.cpu()))
      self.history["train_last_step_diffusion"].append(
        float(train_last_step_diffusion.cpu()))
    return train_loss

  def validation_step(self, batch: torch.Tensor, batch_idx: int):
    del batch_idx
    losses = self._compute_losses(batch)
    val_loss = losses[self.config.loss].mean()
    val_nelbo = losses["nelbo"].mean()
    val_prior = losses["prior"].mean()
    val_diffusion = losses["diffusion"].mean()
    val_reconstruction = losses["reconstruction"].mean()
    val_recon_conf_sq = losses["recon_conf_sq"].mean()
    val_recon_last_step = losses["recon_last_step"].mean()
    val_horocycle_scale = losses["horocycle_scale"].mean()
    val_last_step_diffusion = losses["last_step_diffusion"].mean()
    batch_size = batch.shape[0]

    self.log("val_loss", val_loss, on_step=False, on_epoch=True,
             prog_bar=True, batch_size=batch_size)
    self.log("val_nelbo", val_nelbo, on_step=False, on_epoch=True,
             prog_bar=True, batch_size=batch_size)
    self.log("val_prior", val_prior, on_step=False, on_epoch=True,
             prog_bar=False, batch_size=batch_size)
    self.log("val_diffusion", val_diffusion, on_step=False, on_epoch=True,
             prog_bar=False, batch_size=batch_size)
    self.log("val_reconstruction", val_reconstruction, on_step=False,
             on_epoch=True, prog_bar=False, batch_size=batch_size)
    self.log("val_recon_conf_sq", val_recon_conf_sq, on_step=False,
             on_epoch=True, prog_bar=False, batch_size=batch_size)
    self.log("val_recon_last_step", val_recon_last_step, on_step=False,
             on_epoch=True, prog_bar=False, batch_size=batch_size)
    self.log("val_horocycle_scale", val_horocycle_scale, on_step=False,
             on_epoch=True, prog_bar=False, batch_size=batch_size)
    self.log("val_last_step_diffusion", val_last_step_diffusion,
             on_step=False, on_epoch=True, prog_bar=False,
             batch_size=batch_size)
    return val_loss

  def on_validation_epoch_end(self):
    metrics = self.trainer.callback_metrics
    required = {
      "val_loss",
      "val_nelbo",
      "val_prior",
      "val_diffusion",
      "val_reconstruction",
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
    self.history["val_prior"].append(
      float(metrics["val_prior"].detach().cpu()))
    self.history["val_diffusion"].append(
      float(metrics["val_diffusion"].detach().cpu()))
    self.history["val_reconstruction"].append(
      float(metrics["val_reconstruction"].detach().cpu()))
    self.history["val_recon_conf_sq"].append(
      float(metrics["val_recon_conf_sq"].detach().cpu()))
    self.history["val_recon_last_step"].append(
      float(metrics["val_recon_last_step"].detach().cpu()))
    self.history["val_horocycle_scale"].append(
      float(metrics["val_horocycle_scale"].detach().cpu()))
    self.history["val_last_step_diffusion"].append(
      float(metrics["val_last_step_diffusion"].detach().cpu()))


def plot_history(model: UnigramHyperbolicDLM, output_path: Path):
  if not model.history["train_step"] and not model.history["val_step"]:
    return

  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)

  loss_ax = axes[0, 0]
  nelbo_ax = axes[0, 1]
  diff_ax = axes[1, 0]
  recon_ax = axes[1, 1]
  geom_ax = axes[2, 0]
  scale_ax = axes[2, 1]
  last_step_ax = axes[3, 0]
  empty_ax = axes[3, 1]

  if model.history["train_step"]:
    loss_ax.plot(model.history["train_step"], model.history["train_loss"],
                 label="Train Loss", color="tab:orange", linewidth=1.8)
    nelbo_ax.plot(model.history["train_step"], model.history["train_nelbo"],
                  label="Train NELBO", color="tab:purple", linewidth=1.8)
    nelbo_ax.plot(model.history["train_step"], model.history["train_prior"],
                  label="Train Prior", color="tab:gray",
                  linestyle="--", linewidth=1.2, alpha=0.9)

  if model.history["train_step"]:
    diff_ax.plot(model.history["train_step"], model.history["train_diffusion"],
                 label="Train Diffusion", color="tab:blue", linewidth=1.8)
    recon_ax.plot(
      model.history["train_step"],
      model.history["train_reconstruction"],
      label="Train Reconstruction",
      color="tab:orange",
      linewidth=1.8)

  if model.history["val_step"]:
    loss_ax.plot(model.history["val_step"], model.history["val_loss"],
                 label="Val Loss", color="tab:green", linewidth=1.8)
    nelbo_ax.plot(model.history["val_step"], model.history["val_nelbo"],
                  label="Val NELBO", color="tab:red", linewidth=1.8)
    nelbo_ax.plot(model.history["val_step"], model.history["val_prior"],
                  label="Val Prior", color="black",
                  linestyle=":", linewidth=1.2, alpha=0.9)
    diff_ax.plot(model.history["val_step"], model.history["val_diffusion"],
                 label="Val Diffusion", color="tab:green", linewidth=1.8)
    recon_ax.plot(
      model.history["val_step"],
      model.history["val_reconstruction"],
      label="Val Reconstruction",
      color="tab:red",
      linewidth=1.8)
    scale_ax.plot(
      model.history["val_step"],
      model.history["val_horocycle_scale"],
      label="Val Horocycle Scale",
      color="tab:red",
      linewidth=1.8)
    last_step_ax.plot(
      model.history["val_step"],
      model.history["val_last_step_diffusion"],
      label="Val Last-Step Diffusion",
      color="tab:red",
      linewidth=1.8)

  if model.history["train_step"]:
    geom_ax.plot(
      model.history["train_step"],
      model.history["train_recon_conf_sq"],
      label=r"Train $(1-\|x_{\tau(1)}\|^2)^2$",
      color="tab:blue",
      linewidth=1.8)
    geom_ax.plot(
      model.history["train_step"],
      model.history["train_recon_last_step"],
      label=r"Train $\|\hat{y}_\theta(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2$",
      color="tab:orange",
      linewidth=1.8)
    scale_ax.plot(
      model.history["train_step"],
      model.history["train_horocycle_scale"],
      label="Train Horocycle Scale",
      color="tab:purple",
      linewidth=1.8)
    last_step_ax.plot(
      model.history["train_step"],
      model.history["train_last_step_diffusion"],
      label="Train Last-Step Diffusion",
      color="tab:purple",
      linewidth=1.8)

  nelbo_ax.axhline(
    model.entropy_nats,
    color="tab:brown",
    linestyle="--",
    linewidth=1.4,
    alpha=0.9,
    label="Train Entropy")
  nelbo_ax.axhline(
    model.entropy_nats,
    color="tab:pink",
    linestyle=":",
    linewidth=1.4,
    alpha=0.9,
    label="Val Entropy")

  loss_ax.set_ylabel("Loss")
  loss_ax.set_title("Training and Validation Loss")
  loss_ax.grid(alpha=0.2)
  loss_ax.legend()

  nelbo_ax.set_ylabel("Nats")
  nelbo_ax.set_title("Full NELBO = Prior + Diffusion + Reconstruction")
  nelbo_ax.grid(alpha=0.2)
  nelbo_ax.legend()

  diff_ax.set_ylabel("Nats")
  diff_ax.set_title("NELBO Diffusion Term")
  diff_ax.grid(alpha=0.2)
  diff_ax.legend()

  diff_ax.set_xlabel("Optimization step")
  recon_ax.set_xlabel("Optimization step")
  recon_ax.set_ylabel("Nats")
  recon_ax.set_title("NELBO Reconstruction Term")
  recon_ax.grid(alpha=0.2)
  recon_ax.legend()

  geom_ax.set_xlabel("Optimization step")
  geom_ax.set_ylabel("Value")
  geom_ax.set_title(r"Reconstruction Geometry at $x_{\tau(1)}$")
  geom_ax.set_yscale("log")
  geom_ax.grid(alpha=0.2)
  geom_ax.legend()

  scale_ax.set_xlabel("Optimization step")
  scale_ax.set_ylabel("Value")
  scale_ax.set_title(
    r"Horocycle Scale $\frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta \tau_1}$")
  scale_ax.grid(alpha=0.2)
  scale_ax.legend()

  last_step_ax.set_xlabel("Optimization step")
  last_step_ax.set_ylabel("Value")
  last_step_ax.set_title(
    r"Last-Step Diffusion $\frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta \tau_1}\|\mathrm{residual}\|^2$")
  last_step_ax.grid(alpha=0.2)
  last_step_ax.legend()

  empty_ax.axis("off")

  fig.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--loss", choices=["ce", "l2", "diffusion", "nelbo"],
                      default="nelbo")
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
  parser.add_argument("--proposal",
                      choices=["fixed_interval", "uniform", "exponential"],
                      default="fixed_interval")
  parser.add_argument("--proposal-start", type=float, default=None)
  parser.add_argument("--proposal-end", type=float, default=None)
  parser.add_argument("--proposal-exp-lambda", type=float, default=1.0)
  parser.add_argument("--inference-dt", type=float, default=0.03)
  parser.add_argument("--p-a", type=float, default=0.8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--history-every", type=int, default=1)
  parser.add_argument("--plot-path", type=str, default="unigram_test.png")
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
    proposal=args.proposal,
    proposal_start=args.proposal_start,
    proposal_end=args.proposal_end,
    proposal_exp_lambda=args.proposal_exp_lambda,
    inference_dt=args.inference_dt,
    p_a=args.p_a,
    seed=args.seed,
    num_workers=args.num_workers,
    history_every=args.history_every,
    plot_path=args.plot_path)

  L.seed_everything(config.seed, workers=True)

  datamodule = UnigramDataModule(config)
  model = UnigramHyperbolicDLM(config)
  train_schedule = model.proposal_generator.generate()

  print("Running hyperbolic unigram test")
  print(f"  Loss: {config.loss}")
  print(f"  Dataset entropy: {model.entropy_nats:.6f} nats")
  print(f"  P(A)={config.p_a:.2f}, P(B)={1 - config.p_a:.2f}")
  print(f"  hyper_T={config.hyper_T}, base hyper_dt={config.hyper_dt}")
  print(
    f"  proposal={config.proposal}, "
    f"tau(1)={train_schedule.tau[0].item():.4f}, "
    f"tau(T)={train_schedule.tau[-1].item():.4f}")
  print(f"  inference_dt={config.inference_dt}")

  trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_steps=config.max_steps,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    log_every_n_steps=1,
    num_sanity_val_steps=0)
  trainer.fit(model, datamodule=datamodule)

  plot_path = Path(config.plot_path)
  plot_history(model, plot_path)
  print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
  main()
