import os
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
import torch.nn as nn
import lightning as L

try:
    from unigram.model import MLPLM, OptimalModel, polar_to_cart, vocab_points
    from unigram.dataset import UnigramDataModule, process_ps
    from unigram.visualizer import DataMgr, Recorder
except ModuleNotFoundError:
    from model import MLPLM, OptimalModel, polar_to_cart, vocab_points
    from dataset import UnigramDataModule, process_ps
    from visualizer import DataMgr, Recorder

def isnan_or_inf(x):
    return torch.logical_or(torch.isnan(x), torch.isinf(x))

@dataclass
class LossGeometry:
    POINCARE_POLAR: str = "poincare_polar"
    POINCARE_CARTESIAN: str = "poincare_cartesian"
    LORENTZ_POLAR: str = "lorentz_polar"
    LORENTZ_CARTESIAN: str = "lorentz_cartesian"

class HyperBridge:
    PROPOSAL_EXP_NAME: str = "exp"
    PROPOSAL_STRATIFIED_EXP_NAME: str = "stratified_exp"
    PROPOSAL_TRUNCATED_EXP_NAME: str = "truncated_exp"
    PROPOSAL_UNIF_NAME: str = "unif"

    @staticmethod
    def sample_chi(ns, dtype=torch.float64):
        # chi(n) = sqrt(chi^2(n)), and chi^2(n) ~ Gamma(shape=n/2, scale=2).
        # Sampling Gamma directly avoids allocating sum(ns) standard normals,
        # which blows up when ns is large.
        concentration = ns.to(dtype) / 2
        rate = torch.tensor(0.5, device=ns.device, dtype=dtype)
        chi2 = torch.distributions.Gamma(concentration, rate).sample()
        # print(f"chi2: {isnan_or_inf(chi2).any()}")
        return chi2.sqrt()

    @staticmethod
    def sample_chi_old(ns,dtype=torch.float64):
        nshape = ns.shape
        ns = ns.reshape(-1)
        M = ns.sum().item()
        x = torch.randn(M, device=ns.device, dtype=dtype).square()
        chi2 = torch.segment_reduce(x,'sum',lengths=ns)
        return chi2.sqrt().reshape(nshape)

    @staticmethod
    @torch.no_grad()
    def binary_bridge(ts):
        # print(f"ts: {ts.shape}")
        ns = torch.poisson(ts/8).to(torch.int64)
        # print(f"ns: {ns.shape}")
        ss = ts.sqrt() * HyperBridge.sample_chi(2*ns+3, ts.dtype)
        # print(f"ss: {ss.shape}")
        # print(f"ss: {isnan_or_inf(ss).any()}")
        vs = torch.rand_like(ts)
        # print(f"vs: {isnan_or_inf(vs).any()}")
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
        # print(f"ps: {isnan_or_inf(ps).any()}")
        # print(f"thetas: {isnan_or_inf(thetas).any()}")
        return (ps,thetas)

    @staticmethod
    @torch.no_grad()
    def polar_to_lorentz(rhos, thetas):
        sinh_r = torch.sinh(rhos)
        return torch.stack(
            [torch.cosh(rhos), sinh_r * thetas.cos(), sinh_r * thetas.sin()],
            dim=-1,
        )

    @staticmethod
    @torch.no_grad()
    def binary_bridge_lorentz(ts):
        rhos, thetas = HyperBridge.binary_bridge(ts)
        return HyperBridge.polar_to_lorentz(rhos, thetas)

    # ---- Polar bridge loss ----------------------------------------------
    # logits        (N,V)     float64 (converts)
    # targets       (N,)      int64
    # rhos          (N,)      float64
    # thetas        (N,)      float64
    @staticmethod
    def binary_bridge_loss_poincare_disk_polar(logits, targets, rhos, thetas):
        (N,) = targets.shape
        (N,V) = logits.shape
        device = rhos.device
        assert(rhos.shape == (N,))
        assert(thetas.shape == (N,))
        assert(targets.dtype == torch.int64)
        assert(rhos.dtype == torch.float64)
        assert(thetas.dtype == torch.float64)
        # construct phis
        phis = (
            torch.arange(V, device=device, dtype=torch.float64) + 0.5
        ) * (2 * torch.pi / V)
        # first, we get the horosphere distances
        alphas = thetas[:,None] - phis[None,:]  # angular offsets between z and v
        # print(f"alphas: {isnan_or_inf(alphas).any()}")
        cos_alphas = alphas.cos()
        # print(f"cos_alphas: {isnan_or_inf(cos_alphas).any()}")
        sin_alphas = alphas.sin()
        # print(f"sin_alphas: {isnan_or_inf(sin_alphas).any()}")
        log_two = torch.log(torch.tensor(2.0, device=device, dtype=torch.float64))
        # print(f"log_two: {isnan_or_inf(log_two).any()}")
        horosphere_dists = log_two - torch.logaddexp((1 - cos_alphas).log() + rhos[:,None], (1 + cos_alphas).log() - rhos[:,None])
        # print(f"horosphere_dists: {isnan_or_inf(horosphere_dists).any()}")
        # remake mu and subtract the target
        mu = (horosphere_dists + logits.to(torch.float64)).softmax(-1)
        mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float64)
        # print(f"mu: {isnan_or_inf(mu).any()}")
        # next, we transform the angles alpha after motion by rho
        betas = torch.atan2(sin_alphas, rhos.cosh()[:,None] * cos_alphas - rhos.sinh()[:,None])
        cos_errors = (betas.cos() * mu).sum(-1)
        sin_errors = (betas.sin() * mu).sum(-1)
        # print(f"cos_errors: {isnan_or_inf(cos_errors).any()}")
        # print(f"sin_errors: {isnan_or_inf(sin_errors).any()}")
        return (cos_errors.square() + sin_errors.square())/2

    # ---- Cartesian bridge loss ----------------------------------------------
    # Implements the formula directly, term-by-term:
    #   L(theta; y) = (d-1)^2 / 2 * (1 - ||z_t||^2)^2
    #                 * ||  (y - z_t) / ||y - z_t||^2
    #                     - E_{v ~ mu^theta(.|z_t)}[ (v - z_t) / ||v - z_t||^2 ]  ||^2
    # with mu^theta_v(z_t) = softmax_v( (d-1) h(z_t, v) + logits_v ),
    #      h(z_t, v)       = log[ (1 - ||z_t||^2) / ||v - z_t||^2 ].
    # The "_weighted" variant replaces the target (y - z_t)/||y - z_t||^2 with
    # E_{v ~ mu^*(.|z_t)}[(v - z_t)/||v - z_t||^2], where mu^* is the true Bayes
    # posterior softmax((d-1) h + log_ps).

    @staticmethod
    def _poincare_disk_cartesian_geometry(rhos, thetas, V):
        """Returns (z, v, diff, sq, one_minus_zz, h) used by every variant."""
        z = polar_to_cart(rhos, thetas)                                  # (N, 2)
        v = vocab_points(V, rhos.device, rhos.dtype)                     # (V, 2)
        diff = v - z.unsqueeze(-2)                                       # (N, V, 2)
        sq   = diff.square().sum(-1)                                     # (N, V)
        one_minus_zz = 1 - z.square().sum(-1, keepdim=True)              # (N, 1)
        h    = (one_minus_zz / sq).log()                                 # (N, V)
        return z, v, diff, sq, one_minus_zz, h

    @staticmethod
    def _poincare_disk_expected_radial(mu, diff, sq):
        """E_{v ~ mu}[ (v - z) / ||v - z||^2 ]  =  sum_v mu_v (v-z)/||v-z||^2."""
        return (mu / sq).unsqueeze(-1).mul(diff).sum(-2)                 # (N, 2)

    @staticmethod
    def _poincare_disk_cartesian_squared_residual(target, model, one_minus_zz, d=2):
        """L = (d-1)^2 / 2 * (1 - ||z||^2)^2 * ||target - model||^2."""
        residual = target - model                                        # (N, 2)
        return (d - 1) ** 2 / 2 * one_minus_zz.squeeze(-1).square() \
               * residual.square().sum(-1)

    @staticmethod
    def binary_bridge_loss_poincare_disk_cartesian(logits, targets, rhos, thetas):
        V, d = logits.shape[-1], 2
        z, v, diff, sq, one_minus_zz, h = HyperBridge._poincare_disk_cartesian_geometry(rhos, thetas, V)

        # target term: (y - z) / ||y - z||^2
        y_minus_z = v[targets] - z                                       # (N, 2)
        target = y_minus_z / y_minus_z.square().sum(-1, keepdim=True)    # (N, 2)

        # model term: E_{v ~ mu^theta(.|z)}[ (v - z) / ||v - z||^2 ]
        mu = ((d - 1) * h + logits.to(torch.float64)).softmax(-1)        # (N, V)
        model = HyperBridge._poincare_disk_expected_radial(mu, diff, sq)               # (N, 2)

        return HyperBridge._poincare_disk_cartesian_squared_residual(target, model, one_minus_zz, d=d)

    @staticmethod
    def _lorentz_boundary_points(V, device, dtype):
        phis = (torch.arange(V, device=device, dtype=dtype) + 0.5) * (2 * torch.pi / V)
        return torch.stack([torch.ones_like(phis), phis.cos(), phis.sin()], dim=-1)

    @staticmethod
    def _lorentz_inner(x, y):
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)

    @staticmethod
    def _lorentz_geometry(rhos, thetas, V, d):
        z = HyperBridge.polar_to_lorentz(rhos, thetas)                   # (N, 3)
        xi = HyperBridge._lorentz_boundary_points(V, rhos.device, rhos.dtype)
        inner = HyperBridge._lorentz_inner(z[:, None, :], xi[None, :, :]) # (N, V), negative
        log_poisson = (d - 1) *  (-(-inner).clamp_min(1e-300).log())     # (d - 1) * log 1 / (-<z,xi(y)>)
        directions = xi[None, :, :] / inner[:, :, None]                  # xi(y) / <z,xi(y)>
        return directions, log_poisson

    @staticmethod
    def _lorentz_norm_sq(x):
        return HyperBridge._lorentz_inner(x, x).clamp_min(0)

    @staticmethod
    def binary_bridge_loss_lorentz(logits, targets, rhos, thetas):
        V, d = logits.shape[-1], 2
        directions, log_poisson = HyperBridge._lorentz_geometry(rhos, thetas, V, d)
        # \mu^{\theta}(z_t) = softmax ( f_{\theta}(z_t) + (d-1) \sum_{v \in V} e_v \log \frac{- \langle z, x \rangle_{L}}{- \langle O, x \rangle_{L}} )
        mu = (log_poisson + logits.to(torch.float64)).softmax(-1) 
        target = directions[torch.arange(targets.numel(), device=targets.device), targets]
        model = (mu[:, :, None] * directions).sum(-2)
        residual = target - model
        return (d - 1) ** 2 / 2 * HyperBridge._lorentz_norm_sq(residual)

    @staticmethod
    def binary_nelbo_loss(logits, targets, rhos, thetas, proposal_weight, loss_geometry="poincare_polar"):
        if loss_geometry == LossGeometry.POINCARE_POLAR:
            # print("Use POINCARE_POLAR")
            bridge = HyperBridge.binary_bridge_loss_poincare_disk_polar(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
            )
        elif loss_geometry == LossGeometry.POINCARE_CARTESIAN:
            # print("Use POINCARE_CARTESIAN")
            bridge = HyperBridge.binary_bridge_loss_poincare_disk_cartesian(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
            )
        elif loss_geometry == LossGeometry.LORENTZ_CARTESIAN:
            # print("Use LORENTZ_CARTESIAN")
            bridge = HyperBridge.binary_bridge_loss_lorentz(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
            )
        else:
            raise ValueError(f"Unknown loss_geometry={loss_geometry!r}")
        return bridge * proposal_weight.to(dtype=bridge.dtype)

    @staticmethod
    def proposal(
        proposal_type: str,
        shape,
        device,
        dtype,
        unif_min: float,
        unif_max: float,
        exp_rate: float,
    ):
        proposal_type = proposal_type.lower()
        interval = float(unif_max - unif_min)
        if interval < 0:
            raise ValueError("proposal requires unif_max >= unif_min")

        if proposal_type == HyperBridge.PROPOSAL_UNIF_NAME:
            ts = unif_min + interval * torch.rand(shape, device=device, dtype=dtype)
            weights = torch.full_like(ts, interval)
            return ts, weights
        elif proposal_type == HyperBridge.PROPOSAL_TRUNCATED_EXP_NAME:
            if exp_rate <= 0:
                raise ValueError("proposal_exp_rate must be > 0")
            if interval == 0:
                ts = torch.full(shape, unif_min, device=device, dtype=dtype)
                return ts, torch.zeros_like(ts)
            u = torch.rand(shape, device=device, dtype=dtype).clamp(
                min=1e-12,
                max=1 - 1e-12,
            )
            normalizer = 1 - torch.exp(
                torch.tensor(-exp_rate * interval, device=device, dtype=dtype)
            )
            ts = unif_min - torch.log1p(-u * normalizer) / exp_rate
            density = exp_rate * torch.exp(-exp_rate * (ts - unif_min)) / normalizer
            return ts, density.reciprocal()
        elif proposal_type == HyperBridge.PROPOSAL_EXP_NAME:
            if exp_rate <= 0:
                raise ValueError("proposal_exp_rate must be > 0")
            if interval == 0:
                ts = torch.zeros(shape, device=device, dtype=dtype)
                return ts, torch.zeros_like(ts)
            u = torch.rand(shape, device=device, dtype=dtype).clamp(
                min=1e-12,
                max=1 - 1e-12,
            )
            # Use torch.log(-u) is also correct, but torch.log1p(-u) is more numerically stable because it can handle u close to 0
            ts = - torch.log1p(-u) / exp_rate
            density = exp_rate * torch.exp(-exp_rate * ts)
            return ts, density.reciprocal()
        elif proposal_type == HyperBridge.PROPOSAL_STRATIFIED_EXP_NAME:
            if exp_rate <= 0:
                raise ValueError("proposal_exp_rate must be > 0")
            numel = 1
            for dim in shape:
                numel *= int(dim)
            u = (
                torch.arange(numel, device=device, dtype=dtype)
                + torch.rand(numel, device=device, dtype=dtype)
            ) / numel
            u = u.view(-1)[torch.randperm(u.numel())].view(u.shape)
            u = u.clamp(min=1e-12, max=1 - 1e-12).reshape(shape)
            # Use torch.log(u) is also correct, but torch.log1p(-u) is more numerically stable because it can handle u close to 0
            ts = - torch.log(u) / exp_rate
            density = exp_rate * torch.exp(-exp_rate * ts)
            return ts, density.reciprocal()
        else:
            raise NotImplementedError(f"proposal_type={proposal_type} is not implemented.")

    @staticmethod
    def hyper_proposal(
        proposal_type: str,
        shape,
        device,
        dtype,
        dt: float = 0.01,
        T: int = 1000,
        exp_rate: float = 1.0,
    ):
        if dt is None or T is None or dt <= 0.0 or T <= 0:
            raise ValueError("dt and T must be positive for hyper_proposal.")

        total_time = dt * T
        unif_min = float(max(dt, 1e-8))
        unif_max = float(max(total_time, unif_min))

        ts, proposal_weight = HyperBridge.proposal(
            proposal_type=proposal_type,
            shape=shape,
            device=device,
            dtype=dtype,
            unif_min=unif_min,
            unif_max=unif_max,
            exp_rate=exp_rate,
        )
        return ts, proposal_weight
        

class HyperbolicDLM(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.bridge = HyperBridge()
        self.recorder = Recorder()
        self._test_step_offset = 0
        self._val_epoch_loss_total = 0.0
        self._val_epoch_loss_sq_total = 0.0
        self._val_epoch_weight = 0
        self._test_epoch_loss_total = 0.0
        self._test_epoch_loss_sq_total = 0.0
        self._test_epoch_weight = 0

        self.hyper_dim: int = config.get("hyper_dim", None)
        if self.hyper_dim is None:
            raise ValueError(f"config.hyper_dim, {self.hyper_dim}, shouldn't be None")
        self.loss_geometry = config.get("loss_geometry", None)
        if self.loss_geometry is None:
            raise ValueError(f"config.loss_geometry, {self.loss_geometry}, shouldn't be None")
        self.model_input_dim: int = self.hyper_dim
        self.mode = config.get("mode", None)
        
        if "lorentz" in self.loss_geometry:
            self.model_input_dim = self.hyper_dim + 1

        if self.mode == "tnb":
            self.model = MLPLM(
                vocab_size=config.vocab_size,
                input_dim=self.model_input_dim,
                output_dim=config.hyper_dim,
                hidden_size=config.hidden_size,
                depth=config.depth,
            )
        elif self.mode == "opt":
            self.model = OptimalModel(
                ps=process_ps(config.ps),
            )
        else:
            raise ValueError(f"mode shouldn't be {self.mode}, only support tnb and opt.")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.config.lr))

    @staticmethod
    def _variance_from_sums(total: float, sq_total: float, count: int) -> float:
        if count <= 1:
            return 0.0
        numerator = sq_total - total * total / count
        return max(numerator, 0.0) / (count - 1)

    def _compute_losses(self, batch: torch.Tensor):
        targets = batch.reshape(-1).to(device=self.device, dtype=torch.long)
        batch_size = targets.shape[0]

        ts, proposal_weight = self.bridge.hyper_proposal(
            proposal_type=self.config.proposal_type,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float64,
            dt=self.config.hyper_dt,
            T=self.config.hyper_T,
            exp_rate=self.config.proposal_exp_rate,
        )

        rhos, thetas = self.bridge.binary_bridge(ts=ts)
        if "lorentz" in self.loss_geometry:
            z = self.bridge.polar_to_lorentz(rhos, thetas).to(dtype=torch.float32)
        else:
            z = torch.stack([rhos, thetas], dim=-1).to(dtype=torch.float32)
        logits = self.model(z=z, t=ts.to(dtype=torch.float32))
        nelbo = self.bridge.binary_nelbo_loss(
            logits=logits,
            targets=targets,
            rhos=rhos,
            thetas=thetas,
            proposal_weight=proposal_weight,
            loss_geometry=self.loss_geometry,
        )
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        return {
            "loss": nelbo,
            "nelbo_loss": nelbo,
            "ce": ce,
            "ts": ts.to(dtype=torch.float32),
            "proposal_weight": proposal_weight.to(dtype=torch.float32),
            "rhos": rhos.to(dtype=torch.float32),
            "thetas": thetas.to(dtype=torch.float32),
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        loss_var = losses["loss"].var()
        self.recorder.add("train_loss", step=int(self.global_step) + 1, val=loss)
        self.recorder.add("train_loss_var", step=int(self.global_step) + 1, val=loss_var)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_var", loss_var, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_nelbo_loss",
            losses["nelbo_loss"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train_ce", losses["ce"].mean(), on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        loss_values = losses["loss"].detach().to(dtype=torch.float64)

        batch_size = int(loss_values.numel())
        if not self.trainer.sanity_checking:
            self._val_epoch_loss_total += float(loss_values.sum().cpu())
            self._val_epoch_loss_sq_total += float(loss_values.square().sum().cpu())
            self._val_epoch_weight += batch_size
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_nelbo_loss", losses["nelbo_loss"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_ce", losses["ce"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self._val_epoch_loss_total = 0.0
        self._val_epoch_loss_sq_total = 0.0
        self._val_epoch_weight = 0

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self._val_epoch_weight == 0:
            return
        mean_val_loss = self._val_epoch_loss_total / self._val_epoch_weight
        var_val_loss = self._variance_from_sums(
            self._val_epoch_loss_total,
            self._val_epoch_loss_sq_total,
            self._val_epoch_weight,
        )
        self.recorder.add("val_loss", step=int(self.global_step), val=mean_val_loss)
        self.recorder.add("val_loss_var", step=int(self.global_step), val=var_val_loss)
        self.log("val_loss_var", torch.tensor(var_val_loss, device=self.device, dtype=torch.float64), prog_bar=True)

    def on_test_start(self):
        self._test_step_offset = max(
            self.recorder.last_step("train_loss"),
            int(self.global_step),
        )
        self._test_epoch_loss_total = 0.0
        self._test_epoch_loss_sq_total = 0.0
        self._test_epoch_weight = 0

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        loss_values = losses["loss"].detach().to(dtype=torch.float64)

        batch_size = int(loss_values.numel())
        self._test_epoch_loss_total += float(loss_values.sum().cpu())
        self._test_epoch_loss_sq_total += float(loss_values.square().sum().cpu())
        self._test_epoch_weight += batch_size
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test_nelbo_loss", losses["nelbo_loss"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_ce", losses["ce"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_test_epoch_end(self):
        if self._test_epoch_weight == 0:
            return
        mean_test_loss = self._test_epoch_loss_total / self._test_epoch_weight
        var_test_loss = self._variance_from_sums(
            self._test_epoch_loss_total,
            self._test_epoch_loss_sq_total,
            self._test_epoch_weight,
        )
        self.recorder.add(
            "test_loss",
            step=self._test_step_offset + 1,
            val=mean_test_loss,
        )
        self.recorder.add(
            "test_loss_var",
            step=self._test_step_offset + 1,
            val=var_test_loss,
        )
        self.log("test_loss_var", torch.tensor(var_test_loss, device=self.device, dtype=torch.float64), prog_bar=True)

def save_results(res, folder, file_name: str = "test_metrics.json"):
    output_path = Path(folder) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if res:
        metrics = {name: float(value) for name, value in res[0].items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return output_path

def name_ext(config):
    ext: str = ""
    mode = config.get("mode", None)
    loss_geometry = config.get("loss_geometry", None)
    postfix = config.get("postfix", None)

    if mode is not None:
        ext += "_m-"
        if mode == "opt":
            ext += mode
        elif mode == "tnb":
            ext += mode
        else:
            raise ValueError(f"config.mode, {mode}, is not supported.")
    if loss_geometry is not None:
        ext += "_lg-"
        if loss_geometry == LossGeometry.POINCARE_POLAR:
            ext += "pp"
        elif loss_geometry == LossGeometry.POINCARE_CARTESIAN:
            ext += "pc"
        elif loss_geometry == LossGeometry.LORENTZ_POLAR:
            ext += "lp"
        elif loss_geometry == LossGeometry.LORENTZ_CARTESIAN:
            ext += "lc"
        else:
            raise ValueError(f"config.loss_geometry, {loss_geometry}, is not supported.")
    if postfix is not None:
        ext += f"_{postfix}"

    return ext

class TaskMgr:
    FINISHED_FILE: str = "finished.json"

    def __init__(self, work_dir: Optional[str | Path] = None):
        """
        If work_dir is None, use the current working directory.

        With Hydra:
            hydra.job.chdir=true

        Path.cwd() should be the current Hydra job directory.
        """
        self.work_dir = Path(work_dir) if work_dir is not None else Path("")
        self.finished_path = Path(os.path.join(self.work_dir, self.FINISHED_FILE))

    def finished(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark this task as finished.

        Only call this after the experiment successfully completes.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "finished": True,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }

        if metadata is not None:
            payload["metadata"] = metadata

        with open(self.finished_path, mode="w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def check_finished(self) -> bool:
        """
        Return True if this task has already finished.
        """
        if not self.finished_path.exists():
            return False

        try:
            with open(self.finished_path, mode="r", encoding="utf-8") as f:
                payload = json.load(f)

            return bool(payload.get("finished", False))

        except Exception:
            print(f"The task has been finished at {self.finished_path}.")
            # If the finished file is corrupted, do not skip the task.
            return False

    def remove_finished(self) -> None:
        """
        Remove the finished marker, useful if you want to rerun a task.
        """
        if self.finished_path.exists():
            self.finished_path.unlink()

def folder(config):
    return (
        f"unigram_test2_losses"
        f"_sp{config.max_steps}"
        f"_lr{config.lr}"
        f"_pt{config.proposal_type}"
        f"_per{config.proposal_exp_rate}"
        f"_vs{config.vocab_size}"
        f"{name_ext(config)}"
    )

OmegaConf.register_new_resolver("name_ext", lambda *, _root_: name_ext(_root_), replace=True)
OmegaConf.register_new_resolver("folder", lambda *, _root_: folder(_root_), replace=True)

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    defaults = OmegaConf.create(
        {
            "mode": "opt",
            "vocab_size": 10,
            "hyper_dim": 2,
            "hidden_size": 128,
            "train_size": 20_000,
            "depth": 3,
            "hyper_T": 1e7,
            "hyper_dt": 0.01,
            "val_size": 4_000,
            "test_size": 4_00000,
            "batch_size": 256,
            "max_steps": 2_000,
            "proposal_type": "exp",
            "proposal_exp_rate": 1.0,
            "loss_geometry": "poincare_polar",
            "lr": 1e-5,
            "ps": [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            # "ps": [0.1] * 10,
            "seed": 42,
            "num_workers": 0,
            "folder": "${folder:}",
            # "folder": "unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}",
            "loss_plot_path": "plot.jpg",
            "loss_data_path": "data.json",
            "loss_plot_ma_window": None,
        }
    )
    cfg = OmegaConf.merge(defaults, cfg)
    print(OmegaConf.to_yaml(cfg))

    task_mgr = TaskMgr(work_dir=cfg.folder)
    if task_mgr.check_finished():
        return

    L.seed_everything(int(cfg.seed), workers=True)
    datamodule = UnigramDataModule(config=cfg)
    model = HyperbolicDLM(config=cfg)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=int(cfg.max_steps),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    if cfg.mode == "tnb":
        trainer.fit(model, datamodule=datamodule)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=False)
    data_mgr = DataMgr(cfg.folder)
    saved = data_mgr.save(
        recorder=model.recorder,
        data_file=cfg.loss_data_path,
        fig_file=cfg.loss_plot_path,
        moving_average_window=cfg.loss_plot_ma_window,
    )
    print(f"Saved loss data to: {saved['data_path']}")
    print(f"Saved loss plot to: {saved['figure_path']}")
    if test_metrics:
        print("Test metrics:")
        for name, value in test_metrics[0].items():
            print(f"  {name}: {value:.6f}")

    metrics_path = save_results(test_metrics, cfg.folder)
    print(f"Saved test metrics to: {metrics_path}")

    task_mgr.finished()


if __name__ == "__main__":
    main()
    
