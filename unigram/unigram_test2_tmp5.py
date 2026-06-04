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
    from unigram.visualizer import DataMgr, Recorder, plot_embedding_concentration
except ModuleNotFoundError:
    from model import MLPLM, OptimalModel, polar_to_cart, vocab_points
    from dataset import UnigramDataModule, process_ps
    from visualizer import DataMgr, Recorder, plot_embedding_concentration

def isnan_or_inf(x):
    return torch.logical_or(torch.isnan(x), torch.isinf(x))

@dataclass
class LossGeometry:
    POINCARE_POLAR: str = "poincare_polar"
    POINCARE_POLAR_HOROCYCLE: str = "poincare_polar_horocycle"
    POINCARE_CARTESIAN: str = "poincare_cartesian"
    LORENTZ_POLAR: str = "lorentz_polar"
    LORENTZ_CARTESIAN: str = "lorentz_cartesian"
    HORO_CROSS_ENTROPY: str = "horo_cross_entropy"
    CROSS_ENTROPY: str = "cross_entropy"

@dataclass
class FlowPath:
    HYPERBOLIC_BOUNDARY: str = "hyperbolic_boundary"
    HYPERBOLIC_RFM: str = "hyperbolic_rfm"

class HyperBridge:
    PROPOSAL_EXP_NAME: str = "exp"
    PROPOSAL_STRATIFIED_EXP_NAME: str = "stratified_exp"
    PROPOSAL_TRUNCATED_EXP_NAME: str = "truncated_exp"
    PROPOSAL_UNIF_NAME: str = "unif"
    PROPOSAL_SIMPSON_NAME: str = "simpson"

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
    def wrapped_normal(shape, std, device=None, dtype=None):
        r"""Sample `(rho, u)` from the wrapped normal on `H^d` centred at the origin.

        Draws a tangent-space Gaussian `v ~ N(0, Sigma)` at the origin and reads off
        the polar coordinates of its exponential map. Since
        `exp_o(0, v) = (cosh ||v||, sinh ||v|| * v / ||v||)`, the radial coordinate is
        `rho = ||v||` (the geodesic distance to the origin) and the angular coordinate
        is the unit direction `u = v / ||v||` on `S^{d-1}`. The covariance `Sigma` is
        set by `std` (see below).

        Args:
            shape (`torch.Size` or tuple of `int`):
                Output sample shape `(..., embedding_size)`; the last axis
                `d = embedding_size` is the tangent / embedding dimension.
            std (`float` or `torch.FloatTensor` of shape `()`, `(d,)`, or `(d, d)`):
                Tangent-space Gaussian scale. A scalar or `(d,)` vector scales each
                axis (diagonal `Sigma`); a `(d, d)` matrix is the scale factor `L`
                giving `Sigma = L L^T` (`v = epsilon @ L^T`).

        Returns:
            tuple `(radial, angular)`:
                radial (`torch.FloatTensor` of shape `(...,)`):
                    Radial coordinate `rho = ||v|| >= 0`, the geodesic distance to
                    the origin.
                angular (`torch.FloatTensor` of shape `(..., embedding_size)`):
                    Unit direction `u = v / ||v||` on `S^{d-1}`.
        """

        # Wrapped normal on H^d with base point at the origin (Nagano et al. 2019):
        # sample a tangent-space Gaussian v ~ N(0, Sigma) and exp-map it at the
        # origin. Since exp_o(0, v) = (cosh||v||, sinh||v|| * v/||v||), the polar
        # coordinates are simply rho = ||v|| and direction u = v/||v||.
        v = torch.randn(shape, device=device, dtype=dtype)
        # Apply the tangent-space scale `std`: scalar / (D,) diagonal scale each
        # axis; a (D, D) matrix is a Cholesky-style factor giving Cov = std std^T.
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=v.dtype, device=v.device)
        else:
            std = std.to(dtype=v.dtype, device=v.device)
        if std.ndim < 2:
            v = v * std
        else:
            v = v @ std.transpose(-1, -2)

        ps = v.norm(p=2, dim=-1)
        thetas = v / v.norm(p=2, keepdim=True, dim=-1)
        return ps, thetas

    @staticmethod
    def binary_bridge_old(ts):
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * HyperBridge.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
        return (ps,thetas)

    @staticmethod
    def geodesic(
        src_radial: torch.FloatTensor,
        src_angle: torch.FloatTensor,
        dest_radial: torch.FloatTensor,
        dest_angle: torch.FloatTensor,
        ts: torch.FloatTensor,
    ):
        r"""Constant-speed hyperbolic geodesic on `H^2` from source to destination at fraction `ts`.

        Moves along the geodesic connecting the source polar point
        `(src_radial, src_angle)` to the destination polar point
        `(dest_radial, dest_angle)`, returning the point reached at fraction `ts` in
        the same polar `(rho, theta)` convention as the inputs (`ts = 0` -> source,
        `ts = 1` -> destination). The endpoints are lifted to the Lorentz hyperboloid
        and interpolated at constant speed; the intrinsic distance uses the
        differential form `cosh d - 1 = <x - y, x - y>_L / 2` to avoid the
        catastrophic cancellation of the raw inner product.

        Args:
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`):
                Radial coordinate `rho` of the source (geodesic distance to the origin).
            src_angle (`torch.FloatTensor` of shape `(batch_size,)`):
                Angular coordinate `theta` of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`):
                Radial coordinate `rho` of the destination.
            dest_angle (`torch.FloatTensor` of shape `(batch_size,)`):
                Angular coordinate `theta` of the destination.
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Fraction along the geodesic in `[0, 1]` (`0` -> source, `1` -> destination).

        Returns:
            tuple `(rhos, thetas)`, each `torch.FloatTensor` of shape `(batch_size,)`:
                polar coordinates of the interpolated point, with `rhos >= 0`.
        """
        # Constant-speed geodesic on H^2 from the source polar point to the
        # destination polar point, evaluated at fraction ts (0 -> src, 1 -> dest)
        # and returned in the same polar (rho, theta) convention as the inputs:
        #
        #   gamma(t) = sinh((1 - t) d) / sinh(d) * x + sinh(t d) / sinh(d) * y
        #
        # where x, y are the Lorentz lifts of the endpoints and
        # d = arccosh(-<x, y>_L) is their geodesic distance. The distance is
        # taken from the differential form cosh(d) - 1 = <x - y, x - y>_L / 2 to
        # avoid the catastrophic cancellation of the raw inner product.
        x = HyperBridge.polar_to_lorentz(src_radial, src_angle)    # (batch, 3)
        y = HyperBridge.polar_to_lorentz(dest_radial, dest_angle)  # (batch, 3)

        diff = x - y
        diff_inner = -diff[..., 0].square() + diff[..., 1:].square().sum(-1)
        d = torch.acosh(1.0 + (diff_inner / 2.0).clamp_min(0.0))    # (batch,)

        tiny = torch.finfo(x.dtype).tiny
        d_col = d.unsqueeze(-1)                                     # (batch, 1)
        ts_col = ts.to(dtype=x.dtype).unsqueeze(-1)                 # (batch, 1)
        sinh_d = torch.sinh(d_col).clamp_min(tiny)
        coef_x = torch.sinh((1.0 - ts_col) * d_col) / sinh_d
        coef_y = torch.sinh(ts_col * d_col) / sinh_d
        gamma = coef_x * x + coef_y * y                            # (batch, 3)
        # Coincident endpoints (d -> 0): the sinh ratios are 0/0, fall back to
        # the Euclidean lerp they approach.
        gamma = torch.where(d_col < 1e-6, (1.0 - ts_col) * x + ts_col * y, gamma)

        rho = torch.acosh(gamma[..., 0].clamp_min(1.0))            # (batch,)
        theta = torch.atan2(gamma[..., 2], gamma[..., 1])          # (batch,)
        return rho, theta

    @staticmethod
    def _vocab_angles(
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype,
        word_embedding: Optional[torch.FloatTensor] = None,
    ):
        # Boundary angle phi_v for every vocabulary word v.
        #   word_embedding given : direction of each word's 2-D embedding. atan2
        #     is scale-invariant, so the L2-normalization is a no-op for the angle
        #     and only guards against overflow.
        #   word_embedding None  : equally spaced points (v + 0.5) * 2*pi / V.
        if word_embedding is None:
            return (
                torch.arange(vocab_size, device=device, dtype=dtype) + 0.5
            ) * (2 * torch.pi / vocab_size)
        e = word_embedding.to(dtype)
        e = e / e.norm(dim=-1, p=2, keepdim=True)
        return torch.atan2(e[..., 1], e[..., 0])

    @staticmethod
    def rotate_with_target(
        thetas: torch.FloatTensor,
        targets: torch.LongTensor,
        vocab_size: int,
        word_embedding: Optional[torch.FloatTensor] = None,
    ):
        # Rotate the spike (at angle 0) onto each target word's boundary angle,
        # using the same word->angle map the loss uses (see _vocab_angles).
        phis = HyperBridge._vocab_angles(
            vocab_size=vocab_size,
            device=thetas.device,
            dtype=thetas.dtype,
            word_embedding=word_embedding,
        )
        return thetas + phis[targets]

    @staticmethod
    def binary_bridge(
        ts,
        targets: torch.LongTensor,
        vocab_size: int,
        word_embedding: Optional[torch.FloatTensor] = None,
    ):
        # Radial sampling
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * HyperBridge.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))

        # Free angles
        us = torch.rand_like(ts)
        # Spike angle on 0
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))

        thetas = HyperBridge.rotate_with_target(
            thetas=thetas,
            targets=targets,
            vocab_size=vocab_size,
            word_embedding=word_embedding,
        )
        return ps, thetas
    
    @staticmethod
    def binary_geodesic_bridge(
        ts,
        targets: torch.LongTensor,
        vocab_size: int,
        word_embedding: Optional[torch.FloatTensor] = None,
        std: float = 1.0,
    ):
        # Hyperbolic RFM path on H^2: geodesic from a wrapped-normal noise point
        # (ts=0) to the target word's hyperbolic embedding (ts=1). The word
        # embedding row is read as a tangent vector at the origin -> polar
        # (rho = ||e||, angle = atan2(e_y, e_x)). `ts` is the geodesic fraction in
        # [0, 1]. `vocab_size` is unused (the target endpoint comes from the
        # embedding directly).
        del vocab_size
        if word_embedding is None:
            raise ValueError(
                "binary_geodesic_bridge (HYPERBOLIC_RFM) needs a learnable "
                "word_embedding; set trainable_word_embedding=True."
            )
        dest = word_embedding[targets].to(ts.dtype)                 # (batch, d) tangent vec
        # Noise endpoint: wrapped normal in the tangent space at the origin.
        src_radial, src_dir = HyperBridge.wrapped_normal(
            shape=dest.shape, std=std, device=ts.device, dtype=ts.dtype,
        )
        # H^2: the unit direction <-> a scalar polar angle expected by `geodesic`.
        src_angle = torch.atan2(src_dir[..., 1], src_dir[..., 0])   # (batch,)
        dest_radial = dest.norm(p=2, dim=-1)                        # (batch,)
        dest_angle = torch.atan2(dest[..., 1], dest[..., 0])        # (batch,)
        rhos_t, thetas_t = HyperBridge.geodesic(
            src_radial=src_radial,
            src_angle=src_angle,
            dest_radial=dest_radial,
            dest_angle=dest_angle,
            ts=ts,
        )
        return rhos_t, thetas_t


    @staticmethod
    def polar_to_lorentz(rhos, thetas):
        sinh_r = torch.sinh(rhos)
        return torch.stack(
            [torch.cosh(rhos), sinh_r * thetas.cos(), sinh_r * thetas.sin()],
            dim=-1,
        )

    @staticmethod
    def binary_bridge_lorentz(ts):
        rhos, thetas = HyperBridge.binary_bridge(ts)
        return HyperBridge.polar_to_lorentz(rhos, thetas)

    # ---- Polar bridge loss ----------------------------------------------
    # logits        (N,V)     float64 (converts)
    # targets       (N,)      int64
    # rhos          (N,)      float64
    # thetas        (N,)      float64
    @staticmethod
    def binary_bridge_loss_poincare_disk_polar_old(logits, targets, rhos, thetas):
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
        cos_alphas = alphas.cos()
        sin_alphas = alphas.sin()
        log_two = torch.log(torch.tensor(2.0, device=device, dtype=torch.float64))
        horosphere_dists = log_two - torch.logaddexp((1 - cos_alphas).log() + rhos[:,None], (1 + cos_alphas).log() - rhos[:,None])
        # remake mu and subtract the target
        mu = (horosphere_dists + logits.to(torch.float64)).softmax(-1)
        mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float64)
        # next, we transform the angles alpha after motion by rho
        betas = torch.atan2(sin_alphas, rhos.cosh()[:,None] * cos_alphas - rhos.sinh()[:,None])
        cos_errors = (betas.cos() * mu).sum(-1)
        sin_errors = (betas.sin() * mu).sum(-1)
        return (cos_errors.square() + sin_errors.square())/2

    @staticmethod
    def binary_bridge_loss_poincare_disk_polar(logits, targets, rhos, thetas, word_embedding=None):
        (N,) = targets.shape
        (N,V) = logits.shape
        device = rhos.device
        assert(rhos.shape == (N,))
        assert(thetas.shape == (N,))
        assert(targets.dtype == torch.int64)
        assert(rhos.dtype == torch.float64)
        assert(thetas.dtype == torch.float64)
        assert word_embedding is None or tuple(word_embedding.shape) == (V, 2)
        # phi_v for every vocab word: learnable embedding angles when given,
        # otherwise the equally spaced points used by the *_old variant.
        phis = HyperBridge._vocab_angles(
            vocab_size=V,
            device=device,
            dtype=torch.float64,
            word_embedding=word_embedding,
        )
        # first, we get the horosphere distances
        # print(f"thetas ({thetas.mean().item()}): {torch.isfinite(thetas).all().item()}")
        # print(f"phis ({phis.mean().item()}): {torch.isfinite(phis).all().item()}")
        alphas = thetas[:,None] - phis[None,:]  # angular offsets between z and v
        # print(f"alphas: {torch.isfinite(alphas).all().item()}")
        cos_alphas = alphas.cos()
        sin_alphas = alphas.sin()
        log_two = torch.log(torch.tensor(2.0, device=device, dtype=torch.float64))
        # print(f"cos_alphas ({cos_alphas.mean().item()}): {torch.isfinite(cos_alphas).all().item()}")
        # print(f"sin_alphas ({sin_alphas.mean().item()}): {torch.isfinite(sin_alphas).all().item()}")
        # print(f"1 - cos_alphas ({(1 - cos_alphas).mean().item()}): {torch.isfinite(1 - cos_alphas).all().item()}")
        # print(f"1 + cos_alphas ({(1 + cos_alphas).mean().item()}): {torch.isfinite(1 + cos_alphas).all().item()}")
        horosphere_dists = log_two - torch.logaddexp((1 - cos_alphas).log() + rhos[:,None], (1 + cos_alphas).log() - rhos[:,None])
        # remake mu and subtract the target
        mu = (horosphere_dists + logits.to(torch.float64)).softmax(-1)
        mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float64)
        # next, we transform the angles alpha after motion by rho
        betas = torch.atan2(sin_alphas, rhos.cosh()[:,None] * cos_alphas - rhos.sinh()[:,None])
        cos_errors = (betas.cos() * mu).sum(-1)
        sin_errors = (betas.sin() * mu).sum(-1)
        return (cos_errors.square() + sin_errors.square())/2

    @staticmethod
    def binary_bridge_loss_poincare_disk_polar_horocycle(logits, targets, rhos, thetas, word_embedding=None):
        (N,) = targets.shape
        (N,V) = logits.shape
        device = rhos.device
        assert(rhos.shape == (N,))
        assert(thetas.shape == (N,))
        assert(targets.dtype == torch.int64)
        assert(rhos.dtype == torch.float64)
        assert(thetas.dtype == torch.float64)
        assert word_embedding is None or tuple(word_embedding.shape) == (V, 2)
        # phi_v for every vocab word: learnable embedding angles when given,
        # otherwise the equally spaced points used by the *_old variant.
        phis = HyperBridge._vocab_angles(
            vocab_size=V,
            device=device,
            dtype=torch.float64,
            word_embedding=word_embedding,
        )
        # first, we get the horosphere distances
        # print(f"thetas ({thetas.mean().item()}): {torch.isfinite(thetas).all().item()}")
        # print(f"phis ({phis.mean().item()}): {torch.isfinite(phis).all().item()}")
        alphas = thetas[:,None] - phis[None,:]  # angular offsets between z and v
        # print(f"alphas: {torch.isfinite(alphas).all().item()}")
        cos_alphas = alphas.cos()
        sin_alphas = alphas.sin()
        # remake mu and subtract the target
        mu = logits.to(torch.float64).softmax(-1)
        mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float64)
        # next, we transform the angles alpha after motion by rho
        betas = torch.atan2(sin_alphas, rhos.cosh()[:,None] * cos_alphas - rhos.sinh()[:,None])
        cos_errors = (betas.cos() * mu).sum(-1)
        sin_errors = (betas.sin() * mu).sum(-1)
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
        z = HyperBridge.polar_to_lorentz(rhos, thetas)                   # (N, d+1)
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
    def binary_bridge_loss_horo_crossentropy(logits, targets, rhos, thetas, word_embedding=None):
        (N,) = targets.shape
        (N,V) = logits.shape
        device = rhos.device
        assert(rhos.shape == (N,))
        assert(thetas.shape == (N,))
        assert(targets.dtype == torch.int64)
        assert(rhos.dtype == torch.float64)
        assert(thetas.dtype == torch.float64)
        # phi_v for every vocab word: learnable embedding angles when given,
        # otherwise the equally spaced points used by the *_old variant.
        phis = HyperBridge._vocab_angles(
            vocab_size=V,
            device=device,
            dtype=torch.float64,
            word_embedding=word_embedding,
        )
        # horosphere distances (the Busemann/Poisson term added to the logits)
        alphas = thetas[:,None] - phis[None,:]  # angular offsets between z and v
        cos_alphas = alphas.cos()
        log_two = torch.log(torch.tensor(2.0, device=device, dtype=torch.float64))
        horosphere_dists = log_two - torch.logaddexp((1 - cos_alphas).log() + rhos[:,None], (1 + cos_alphas).log() - rhos[:,None])
        # cross-entropy on the horosphere-augmented logits (pass logits, NOT the
        # softmaxed probabilities — cross_entropy applies log_softmax internally).
        return torch.nn.functional.cross_entropy(
            horosphere_dists + logits.to(torch.float64), targets, reduction='none'
        )

    @staticmethod
    def binary_bridge_loss_crossentropy(logits, targets, rhos, thetas):
        return torch.nn.functional.cross_entropy(logits, targets, reduction='none')

    @staticmethod
    def weighted_binary_loss(logits, targets, rhos, thetas, proposal_weight, word_embedding=None, loss_geometry="poincare_polar"):
        if loss_geometry == LossGeometry.POINCARE_POLAR:
            # print("Use POINCARE_POLAR")
            bridge = HyperBridge.binary_bridge_loss_poincare_disk_polar(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
                word_embedding=word_embedding,
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
        elif loss_geometry == LossGeometry.HORO_CROSS_ENTROPY:
            bridge = HyperBridge.binary_bridge_loss_horo_crossentropy(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
                word_embedding=word_embedding,
            )
        elif loss_geometry == LossGeometry.CROSS_ENTROPY:
            bridge = HyperBridge.binary_bridge_loss_crossentropy(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
            )
        else:
            raise ValueError(f"Unknown loss_geometry={loss_geometry!r}")
        return bridge * proposal_weight.to(dtype=bridge.dtype), bridge

    @staticmethod
    def weighted_binary_nelbo(logits, targets, rhos, thetas, proposal_weight, word_embedding=None, loss_geometry="poincare_polar"):
        if loss_geometry == LossGeometry.POINCARE_POLAR:
            bridge = HyperBridge.binary_bridge_loss_poincare_disk_polar(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
                word_embedding=word_embedding,
            )
        elif loss_geometry == LossGeometry.POINCARE_POLAR_HOROCYCLE:
            bridge = HyperBridge.binary_bridge_loss_poincare_disk_polar_horocycle(
                logits=logits,
                targets=targets,
                rhos=rhos,
                thetas=thetas,
                word_embedding=word_embedding,
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
        return bridge * proposal_weight.to(dtype=bridge.dtype), bridge

    @staticmethod
    def proposal(
        proposal_type: str,
        shape,
        device,
        dtype,
        unif_min: float,
        unif_max: float,
        exp_rate: float,
        generator: Optional[torch.Generator] = None,
    ):
        proposal_type = proposal_type.lower()
        interval = float(unif_max - unif_min)
        if interval < 0:
            raise ValueError("proposal requires unif_max >= unif_min")

        if proposal_type == HyperBridge.PROPOSAL_UNIF_NAME:
            ts = unif_min + interval * torch.rand(shape, device=device, dtype=dtype, generator=generator)
            weights = torch.full_like(ts, interval)
            return ts, weights
        elif proposal_type == HyperBridge.PROPOSAL_TRUNCATED_EXP_NAME:
            if exp_rate <= 0:
                raise ValueError("proposal_exp_rate must be > 0")
            if interval == 0:
                ts = torch.full(shape, unif_min, device=device, dtype=dtype)
                return ts, torch.zeros_like(ts)
            u = torch.rand(shape, device=device, dtype=dtype, generator=generator).clamp(
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
            u = torch.rand(shape, device=device, dtype=dtype, generator=generator).clamp(
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
                + torch.rand(numel, device=device, dtype=dtype, generator=generator)
            ) / numel
            u = u.view(-1)[torch.randperm(u.numel(), device=device, generator=generator)].view(u.shape)
            u = u.clamp(min=1e-12, max=1 - 1e-12).reshape(shape)
            # The same ts sampling as: ts = - torch.log(u) / exp_rate
            # ts = - torch.log1p(-u) / exp_rate
            ts = - torch.log(u) / exp_rate
            # The same equation as: weights = (exp_rate * ts).exp() / exp_rate
            weights = 1.0 / (exp_rate * u)
            # weights = (exp_rate * ts).exp() / exp_rate
            return ts, weights
        elif proposal_type == HyperBridge.PROPOSAL_SIMPSON_NAME:
            # Deterministic composite Simpson's 1/3 rule on [unif_min, unif_max].
            # The estimator is `mean(L * w)`, so we scale the per-node Simpson
            # weights by `numel` such that mean equals the Simpson sum:
            #   integral over [t_min, t_max] of L(t) dt
            #   approximately equal to sum_k w_simpson(t_k) * L(t_k)
            #   equal to mean(L * (numel * w_simpson)).
            numel = 1
            for dim in shape:
                numel *= int(dim)
            if numel < 3 or numel % 2 == 0:
                raise ValueError(
                    f"simpson proposal requires odd batch size >= 3 "
                    f"(an even number of sub-intervals), got numel={numel}"
                )
            if interval == 0:
                ts = torch.full(shape, unif_min, device=device, dtype=dtype)
                return ts, torch.zeros_like(ts)
            nodes, w_simpson = HyperBridge.simpson_nodes_weights(
                t_min=unif_min,
                t_max=unif_max,
                n_nodes=numel,
                device=device,
                dtype=dtype,
            )
            weights = w_simpson * float(numel)
            return nodes.reshape(shape), weights.reshape(shape)
        else:
            raise NotImplementedError(f"proposal_type={proposal_type} is not implemented.")

    @staticmethod
    def simpson_nodes_weights(t_min: float, t_max: float, n_nodes: int, device, dtype):
        """
        Composite Simpson's 1/3 rule nodes and weights on [t_min, t_max].

        Requires `n_nodes` odd and >= 3 (i.e. an even number of sub-intervals).
        Returns `(nodes, weights)` of shape `(n_nodes,)` such that
            integral over t in [t_min, t_max] of f(t) dt
            approximately equal to (weights * f(nodes)).sum().
        """
        if n_nodes < 3 or n_nodes % 2 == 0:
            raise ValueError(
                f"simpson_nodes_weights requires odd n_nodes >= 3, got {n_nodes}"
            )
        nodes = torch.linspace(t_min, t_max, n_nodes, device=device, dtype=dtype)
        h = (t_max - t_min) / (n_nodes - 1)
        weights = torch.ones(n_nodes, device=device, dtype=dtype)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        weights *= h / 3.0
        return nodes, weights

    @staticmethod
    def hyper_proposal(
        proposal_type: str,
        shape,
        device,
        dtype,
        dt: float = 0.01,
        T: int = 1000,
        exp_rate: float = 1.0,
        generator: Optional[torch.Generator] = None,
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
            generator=generator,
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
        self._test_epoch_wnelbo_total = 0.0
        self._test_epoch_wnelbo_sq_total = 0.0
        self._test_epoch_nelbo_total = 0.0
        self._test_epoch_nelbo_sq_total = 0.0
        self._test_epoch_ce_total = 0.0
        self._test_epoch_ce_sq_total = 0.0
        # Per-sample (timestep, loss, density) data collected over the test epoch.
        self._test_ts_chunks: list[torch.Tensor] = []
        self._test_loss_chunks: list[torch.Tensor] = []
        self._test_nelbo_chunks: list[torch.Tensor] = []
        self._test_weight_chunks: list[torch.Tensor] = []
        self.test_timesteps: Optional[torch.Tensor] = None
        self.test_losses: Optional[torch.Tensor] = None
        self.test_nelbos: Optional[torch.Tensor] = None
        self.test_proposal_weights: Optional[torch.Tensor] = None

        self.hyper_dim: int = config.get("hyper_dim", None)
        if self.hyper_dim is None:
            raise ValueError(f"config.hyper_dim, {self.hyper_dim}, shouldn't be None")
        self.loss_geometry = config.get("loss_geometry", None)
        if self.loss_geometry is None:
            raise ValueError(f"config.loss_geometry, {self.loss_geometry}, shouldn't be None")
        self.model_input_dim: int = self.hyper_dim
        self.mode = config.get("mode", None)
        # If False, the per-word boundary angles phi_v are fixed equally-spaced
        # ((v+0.5)*2*pi/V) instead of learned; the lm-head still trains as the logit
        # readout. See the word_embedding property.
        self.trainable_word_embedding = bool(config.get("trainable_word_embedding", None))
        self.flow_path = str(self.config.get("flow_path", None))

        if not self.trainable_word_embedding and self.flow_path == FlowPath.HYPERBOLIC_RFM:
            raise ValueError(f"config.flow_path = {FlowPath.HYPERBOLIC_RFM} requires config.trainable_word_embedding = True.")


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

    @property
    def word_embedding(self) -> Optional[torch.Tensor]:
        # Learnable boundary embedding (lm-head weights, shape (V, hyper_dim)) used to
        # define each word's boundary angle phi_v = atan2(e_v). Returns None — so the
        # bridge and loss fall back to fixed equally-spaced angles — when the backbone
        # has no such table (e.g. OptimalModel) OR when trainable_word_embedding is
        # False (the lm-head still trains as the logit readout in that case).
        if not self.trainable_word_embedding:
            return None
        head = getattr(self.model, "lm_head", None)
        return head.weight if head is not None else None

    @torch.no_grad()
    def _record_embedding_concentration(self, step: int) -> None:
        """Track how concentrated the learnable word embedding becomes.

        The loss sees the embedding only through its boundary angle
        phi_v = atan2(e_v), so concentration is measured on the circle:
          - emb_R         mean resultant length ||mean unit-vector|| in [0, 1];
                          0 = angles uniformly spread, 1 = all collapsed to one angle.
          - emb_circ_std  circular std sqrt(-2 ln R) (rad); small = concentrated.
          - emb_phi_min_gap  smallest wrap-around gap between adjacent angles (rad);
                          small = two words' angles merging.
          - emb_norm_{min,mean,max}  radial spread of the rows (the angle discards it).
        No-op when there is no learnable embedding (e.g. OptimalModel).
        """
        we = self.word_embedding
        if we is None:
            return
        e = we.detach().to(torch.float64)
        norms = e.norm(dim=-1)
        u = e / norms.clamp_min(1e-12).unsqueeze(-1)              # unit (cos phi, sin phi)
        R = u.mean(dim=0).norm()
        circ_std = torch.sqrt((-2.0 * R.clamp_min(1e-12).log()).clamp_min(0.0))
        metrics = {
            "emb_R": R,
            "emb_circ_std": circ_std,
            "emb_norm_min": norms.min(),
            "emb_norm_mean": norms.mean(),
            "emb_norm_max": norms.max(),
        }
        phis = torch.atan2(u[:, 1], u[:, 0])
        if not hasattr(self, "_init_phis"):
            self._init_phis = phis.clone()                       # snapshot for the before/after plot
        # Strided trajectory snapshots (first 2 dims) for the Poincaré-disk animation.
        if not hasattr(self, "_emb_snapshots"):
            self._emb_snapshots = []
            self._emb_snap_every = max(1, int(float(self.config.get("max_steps", 1))) // 300)
        if step == 1 or step % self._emb_snap_every == 0:
            self._emb_snapshots.append((step, e[:, :2].cpu().clone()))
        if e.shape[0] >= 2:
            sp = phis.sort().values
            gaps = torch.diff(sp)
            wrap = (sp[0] + 2 * torch.pi) - sp[-1]
            metrics["emb_phi_min_gap"] = torch.minimum(gaps.min(), wrap)
        for name, val in metrics.items():
            self.recorder.add(name, step=step, val=val)
        self.log("emb_R", R, on_step=True, on_epoch=False, prog_bar=True)

    def visualize_embedding_concentration(self, fig_path):
        """Render the embedding-collapse figure (concentration curves + a polar
        init-vs-final angle scatter) from the tracked series and current weights.
        Returns the figure path, or None when there is nothing to show (no
        learnable embedding, or training never recorded the series)."""
        we = self.word_embedding
        if we is None or "emb_R" not in self.recorder.history_dict:
            return None
        e = we.detach().to(torch.float64)
        norms = e.norm(dim=-1)
        u = e / norms.clamp_min(1e-12).unsqueeze(-1)
        final_phis = torch.atan2(u[:, 1], u[:, 0]).cpu().numpy()
        init = getattr(self, "_init_phis", None)
        return plot_embedding_concentration(
            recorder=self.recorder,
            output_path=fig_path,
            init_phis=None if init is None else init.cpu().numpy(),
            final_phis=final_phis,
            final_norms=norms.cpu().numpy(),
        )

    def animate_embedding_trajectory(self, out_path, fps: int = 12):
        """Animate the trainable word-embedding rows moving on the Poincaré disk
        across training steps (one frame per recorded snapshot, see
        `_record_embedding_concentration`). Each word keeps a fixed colour;
        the unit circle is the disk boundary. Returns the GIF path, or None when
        there is nothing to animate (no learnable embedding / no snapshots)."""
        snaps = getattr(self, "_emb_snapshots", None)
        if self.word_embedding is None or not snaps:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import numpy as np

        steps = [int(s) for s, _ in snaps]
        coords = [c.numpy() for _, c in snaps]                    # list of (V, 2)
        V = coords[0].shape[0]
        allc = np.concatenate(coords, axis=0)
        finite = allc[np.isfinite(allc).all(axis=1)]
        lim = 1.05 if finite.size == 0 else max(1.05, float(np.abs(finite).max()) * 1.1)
        colors = plt.cm.hsv(np.linspace(0.0, 1.0, V, endpoint=False))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1.3))   # disk boundary
        ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # Faint full trajectories behind the moving points (only when legible).
        if V <= 100 and len(coords) > 1:
            traj = np.stack(coords, axis=0)                       # (T, V, 2)
            for v in range(V):
                ax.plot(traj[:, v, 0], traj[:, v, 1], color=colors[v], lw=0.8, alpha=0.35, zorder=2)
        scat = ax.scatter(coords[0][:, 0], coords[0][:, 1], c=colors, s=70,
                          edgecolors="k", linewidths=0.5, zorder=4)
        title = ax.set_title(f"word embedding @ step {steps[0]}  (V={V})")

        def update(i):
            scat.set_offsets(coords[i])
            title.set_text(f"word embedding @ step {steps[i]}  (V={V})")
            return scat, title

        out_path = str(out_path)
        anim = FuncAnimation(fig, update, frames=len(coords),
                             interval=1000.0 / max(fps, 1), blit=False)
        anim.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return out_path

    @staticmethod
    def _variance_from_sums(total: float, sq_total: float, count: int) -> float:
        if count <= 1:
            return 0.0
        numerator = sq_total - total * total / count
        return max(numerator, 0.0) / (count - 1)

    @staticmethod
    def _std_from_sums(total: float, sq_total: float, count: int) -> float:
        return HyperbolicDLM._variance_from_sums(total=total, sq_total=sq_total, count=count) ** 0.5

    def get_logits_inputs(
        self,
        batch_size: int,
        targets: torch.LongTensor,
        hyper_dt: float,
        hyper_T: int,
        proposal_type: str,
        proposal_exp_rate: float,
        vocab_size: int,
        word_embedding: Optional[torch.FloatTensor],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ):
        ts, proposal_weight = self.bridge.hyper_proposal(
            proposal_type=proposal_type,
            shape=(batch_size,),
            device=device,
            dtype=torch.float64,
            dt=hyper_dt,
            T=hyper_T,
            exp_rate=proposal_exp_rate,
            generator=generator,
        )

        
        # For calculating posterior
        # Case 1: The word embedding is Equally divided around the circle
        # rhos, thetas = self.bridge.binary_bridge(ts=ts)
        # if self.rotate_emb:
        #     thetas = thetas + (
        #         targets.to(dtype=torch.float64) + 0.5
        #     ) * (2 * torch.pi / int(vocab_size))

        # Case 2: The word embedding is learnable
        if self.config.flow_path == FlowPath.HYPERBOLIC_BOUNDARY:
            rhos, thetas = self.bridge.binary_bridge(
                ts=ts,
                targets=targets,
                vocab_size=vocab_size,
                word_embedding=word_embedding,
            )
        elif self.config.flow_path == FlowPath.HYPERBOLIC_RFM:
            # RFM flow time is the geodesic fraction t ~ Uniform[0,1] (ts=0 -> noise,
            # ts=1 -> target). The boundary-bridge importance proposal does not apply,
            # so we override ts/weight here (weight = 1, plain CE expectation over t)
            # and feed this same t to the model below.
            ts = torch.rand((batch_size,), device=device, dtype=torch.float64, generator=generator)
            proposal_weight = torch.ones_like(ts)
            rhos, thetas = self.bridge.binary_geodesic_bridge(
                ts=ts,
                targets=targets,
                vocab_size=vocab_size,
                word_embedding=word_embedding,
            )
        else:
            raise ValueError(f"config.flow_path = {self.config.flow_path} is not supported, only suppport ({FlowPath.HYPERBOLIC_BOUNDARY}, {FlowPath.HYPERBOLIC_RFM}).")

        if "lorentz" in self.loss_geometry:
            z = self.bridge.polar_to_lorentz(rhos, thetas).to(dtype=torch.float32)
        else:
            z = torch.stack([rhos, thetas], dim=-1).to(dtype=torch.float32)
        logits = self.model(z=z, t=ts.to(dtype=torch.float32))
        return logits, ts, rhos, thetas, proposal_weight

    def _make_step_generator(self, salt: int) -> torch.Generator:
        """Per-step, per-path torch.Generator on self.device."""
        STEP_STRIDE = 1_000_003
        seed_value = (
            int(self.config.seed) * STEP_STRIDE
            + int(self.global_step) * 2
            + int(salt)
        ) & 0x7FFF_FFFF_FFFF_FFFF
        return torch.Generator(device=self.device).manual_seed(seed_value)

    def _compute_losses(self, batch: torch.Tensor):
        targets = batch.reshape(-1).to(device=self.device, dtype=torch.long)
        batch_size = targets.shape[0]

        loss_gen = self._make_step_generator(salt=0)
        loss_logits, ts_loss, rhos_loss, thetas_loss, pw_loss = self.get_logits_inputs(
            batch_size=batch_size,
            targets=targets,
            hyper_dt=self.config.hyper_dt,
            hyper_T=self.config.hyper_T,
            proposal_type=self.config.loss_proposal_type,
            proposal_exp_rate=self.config.loss_proposal_exp_rate,
            vocab_size=self.config.vocab_size,
            word_embedding=self.word_embedding,
            device=self.device,
            generator=loss_gen,
        )
        wloss, loss = self.bridge.weighted_binary_loss(
            logits=loss_logits,
            targets=targets,
            rhos=rhos_loss,
            thetas=thetas_loss,
            proposal_weight=pw_loss,
            loss_geometry=self.loss_geometry,
            word_embedding=self.word_embedding,
        )
        ce = torch.nn.functional.cross_entropy(loss_logits, targets, reduction="none")

        if (self.config.nelbo_proposal_type == self.config.loss_proposal_type
              and self.config.nelbo_proposal_exp_rate == self.config.loss_proposal_exp_rate
              and self.config.nelbo_geometry == self.loss_geometry
              and self.config.loss_geometry not in {LossGeometry.CROSS_ENTROPY, LossGeometry.HORO_CROSS_ENTROPY}):
            # Identical configs — reuse the loss-path computation.
            wnelbo = wloss
            nelbo = loss
            ts_nelbo = ts_loss
            pw_nelbo = pw_loss
        else:
            nelbo_gen = self._make_step_generator(salt=1)
            nelbo_logits, ts_nelbo, rhos_nelbo, thetas_nelbo, pw_nelbo = self.get_logits_inputs(
                batch_size=batch_size,
                targets=targets,
                hyper_dt=self.config.hyper_dt,
                hyper_T=self.config.hyper_T,
                proposal_type=self.config.nelbo_proposal_type,
                proposal_exp_rate=self.config.nelbo_proposal_exp_rate,
                vocab_size=self.config.vocab_size,
                word_embedding=self.word_embedding,
                device=self.device,
                generator=nelbo_gen,
            )
            wnelbo, nelbo = self.bridge.weighted_binary_nelbo(
                logits=nelbo_logits,
                targets=targets,
                rhos=rhos_nelbo,
                thetas=thetas_nelbo,
                proposal_weight=pw_nelbo,
                loss_geometry=self.config.nelbo_geometry,
                word_embedding=self.word_embedding,
            )

        return {
            "loss": wloss,
            "wnelbo_loss": wnelbo,
            "nelbo_loss": nelbo,
            "ce": ce,
            "ts": ts_nelbo.to(dtype=torch.float32),
            "proposal_weight": pw_nelbo.to(dtype=torch.float32),
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        loss_std = losses["loss"].std()
        self.recorder.add("train_loss", step=int(self.global_step) + 1, val=loss)
        self.recorder.add("train_loss_std", step=int(self.global_step) + 1, val=loss_std)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_std", loss_std, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_nelbo_loss",
            losses["nelbo_loss"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train_ce", losses["ce"].mean(), on_step=True, on_epoch=True, prog_bar=False)
        self._record_embedding_concentration(step=int(self.global_step) + 1)
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
        std_val_loss = self._std_from_sums(
            self._val_epoch_loss_total,
            self._val_epoch_loss_sq_total,
            self._val_epoch_weight,
        )
        self.recorder.add("val_loss", step=int(self.global_step), val=mean_val_loss)
        self.recorder.add("val_loss_std", step=int(self.global_step), val=std_val_loss)
        self.log("val_loss_std", torch.tensor(std_val_loss, device=self.device, dtype=torch.float64), prog_bar=True)

    def on_test_start(self):
        self._test_step_offset = max(
            self.recorder.last_step("train_loss"),
            int(self.global_step),
        )
        self._test_epoch_loss_total = 0.0
        self._test_epoch_loss_sq_total = 0.0
        self._test_epoch_weight = 0
        self._test_epoch_wnelbo_total = 0.0
        self._test_epoch_wnelbo_sq_total = 0.0
        self._test_epoch_nelbo_total = 0.0
        self._test_epoch_nelbo_sq_total = 0.0
        self._test_epoch_ce_total = 0.0
        self._test_epoch_ce_sq_total = 0.0
        self._test_ts_chunks = []
        self._test_loss_chunks = []
        self._test_nelbo_chunks = []
        self._test_weight_chunks = []

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        del batch_idx
        losses = self._compute_losses(batch)
        loss = losses["loss"].mean()
        loss_values = losses["loss"].detach().to(dtype=torch.float64)

        batch_size = int(loss_values.numel())
        self._test_epoch_loss_total += float(loss_values.sum().cpu())
        self._test_epoch_loss_sq_total += float(loss_values.square().sum().cpu())
        self._test_epoch_weight += batch_size
        wnelbo_values = losses["wnelbo_loss"].detach().to(dtype=torch.float64)
        self._test_epoch_wnelbo_total += float(wnelbo_values.sum().cpu())
        self._test_epoch_wnelbo_sq_total += float(wnelbo_values.square().sum().cpu())
        nelbo_values = losses["nelbo_loss"].detach().to(dtype=torch.float64)
        self._test_epoch_nelbo_total += float(nelbo_values.sum().cpu())
        self._test_epoch_nelbo_sq_total += float(nelbo_values.square().sum().cpu())
        ce_values = losses["ce"].detach().to(dtype=torch.float64)
        self._test_epoch_ce_total += float(ce_values.sum().cpu())
        self._test_epoch_ce_sq_total += float(ce_values.square().sum().cpu())
        # Record the per-sample (timestep, loss, weight) distribution.
        self._test_ts_chunks.append(losses["ts"].detach().to(dtype=torch.float32, device="cpu"))
        self._test_loss_chunks.append(loss_values.to(dtype=torch.float32, device="cpu"))
        self._test_nelbo_chunks.append(
            losses["nelbo_loss"].detach().to(dtype=torch.float32, device="cpu")
        )
        self._test_weight_chunks.append(
            losses["proposal_weight"].detach().to(dtype=torch.float32, device="cpu")
        )
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test_nelbo", losses["nelbo_loss"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_ce", losses["ce"].mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_test_epoch_end(self):
        if self._test_epoch_weight == 0:
            return
        mean_test_loss = self._test_epoch_loss_total / self._test_epoch_weight
        std_test_loss = self._std_from_sums(
            self._test_epoch_loss_total,
            self._test_epoch_loss_sq_total,
            self._test_epoch_weight,
        ) ** 0.5
        self.recorder.add(
            "test_loss",
            step=self._test_step_offset + 1,
            val=mean_test_loss,
        )
        self.recorder.add(
            "test_loss_std",
            step=self._test_step_offset + 1,
            val=std_test_loss,
        )
        self.log("test_loss_std", torch.tensor(std_test_loss, device=self.device, dtype=torch.float64), prog_bar=True)
        mean_test_wnelbo = self._test_epoch_wnelbo_total / self._test_epoch_weight
        std_test_wnelbo = self._std_from_sums(
            self._test_epoch_wnelbo_total,
            self._test_epoch_wnelbo_sq_total,
            self._test_epoch_weight,
        )
        std_test_nelbo = self._std_from_sums(
            self._test_epoch_nelbo_total,
            self._test_epoch_nelbo_sq_total,
            self._test_epoch_weight,
        )
        std_test_ce = self._std_from_sums(
            self._test_epoch_ce_total,
            self._test_epoch_ce_sq_total,
            self._test_epoch_weight,
        )
        self.log("test_wnelbo", torch.tensor(mean_test_wnelbo, device=self.device, dtype=torch.float64))
        self.log("test_wnelbo_std", torch.tensor(std_test_wnelbo, device=self.device, dtype=torch.float64))
        self.log("test_nelbo_std", torch.tensor(std_test_nelbo, device=self.device, dtype=torch.float64))
        self.log("test_ce_std", torch.tensor(std_test_ce, device=self.device, dtype=torch.float64))
        if self._test_ts_chunks:
            self.test_timesteps = torch.cat(self._test_ts_chunks)
            self.test_losses = torch.cat(self._test_loss_chunks)
            self.test_nelbos = torch.cat(self._test_nelbo_chunks)
            self.test_proposal_weights = torch.cat(self._test_weight_chunks)

def save_results(res, folder, file_name: str = "test_metrics.json"):
    output_path = Path(folder) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if res:
        metrics = {name: float(value) for name, value in res[0].items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return output_path

def plot_test_loss_vs_timestep(
    timesteps,
    weighted_nelbo,
    nelbo,
    proposal_density,
    fig_path,
    data_path,
    subsample: float = 1.0,
    title: str = "test_loss vs timestep",
):
    """Record per-sample (timestep, loss, density) data and scatter-plot it.

    Three series are plotted against `timesteps` on a shared log-x axis: the
    importance-weighted NELBO, the unweighted NELBO (the loss integrand), and
    the proposal density. `timesteps`, `weighted_nelbo`, `nelbo` and
    `proposal_density` are equal-length 1-D sequences, one entry per test
    sample. `subsample` is the fraction of points in [0.0, 1.0] to randomly
    keep (1.0 keeps all); it governs both the JSON written to `data_path` and
    the log-log scatter saved to `fig_path`. Returns the two output paths.
    """
    if not 0.0 <= subsample <= 1.0:
        raise ValueError(f"subsample must be in [0.0, 1.0], got {subsample}")

    ts = np.asarray(timesteps, dtype=np.float64).reshape(-1)
    series = {
        "weighted_nelbo": np.asarray(weighted_nelbo, dtype=np.float64).reshape(-1),
        "nelbo": np.asarray(nelbo, dtype=np.float64).reshape(-1),
        "proposal_density": np.asarray(proposal_density, dtype=np.float64).reshape(-1),
    }
    for name, arr in series.items():
        if arr.shape != ts.shape:
            raise ValueError(f"{name} and timesteps must have the same length")

    keep = int(round(subsample * ts.size))
    if keep < ts.size:
        idx = np.random.default_rng(0).choice(ts.size, size=keep, replace=False)
        ts = ts[idx]
        series = {name: arr[idx] for name, arr in series.items()}

    data_path, fig_path = Path(data_path), Path(fig_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"timestep": ts.tolist(), **{n: a.tolist() for n, a in series.items()}}, f
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("weighted_nelbo", "weighted NELBO"),
        ("nelbo", "NELBO (unweighted)"),
        ("proposal_density", "proposal density"),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 12), sharex=True)
    x_log = bool(np.any(ts > 0))
    for ax, (key, ylabel) in zip(axes, panels):
        arr = series[key]
        ax.scatter(ts, arr, s=4, alpha=0.3)
        if x_log:
            ax.set_xscale("log")
        if bool(np.any(arr > 0)):
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return {"data_path": data_path, "figure_path": fig_path}

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
        elif loss_geometry == LossGeometry.HORO_CROSS_ENTROPY:
            ext += "hce"
        elif loss_geometry == LossGeometry.CROSS_ENTROPY:
            ext += "ce"
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
        f"_pt{config.loss_proposal_type}"
        f"_per{config.loss_proposal_exp_rate}"
        f"_hT{config.hyper_T}"
        f"_hdt{config.hyper_dt}"
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
            "test_size": 4_000000,
            "batch_size": 2048,
            "max_steps": 2_0000,
            "loss_proposal_type": "exp",
            "loss_proposal_exp_rate": 1.0,
            "loss_geometry": "poincare_polar",
            "nelbo_proposal_type": "exp",
            "nelbo_proposal_exp_rate": 1.0,
            "nelbo_geometry": "poincare_polar",
            "rotate_emb": False,
            "trainable_word_embedding": True,
            "flow_path": "hyperbolic_boundary",
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
            "ts_loss_plot_path": "test_loss_vs_timestep.jpg",
            "ts_loss_data_path": "test_loss_vs_timestep.json",
            "ts_loss_subsample": 1.0,
        }
    )
    cfg = OmegaConf.merge(defaults, cfg)
    print(OmegaConf.to_yaml(cfg))

    # Resolve the ps spec (e.g. a list, or a named string like "c1e3_exp1.0")
    # through the data module, then align ps and vocab_size on the config so the
    # model, the run-folder name, and the data all use the same distribution.
    datamodule = UnigramDataModule(config=cfg)
    cfg.ps = datamodule.ps
    cfg.vocab_size = datamodule.vocab_size

    task_mgr = TaskMgr(work_dir=cfg.folder)
    if task_mgr.check_finished():
        return

    L.seed_everything(int(cfg.seed), workers=True)
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
    emb_fig = model.visualize_embedding_concentration(
        os.path.join(cfg.folder, "emb_concentration.jpg")
    )
    if emb_fig is not None:
        print(f"Saved embedding-concentration plot to: {emb_fig}")
    emb_anim = model.animate_embedding_trajectory(
        os.path.join(cfg.folder, "emb_trajectory.gif")
    )
    if emb_anim is not None:
        print(f"Saved embedding-trajectory animation to: {emb_anim}")
    if model.test_timesteps is not None:
        proposal_density = 1.0 / model.test_proposal_weights
        ts_saved = plot_test_loss_vs_timestep(
            timesteps=model.test_timesteps,
            weighted_nelbo=model.test_losses,
            nelbo=model.test_nelbos,
            proposal_density=proposal_density,
            fig_path=os.path.join(cfg.folder, cfg.ts_loss_plot_path),
            data_path=os.path.join(cfg.folder, cfg.ts_loss_data_path),
            subsample=float(cfg.ts_loss_subsample),
        )
        print(f"Saved test_loss-vs-timestep data to: {ts_saved['data_path']}")
        print(f"Saved test_loss-vs-timestep plot to: {ts_saved['figure_path']}")
    if test_metrics:
        print("Test metrics:")
        for name, value in test_metrics[0].items():
            print(f"  {name}: {value:.6f}")

    metrics_path = save_results(test_metrics, cfg.folder)
    print(f"Saved test metrics to: {metrics_path}")

    task_mgr.finished()


if __name__ == "__main__":
    main()
    
