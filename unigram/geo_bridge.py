"""Closed-form binary (`d == 2`) hyperbolic heat-kernel sampler and geodesic.

This module is the self-contained `H^2` slice of the heat-kernel machinery:

- [`BinaryHyperbolicHeatKernel`]: the closed-form `d == 2` free heat
  kernel, its target-conditioned bridge, and the constant-speed geodesic.
- [`GeoUtils`]: pure coordinate converters between Poincare-disk and
  Lorentz-Cartesian representations plus the numeric boundary guards; no
  random state.

Numerical guards:
- `_LORENTZ_RHO_MAX = 20`: any Lorentz-Cartesian output beyond this raises
  `ValueError`. Polar outputs are unrestricted.
- `_SPHERE_T_MAX = 0.5`: carried over from the extraction and unused here.

See `unigram/hyper_dm.md` for the underlying math.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Geometry:
    """String tags identifying which manifold representation a tensor lives in."""

    POINCARE: str = "poincare"
    LORENTZ: str = "lorentz"

@dataclass
class Coordinate:
    """String tags selecting polar vs Cartesian output from kernel/bridge methods."""

    HYPERBOLIC_POLAR: str = "polar"
    CARTESIAN: str = "cartesian"

# ---------------------------------------------------------------------------
# Numerical-stability boundary constants
# ---------------------------------------------------------------------------
# Lorentz-Cartesian output becomes unreliable in float64 once cosh(rho) is large
# enough that the relative cancellation noise `cosh^2 * eps` exceeds the on-
# manifold tolerance. At `rho = 20`, `cosh(rho) ~ 2.4e8`, and the absolute defect
# `<z, z>_L + 1` floats at ~5e-8 — still well within an O(1e-6) acceptance — so 20
# is the conservative cutoff. Above this the polar form remains exact; only the
# Cartesian Lorentz conversion is refused.
_LORENTZ_RHO_MAX: float = 20.0

# The spherical Gruet ansatz `cos(phi) = v^2 + (1 - v^2) cos(s)` with the
# hyperbolic Poisson rate (d-1)^2 t / 8 is approximately correct only in the
# small-t regime; for `t > _SPHERE_T_MAX` the radial moments drift visibly from
# the analytic heat-kernel marginal (e.g. at d=2, t=10 the empirical
# `E[cos phi] ~ 0.34` versus the analytic `exp(-d t) ~ 2e-9`). The sampler refuses
# inputs that would exceed this regime.
_SPHERE_T_MAX: float = 0.5

class GeoUtils:
    # ---------------------------------------------------------------------------
    # Boundary-check helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _check_lorentz_rho_bound(
        rhos: torch.Tensor, d: int, ts: Optional[torch.Tensor] = None
    ) -> None:
        """Refuse Lorentz-Cartesian conversion when float64 precision is insufficient.

        The on-manifold defect `|z[0] - sqrt(1 + ||z[1:]||^2)|` floats at order
        `cosh(rho) * eps`, so for `rho > _LORENTZ_RHO_MAX = 20` the Cartesian image
        cannot be trusted to better than ~1e-6 even at float64. Polar output is
        unaffected; callers who need large rho should keep ``output_coord=HYPERBOLIC_POLAR``.
        """
        if rhos.numel() == 0:
            return
        rho_max = float(rhos.max().item())
        if rho_max > _LORENTZ_RHO_MAX:
            t_info = ""
            if ts is not None and ts.numel() > 0:
                t_info = f"max(ts)={float(ts.max().item()):.3g}, "
            raise ValueError(
                f"Lorentz-Cartesian output requires rho <= {_LORENTZ_RHO_MAX} for float64 "
                f"on-manifold precision; got d={d}, {t_info}max(rho)={rho_max:.3f}. "
                f"Use output_coord=HYPERBOLIC_POLAR for these parameters."
            )


    @staticmethod
    def _uniform_sphere(
        B: int, d: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Sample `B` points uniformly on `S^{d-1}` via normalized Gaussian draws."""
        if d == 1:
            raise ValueError("uniform sphere on S^0 (d=1) is not supported")
        g = torch.randn(B, d, dtype=dtype, device=device)
        return g / g.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)


    @staticmethod
    def _check_sphere_t_bound(ts: torch.Tensor, d: int) -> None:
        """Refuse spherical-heat-kernel inputs outside the small-t regime where the
        Gruet ansatz `cos(phi) = v^2 + (1 - v^2) cos(s)` is empirically reliable.
        """
        if ts.numel() == 0:
            return
        t_max = float(ts.max().item())
        if t_max > _SPHERE_T_MAX:
            raise ValueError(
                f"FreeSphericalHeatKernel is only validated for t <= {_SPHERE_T_MAX}; "
                f"got d={d}, max(ts)={t_max:.3g}. Larger t requires a different sampler."
            )

    # ---------------------------------------------------------------------------
    # Geodesic helper
    # ---------------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _geodesic_kernel(
        x: torch.Tensor, y: torch.Tensor, t, kappa: int
    ) -> torch.Tensor:
        """Constant-speed geodesic from x to y at fraction t.

        kappa = -1: Lorentz lerp on hyperboloid.  d_H = arccosh(-<x, y>_L).
        kappa = +1: SLERP on sphere.              d_S = arccos(<x, y>).

        The intrinsic distance is computed from the differential form
        `cosh(d_H) - 1 = <x - y, x - y>_L / 2` (hyperbolic) and
        `1 - cos(d_S) = ||x - y||^2 / 2`         (spherical), which avoids the
        catastrophic cancellation that would otherwise plague the inner product at
        high dimensions.
        """
        diff = x - y
        if kappa == -1:
            diff_inner = -diff[..., 0] * diff[..., 0] + (diff[..., 1:] * diff[..., 1:]).sum(-1)
            cosh_d_minus_one = (diff_inner / 2.0).clamp_min(0.0)
            d = torch.acosh(1.0 + cosh_d_minus_one)
            f = torch.sinh
        elif kappa == 1:
            diff_sq = (diff * diff).sum(-1)
            one_minus_cos = (diff_sq / 2.0).clamp(0.0, 2.0)
            d = torch.acos((1.0 - one_minus_cos).clamp(-1.0, 1.0))
            f = torch.sin
        else:
            raise ValueError(f"kappa must be -1 or +1; got {kappa}")

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(dtype=x.dtype, device=x.device)

        euclid = (1.0 - t) * x + t * y
        small = d < 1e-6
        tiny = torch.finfo(x.dtype).tiny
        fd = f(d).clamp_min(tiny)
        coef_x = f((1.0 - t) * d) / fd
        coef_y = f(t * d) / fd
        gamma = coef_x.unsqueeze(-1) * x + coef_y.unsqueeze(-1) * y
        return torch.where(small.unsqueeze(-1), euclid, gamma)

    # ---------------------------------------------------------------------------
    # Random-distribution generators
    # ---------------------------------------------------------------------------

    @staticmethod
    def sample_chi(ns: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Sample `chi(n)` via the Gamma identity `chi^2(n) ~ Gamma(n/2, scale=2)`.

        Direct Gamma sampling avoids materializing `sum(ns)` standard normals,
        which is critical when `ns` carries large counts (e.g. via Poisson rates
        at large `t`).

        Args:
            ns (`torch.Tensor` of shape `(batch_size,)`):
                Integer degrees of freedom.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float64`):
                Floating-point precision of the draw.

        Returns:
            `torch.Tensor` of shape `(batch_size,)`: chi samples.
        """
        concentration = ns.to(dtype) / 2
        rate = torch.tensor(0.5, device=ns.device, dtype=dtype)
        chi2 = torch.distributions.Gamma(concentration, rate).sample()
        return chi2.sqrt()

    @staticmethod
    def sample_chi_old(ns: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Legacy `chi(n)` sampler via summed squared normals.

        Retained for reference and parity checks against [`sample_chi`]; not used
        on the hot path because it allocates `sum(ns)` standard normals.

        Args:
            ns (`torch.Tensor`):
                Integer degrees of freedom, any shape.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float64`):
                Floating-point precision of the draw.

        Returns:
            `torch.Tensor` of the same shape as `ns`: chi samples.
        """
        nshape = ns.shape
        ns = ns.reshape(-1)
        M = ns.sum().item()
        x = torch.randn(M, device=ns.device, dtype=dtype).square()
        chi2 = torch.segment_reduce(x,'sum',lengths=ns)
        return chi2.sqrt().reshape(nshape)

    # ---------------------------------------------------------------------------
    # Coordinate converters
    # ---------------------------------------------------------------------------

    @staticmethod
    def _binary_polar_direction(
        thetas: torch.Tensor
    ) -> torch.Tensor:
        """Resolve a scalar angle (d=2) into a unit direction vector.

        Args:
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): angles.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 2)`: unit vectors
            `(cos theta, sin theta)`, one per row (`||.|| == 1`).
        """
        return torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)

    @staticmethod
    def _polar_direction(
        thetas: torch.Tensor
    ) -> torch.Tensor:
        """Resolve an d-dimensional angle into a unit direction vector.

        Args:
            thetas (`torch.FloatTensor` of shape `(..., d)`): angles.

        Returns:
            `torch.FloatTensor` of shape `(..., d)`: unit vectors
            `(cos theta, sin theta)`, one per row (`||.|| == 1`).
        """
        pass

    @staticmethod
    @torch.no_grad()
    def binary_hyperbolic_polar_to_poincare_cartesian(
        rhos: torch.Tensor,
        thetas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute z = tanh(rho/2) * direction inside the open unit disk.

        `tanh(rho/2)` saturates to 1.0 in float64 for `rho >= ~36`. The scale is
        clamped below 1 by one ulp so the strict invariant `||z|| < 1` holds for
        arbitrarily large `rho`.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`): hyperbolic radial.
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): angles.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 2)`: Poincare-disk Cartesian
            coordinates `(x, y)` with `||z|| < 1`.
        """
        direction = GeoUtils._binary_polar_direction(thetas=thetas)
        scale = torch.tanh(rhos / 2)
        one_minus_eps = 1.0 - torch.finfo(scale.dtype).eps
        scale = scale.clamp(max=one_minus_eps)
        return scale.unsqueeze(-1) * direction

    @staticmethod
    @torch.no_grad()
    def hyperbolic_polar_to_poincare_cartesian(
        rhos: torch.Tensor,
        thetas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute z = tanh(rho/2) * direction inside the open unit disk.

        `tanh(rho/2)` saturates to 1.0 in float64 for `rho >= ~36`. The scale is
        clamped below 1 by one ulp so the strict invariant `||z|| < 1` holds for
        arbitrarily large `rho`.

        Args:
            rhos (`torch.FloatTensor` of shape `(...,)`): hyperbolic radial.
            thetas (`torch.FloatTensor` of shape `(..., d)`): angles.

        Returns:
            `torch.FloatTensor` of shape `(..., d)`: Poincare-disk Cartesian
            coordinates `(x, y)` with `||z|| < 1`.
        """
        # TODO: Double check if it's correct
        direction = GeoUtils._polar_direction(thetas=thetas)
        scale = torch.tanh(rhos / 2)
        one_minus_eps = 1.0 - torch.finfo(scale.dtype).eps
        scale = scale.clamp(max=one_minus_eps)
        return scale.unsqueeze(-1) * direction

    @staticmethod
    @torch.no_grad()
    def hyperbolic_polar_to_lorentz_cartesian(
        rhos: torch.FloatTensor,
        thetas: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Convert `(rho, theta)` on `H^d` to Lorentz-Cartesian coordinates.

        Args:
            rhos (`torch.FloatTensor` of shape `(...,)`):
                Hyperbolic radial coordinate.
            thetas (`torch.FloatTensor` of shape `(..., d)`):
                Azimuthal angle.

        Returns:
            `torch.FloatTensor` of shape `(..., d + 1)`: Lorentz-Cartesian
            coordinates `(cosh rho, sinh rho * cos theta, sinh rho * sin theta)`.
        """
        sinh_r = torch.sinh(rhos)
        # TODO: FInish it

    @staticmethod
    @torch.no_grad()
    def binary_lorentz_cartesian_to_hyperbolic_polar(
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Lorentz-Cartesian to polar `(rho, thetas)`.

        `rho = arccosh(z[0])` recovers the radial geodesic distance to the origin; the
        angular part is `theta = atan2(z[2], z[1])` over the two spatial components in
        `d == 2`.

        Args:
            z (`torch.Tensor` of shape `(..., 3)`):
                Ambient Lorentz-Cartesian coordinates `(cosh rho, sinh rho * cos theta,
                sinh rho * sin theta)` with `z[0] >= 1`.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - `rhos` of shape `(...)`, `>= 0`.
                - `thetas` of shape `(...)`, scalar angle in `(-pi, pi]` for `d == 2`.
        """
        d = z.shape[-1]
        if d != 3:
            raise ValueError(f"Cartesian dimension should be 3, not {d}.")
        rhos = torch.acosh(z[..., 0].clamp_min(1.0))
        theta = torch.atan2(z[..., 2], z[..., 1])
        return rhos, theta

    @staticmethod
    @torch.no_grad()
    def lorentz_cartesian_to_hyperbolic_polar(
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Lorentz-Cartesian to polar `(rho, thetas)`.

        `rho = arccosh(z[0])` recovers the radial geodesic distance to the origin; the
        angular part is `theta = ` over the d spatial components.

        Args:
            z (`torch.Tensor` of shape `(..., d + 1)`):
                Ambient Lorentz-Cartesian coordinates `(cosh rho, ...)` with `z[0] >= 1`.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - `rhos` of shape `(...)`, `>= 0`.
                - `thetas` of shape `(..., d)`, scalar angle in `(-pi, pi]` for `d` dimension.
        """
        pass
        # TODO: Finish it

    @staticmethod
    @torch.no_grad()
    def poincare_cartesian_to_lorentz_cartesian(z: torch.Tensor) -> torch.Tensor:
        """Convert Poincare-disk Cartesian to Lorentz-Cartesian via stereographic lift.

        The map is `z -> ((1 + ||z||^2) / (1 - ||z||^2), 2 z / (1 - ||z||^2))`, taking
        `B^d` into the upper hyperboloid in `R^{1, d}`.

        Args:
            z (`torch.Tensor` of shape `(..., d)`):
                Poincare-disk Cartesian coordinates satisfying `||z|| < 1`.

        Returns:
            `torch.Tensor` of shape `(..., d + 1)`: ambient Lorentz-Cartesian coordinates
            satisfying `-z[0]^2 + sum(z[1:]^2) = -1`.
        """
        norm_sq = (z * z).sum(-1)
        denom = (1.0 - norm_sq).clamp_min(torch.finfo(z.dtype).tiny)
        t = (1.0 + norm_sq) / denom
        spatial = 2.0 * z / denom.unsqueeze(-1)
        return torch.cat([t.unsqueeze(-1), spatial], dim=-1)

    @staticmethod
    @torch.no_grad()
    def lorentz_cartesian_to_poincare_cartesian(z: torch.Tensor) -> torch.Tensor:
        """Convert Lorentz-Cartesian to Poincare-disk-Cartesian via stereographic projection.

        The map is `z -> z[..., 1:] / (1 + z[..., 0])`, the inverse of
        [`poincare_cartesian_to_lorentz_cartesian`], taking the upper hyperboloid in
        `R^{1, d}` into the open disk `B^d`.

        Args:
            z (`torch.Tensor` of shape `(..., d + 1)`):
                Lorentz-Cartesian coordinates on the hyperboloid with `z[0] >= 1`.

        Returns:
            `torch.Tensor` of shape `(..., d)`: Poincare-disk Cartesian coordinates
            satisfying `||z|| < 1`.
        """
        spatial = z[..., 1:]
        t = z[..., 0]
        denom = (1.0 + t).unsqueeze(-1).clamp_min(torch.finfo(z.dtype).tiny)
        return spatial / denom

class BinaryHyperbolicHeatKernel(GeoUtils):
    """Closed-form free hyperbolic heat kernel and bridge on `H^2` (d=2).

    Implements Gruet's series representation specialized to the disk: a Poisson
    count `n ~ Poisson(t / 8)`, a chi draw `s = sqrt(t) * chi(2n + 3)`, and a
    uniform mixing variable `v` jointly realize
    `rho = arccosh(v^2 + (1 - v^2) cosh(s))` distributed as the radial marginal
    of `H^2` Brownian motion at time `t`. The free heat kernel from the origin is
    rotationally symmetric, so its azimuthal angle is uniform on `[-pi, pi)`. The
    target-conditioned bridge methods reshape that uniform angle through the Poisson
    kernel `(cosh rho - sinh rho cos theta)^{-1}` before rotating it to the target.

    All methods are `@staticmethod` and run under `torch.no_grad()`; they accept
    a `ts` batch of heat times of shape `(batch_size,)` and an optional
    `output_coord` in `{Coordinate.HYPERBOLIC_POLAR, Coordinate.CARTESIAN}` selecting the
    return geometry.
    """

    @staticmethod
    @torch.no_grad()
    def binary_free_hyperbolic_heat_kernel(
        ts: torch.FloatTensor,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^2.

        `rho` is the radial coordinate (the geodesic distance to the origin);
        `theta` is the angular coordinate, shared by the Poincare and Lorentz models.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times, `> 0`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)`, each `torch.FloatTensor` of
            shape `(batch_size,)`; `rhos >= 0` and `thetas` uniform on `[-pi, pi)`.
        """
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * BinaryHyperbolicHeatKernel.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        rhos = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        # theta ~ Uniform[-pi, pi): the free heat kernel from the origin is
        # rotationally symmetric, so the angle is uniform. (The target-conditioned
        # Poisson-kernel angle is applied later in the bridge methods, not here.)
        # `torch.pi` is a float constant, not a callable.
        thetas = (us - 0.5) * (2 * torch.pi)
        return rhos, thetas

    @staticmethod
    @torch.no_grad()
    def binary_free_poincare_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^2.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times.
            output_coord (`str`, *optional*): `"polar"` (default) or `"cartesian"`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)`, each `torch.FloatTensor` of
            shape `(batch_size,)`. CARTESIAN: `torch.FloatTensor` of shape
            `(batch_size, 2)`, the Poincare-disk point `z` with `||z|| < 1`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts=ts)
        if output_coord == Coordinate.CARTESIAN:
            return GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
        return rhos, thetas

    @staticmethod
    @torch.no_grad()
    def binary_free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Sample from the free hyperbolic heat kernel on `H^2` in Lorentz form.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                Either `Coordinate.HYPERBOLIC_POLAR` (returns `(rho, theta)`) or
                `Coordinate.CARTESIAN` (returns Lorentz-Cartesian coords).

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, thetas
        GeoUtils._check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)

    @staticmethod
    @torch.no_grad()
    def binary_poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Sample the `H^2` bridge endpoint conditioned on a target embedding.

        The free kernel's uniform angle is reshaped by the Poisson-kernel
        inverse-CDF (concentration `exp(-rho)`, giving angular density
        `(cosh rho - sinh rho cos theta)^{-1}`) and then rotated by
        `atan2(e[1], e[0])` so the sample concentrates near the target direction
        `e = word_embedding[targets]`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, 2)`):
                Word-embedding table. Only the angular part is used.
            output_coord (`str`, *optional*, defaults to `Coordinate.HYPERBOLIC_POLAR`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
                `thetas` is unwrapped (the target rotation may push it outside
                `(-pi, pi]`); downstream consumers use it only via `cos`/`sin`.
            CARTESIAN: Poincare-disk coordinates of shape `(batch_size, 2)`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
        )
        # reshape the uniform free angle into the Poisson-kernel angle (concentration
        # exp(-rho), centered at 0): density (cosh rho - sinh rho cos theta)^{-1}.
        thetas = 2 * torch.atan((-rhos).exp() * torch.tan(thetas * 0.5))
        e = word_embedding[targets].to(ts.dtype)
        target_angle = torch.atan2(e[..., 1], e[..., 0])
        thetas = thetas + target_angle
        if output_coord == Coordinate.CARTESIAN:
            return GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
        return rhos, thetas

    @staticmethod
    @torch.no_grad()
    def binary_lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Lorentz-form `H^2` bridge endpoint conditioned on a target embedding.

        Lorentz analogue of [`binary_poincare_bridge`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, 2)`):
                Word-embedding table.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: Lorentz-Cartesian coords of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = BinaryHyperbolicHeatKernel.binary_poincare_bridge(
            ts=ts,
            targets=targets,
            word_embedding=word_embedding,
            output_coord=Coordinate.HYPERBOLIC_POLAR,
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, thetas
        GeoUtils._check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t,
        src_cartesian: Optional[torch.FloatTensor] = None,
        dest_cartesian: Optional[torch.FloatTensor] = None,
        cartesian_model: Optional[str] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed hyperbolic geodesic on `H^2` from source to destination at fraction `t`.

        Each endpoint is accepted either as a Cartesian tensor
        (`src_cartesian` / `dest_cartesian`, interpreted per `cartesian_model` as
        Poincare-disk or Lorentz) or as a polar pair (`src_radial`, `src_angular`
        / `dest_radial`, `dest_angular`); exactly one form per endpoint must be
        provided. The intrinsic distance uses the differential form
        `cosh d - 1 = <x - y, x - y>_L / 2` to avoid cancellation at large `d`.

        Args:
            t (`float`, or `torch.Tensor` of shape `()` or `(batch_size, 1)`):
                Fraction along the geodesic (`0` -> source, `1` -> destination).
                A per-sample column `(batch_size, 1)` broadcasts against the
                `(batch_size, 3)` ambient points; a bare `(batch_size,)` vector
                does not and is unsupported.
            src_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian source, interpreted per `cartesian_model`: shape
                `(batch_size, 3)` Lorentz when `cartesian_model == Geometry.LORENTZ`,
                or `(batch_size, 2)` Poincare-disk when `== Geometry.POINCARE`.
            dest_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian destination; same shape/interpretation as `src_cartesian`.
            cartesian_model (`str`, *optional*):
                `Geometry.POINCARE` or `Geometry.LORENTZ`; the local chart of the
                Cartesian coordinates. Required whenever a cartesian endpoint is
                given or cartesian output is requested. Governs both endpoints.
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the source.
            src_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the destination.
            dest_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the destination.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.HYPERBOLIC_POLAR`. Defaults to
                `Coordinate.CARTESIAN` when a cartesian source is given, else
                `Coordinate.HYPERBOLIC_POLAR`.

        Returns:
            CARTESIAN: chart-aware Cartesian output (requires `cartesian_model`) -
                `torch.FloatTensor` of shape `(batch_size, 3)` Lorentz-Cartesian when
                `cartesian_model == Geometry.LORENTZ`, or `(batch_size, 2)`
                Poincare-disk when `cartesian_model == Geometry.POINCARE`.
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each `torch.FloatTensor` of shape
                `(batch_size,)`.

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, if a
                cartesian endpoint or cartesian output lacks a valid `cartesian_model`,
                or if a polar input has `rho > _LORENTZ_RHO_MAX`.
        """
        if (src_cartesian is not None and (src_radial is not None or src_angular is not None)) or (
            src_cartesian is None and (src_radial is None or src_angular is None)
        ):
            raise ValueError(
                "Only accept one source, either src or (src_radial, src_angular)"
            )
        if (dest_cartesian is not None and (dest_radial is not None or dest_angular is not None)) or (
            dest_cartesian is None and (dest_radial is None or dest_angular is None)
        ):
            raise ValueError(
                "Only accept one destination, either dest or (dest_radial, dest_angular)"
            )

        if output_coord is None:
            output_coord = Coordinate.CARTESIAN if src_cartesian is not None else Coordinate.HYPERBOLIC_POLAR

        if src_cartesian is not None:
            if cartesian_model == Geometry.POINCARE:
                x_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z=src_cartesian)
            elif cartesian_model == Geometry.LORENTZ:
                x_amb = src_cartesian
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            GeoUtils._check_lorentz_rho_bound(src_radial, d=2)
            x_amb = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos=src_radial, thetas=src_angular)
        if dest_cartesian is not None:
            if cartesian_model == Geometry.POINCARE:
                y_amb = GeoUtils.poincare_cartesian_to_lorentz_cartesian(z=dest_cartesian)
            elif cartesian_model == Geometry.LORENTZ:
                y_amb = dest_cartesian
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            GeoUtils._check_lorentz_rho_bound(dest_radial, d=2)
            y_amb = GeoUtils.binary_hyperbolic_polar_to_lorentz_cartesian(rhos=dest_radial, thetas=dest_angular)

        interpolate = BinaryHyperbolicHeatKernel._geodesic_kernel(x_amb, y_amb, t, kappa=-1)

        if output_coord == Coordinate.CARTESIAN:
            if cartesian_model == Geometry.LORENTZ:
                return interpolate
            elif cartesian_model == Geometry.POINCARE:
                return GeoUtils.lorentz_cartesian_to_poincare_cartesian(z=interpolate)
            else:
                raise ValueError(f"cartesian_model, {cartesian_model}, is not supported, only support ({Geometry.LORENTZ}, {Geometry.POINCARE}).")
        return GeoUtils.binary_lorentz_cartesian_to_hyperbolic_polar(interpolate)

class HyperbolicHeatKernel(GeoUtils):
    """Closed-form free hyperbolic heat kernel and bridge on `H^d`.
    """

    @staticmethod
    @torch.no_grad()
    def free_hyperbolic_heat_kernel(
        ts: torch.FloatTensor,
        seq_len: int,
        embedding_size: int,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^d.

        `rho` is the radial coordinate (the geodesic distance to the origin);
        `theta` is the angular coordinate, shared by the Poincare and Lorentz models.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times, `> 0`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with `torch.FloatTensor` of
            shape `(batch_size, seq_len)` and `(batch_size, seq_len, embedding_size)` respectively; 
            `rhos >= 0` and `thetas` uniform on `[-pi, pi)`.
        """
        pass
        # TODO: Finish it

    @staticmethod
    @torch.no_grad()
    def free_poincare_heat_kernel(
        ts: torch.FloatTensor,
        seq_len: int,
        embedding_size: int,
        output_coord: Optional[str] = None,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^d.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times.
            output_coord (`str`, *optional*): `"polar"` (default) or `"cartesian"`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with `torch.FloatTensor` of
            shape `(batch_size, seq_len)` and `(batch_size, seq_len, embedding_size)` respectively; 
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size)`, 
            the Poincare-disk point `z` with `||z|| < 1`.
        """
        pass
        # TODO: Finish it

    @staticmethod
    @torch.no_grad()
    def free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        seq_len: int,
        embedding_size: int,
        output_coord: Optional[str] = None,
    ):
        """Sample from the free hyperbolic heat kernel on `H^2` in Lorentz form.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                Either `Coordinate.HYPERBOLIC_POLAR` (returns `(rho, theta)`) or
                `Coordinate.CARTESIAN` (returns Lorentz-Cartesian coords).

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with `torch.FloatTensor` of
            shape `(batch_size, seq_len)` and `(batch_size, seq_len, embedding_size)` respectively; 
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size)`. 
            Raises `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        pass
        # TODO: Finish it

    @staticmethod
    @torch.no_grad()
    def binary_poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Sample the `H^2` bridge endpoint conditioned on a target embedding.

        The free kernel's uniform angle is reshaped by the Poisson-kernel
        inverse-CDF (concentration `exp(-rho)`, giving angular density
        `(cosh rho - sinh rho cos theta)^{-1}`) and then rotated by
        `atan2(e[1], e[0])` so the sample concentrates near the target direction
        `e = word_embedding[targets]`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size, seq_len, embedding_size)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, embedding_size)`):
                Word-embedding table. Only the angular part is used.
            output_coord (`str`, *optional*, defaults to `Coordinate.HYPERBOLIC_POLAR`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with shape `(batch_size, seq_len)` 
                and `(batch_size, seq_len, embedding_size)` respectively.
                `thetas` is unwrapped (the target rotation may push it outside
                `(-pi, pi]`); downstream consumers use it only via `cos`/`sin`.
            CARTESIAN: Poincare-disk coordinates of shape `(batch_size, seq_len, embedding_size)`.
        """
        pass

    @staticmethod
    @torch.no_grad()
    def binary_lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Lorentz-form `H^2` bridge endpoint conditioned on a target embedding.

        Lorentz analogue of [`binary_poincare_bridge`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size, seq_len, embedding_size)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, embedding_size)`):
                Word-embedding table.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with shape `(batch_size, seq_len)` 
                and `(batch_size, seq_len, embedding_size)` respectively.
            CARTESIAN: Lorentz-Cartesian coords of shape `(batch_size, seq_len, embedding_size + 1)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        pass
        # TODO: Finish it

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t,
        src_cartesian: Optional[torch.FloatTensor] = None,
        dest_cartesian: Optional[torch.FloatTensor] = None,
        cartesian_model: Optional[str] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed hyperbolic geodesic on `H^2` from source to destination at fraction `t`.

        Each endpoint is accepted either as a Cartesian tensor
        (`src_cartesian` / `dest_cartesian`, interpreted per `cartesian_model` as
        Poincare-disk or Lorentz) or as a polar pair (`src_radial`, `src_angular`
        / `dest_radial`, `dest_angular`); exactly one form per endpoint must be
        provided. The intrinsic distance uses the differential form
        `cosh d - 1 = <x - y, x - y>_L / 2` to avoid cancellation at large `d`.

        Args:
            t (`float`, or `torch.Tensor` of shape `()` or `(batch_size, 1)`):
                Fraction along the geodesic (`0` -> source, `1` -> destination).
                A per-sample column `(batch_size, seq_len, 1)` broadcasts against the
                `(batch_size, seq_len, embedding_size + 1)` ambient points; 
                a bare `(batch_size, seq_len)` vector does not and is unsupported.
            src_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian source, interpreted per `cartesian_model`: shape
                `(batch_size, seq_len, embedding_size + 1)` Lorentz when 
                `cartesian_model == Geometry.LORENTZ`,
                or `(batch_size, seq_len, embedding_size)` Poincare-disk 
                when `== Geometry.POINCARE`.
            dest_cartesian (`torch.FloatTensor`, *optional*):
                Cartesian destination; same shape/interpretation as `src_cartesian`.
            cartesian_model (`str`, *optional*):
                `Geometry.POINCARE` or `Geometry.LORENTZ`; the local chart of the
                Cartesian coordinates. Required whenever a cartesian endpoint is
                given or cartesian output is requested. Governs both endpoints.
            src_radial (`torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size)`, *optional*):
                Polar radial coordinate of the source.
            src_angular (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Polar angle of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Polar radial coordinate of the destination.
            dest_angular (`torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size)`, *optional*):
                Polar angle of the destination.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.HYPERBOLIC_POLAR`. Defaults to
                `Coordinate.CARTESIAN` when a cartesian source is given, else
                `Coordinate.HYPERBOLIC_POLAR`.

        Returns:
            CARTESIAN: chart-aware Cartesian output (requires `cartesian_model`) -
                `torch.FloatTensor` of shape `(batch_size, seq_len, embedding_size + 1)` 
                Lorentz-Cartesian when `cartesian_model == Geometry.LORENTZ`, or 
                `(batch_size, seq_len, embedding_size)` Poincare-disk when 
                `cartesian_model == Geometry.POINCARE`.
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` with `torch.FloatTensor` of shape
                `(batch_size, seq_len)` and `(batch_size, seq_len, embedding_size)`

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, if a
                cartesian endpoint or cartesian output lacks a valid `cartesian_model`,
                or if a polar input has `rho > _LORENTZ_RHO_MAX`.
        """
        pass
        # TODO: Finish it