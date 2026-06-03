"""Free hyperbolic / spherical heat-kernel samplers and geodesic primitives.

This module exposes three parallel sampler classes that share an identical
API surface (modulo a `poincare` / `lorentz` / `sphere` prefix):

- [`FreeBinaryHyperbolicHeatKernel`]: closed-form `d == 2` reference; every
  `d == 2` call from the higher-dimensional classes dispatches here for
  bit-exact parity.
- [`FreeHyperbolicHeatKernel`]: `d`-dimensional free hyperbolic heat kernel
  on `H^d` with three interchangeable angular samplers (`boost`, `icdf`,
  `vmf`).
- [`FreeSphericalHeatKernel`]: dual `S^d` sampler via the kappa-flipped
  Gruet ansatz; validated only for `t <= _SPHERE_T_MAX`.

The module also exposes a set of coordinate converters between Poincare-ball,
Lorentz-Cartesian, and ambient-sphere representations; these are pure tensor
ops with no random state.

Numerical guards:
- `_LORENTZ_RHO_MAX = 20`: any Lorentz-Cartesian output beyond this raises
  `ValueError`. Polar outputs are unrestricted.
- `_SPHERE_T_MAX = 0.5`: any spherical heat-kernel input beyond this raises
  `ValueError`.

See `unigram/hyper_dm.md` for the underlying math.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Geometry:
    """String tags identifying which manifold representation a tensor lives in."""

    POINCARE: str = "poincare"
    LORENTZ_POLAR: str = "lorentz_polar"
    LORENTZ_CARTESIAN: str = "lorentz_cartesian"


@dataclass
class Coordinate:
    """String tags selecting polar vs Cartesian output from kernel/bridge methods."""

    POLAR: str = "polar"
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


# ---------------------------------------------------------------------------
# Boundary-check helpers
# ---------------------------------------------------------------------------

def _check_lorentz_rho_bound(
    rhos: torch.Tensor, d: int, ts: Optional[torch.Tensor] = None
) -> None:
    """Refuse Lorentz-Cartesian conversion when float64 precision is insufficient.

    The on-manifold defect `|z[0] - sqrt(1 + ||z[1:]||^2)|` floats at order
    `cosh(rho) * eps`, so for `rho > _LORENTZ_RHO_MAX = 20` the Cartesian image
    cannot be trusted to better than ~1e-6 even at float64. Polar output is
    unaffected; callers who need large rho should keep ``output_coord=POLAR``.
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
            f"Use output_coord=POLAR for these parameters."
        )


def _uniform_sphere(
    B: int, d: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Sample `B` points uniformly on `S^{d-1}` via normalized Gaussian draws."""
    if d == 1:
        raise ValueError("uniform sphere on S^0 (d=1) is not supported")
    g = torch.randn(B, d, dtype=dtype, device=device)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)


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
# Module-level coordinate converters
# ---------------------------------------------------------------------------

def _polar_direction(rhos: torch.Tensor, thetas_or_u: torch.Tensor) -> torch.Tensor:
    """Resolve scalar-theta (d=2) vs unit-vector (d>=3) into a (..., d) direction.

    Unit-vector inputs are renormalized so the Lorentz/sphere invariants are not
    polluted by residual norm error from upstream samplers.
    """
    if thetas_or_u.ndim == rhos.ndim:
        return torch.stack([torch.cos(thetas_or_u), torch.sin(thetas_or_u)], dim=-1)
    if thetas_or_u.ndim == rhos.ndim + 1:
        tiny = torch.finfo(thetas_or_u.dtype).tiny
        return thetas_or_u / thetas_or_u.norm(dim=-1, keepdim=True).clamp_min(tiny)
    raise ValueError(
        f"thetas_or_u.ndim ({thetas_or_u.ndim}) must equal rhos.ndim ({rhos.ndim}) "
        f"or rhos.ndim + 1"
    )


@torch.no_grad()
def poincare_polar_to_poincare_cartesian(
    rhos: torch.Tensor, thetas_or_u: torch.Tensor
) -> torch.Tensor:
    """Compute z = tanh(rho/2) * direction inside the open unit disk.

    `tanh(rho/2)` saturates to 1.0 in float64 for `rho >= ~36`. The scale is
    clamped below 1 by one ulp so the strict invariant `||z|| < 1` holds for
    arbitrarily large `rho`.
    """
    direction = _polar_direction(rhos, thetas_or_u)
    scale = torch.tanh(rhos / 2)
    one_minus_eps = 1.0 - torch.finfo(scale.dtype).eps
    scale = scale.clamp(max=one_minus_eps)
    return scale.unsqueeze(-1) * direction


@torch.no_grad()
def poincare_polar_to_lorentz_cartesian(
    rhos: torch.Tensor, thetas_or_u: torch.Tensor
) -> torch.Tensor:
    """Compute `z = (cosh rho, sinh rho * direction)`.

    The spatial vector is rescaled so its norm-squared equals `cosh(rho)^2 - 1`
    (the algebraic target) under the test's `sum(z[1:]^2)` reduction, ensuring
    the Minkowski invariant `-z[0]^2 + sum(z[1:]^2) + 1` cancels at the float
    precision of `cosh^2`.
    """
    direction = _polar_direction(rhos, thetas_or_u)
    cosh = torch.cosh(rhos)
    target_sq = (cosh * cosh - 1.0).clamp_min(0.0)
    current_sq = (direction * direction).sum(-1, keepdim=True).clamp_min(
        torch.finfo(direction.dtype).tiny
    )
    scale = (target_sq.unsqueeze(-1) / current_sq).sqrt()
    spatial = direction * scale
    return torch.cat([cosh.unsqueeze(-1), spatial], dim=-1)


@torch.no_grad()
def poincare_cartesian_to_poincare_polar(
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a Poincare-disk Cartesian point into polar `(rho, theta_or_u)`.

    For `d == 2` the second return is a scalar `theta = atan2(y, x)`; for `d >= 3`
    it is the unit direction `z / ||z||` of shape `(..., d)`.

    Args:
        z (`torch.Tensor` of shape `(..., d)`):
            Poincare-disk Cartesian coordinates satisfying `||z|| < 1`.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`:
            - `rhos` of shape `(...)`: hyperbolic radial coordinate `rho = 2 * atanh(||z||)`.
            - `theta_or_u`: angle of shape `(...)` if `d == 2`, else unit vector of shape `(..., d)`.
    """
    tiny = torch.finfo(z.dtype).tiny
    norms = z.norm(dim=-1)
    rhos = 2.0 * torch.atanh(norms.clamp(max=1.0 - torch.finfo(z.dtype).eps))
    d = z.shape[-1]
    if d == 2:
        theta = torch.atan2(z[..., 1], z[..., 0])
        return rhos, theta
    direction = z / norms.unsqueeze(-1).clamp_min(tiny)
    return rhos, direction


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


@torch.no_grad()
def lorentz_cartesian_to_poincare_polar(
    z_lorentz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Lorentz-Cartesian to polar `(rho, theta_or_u)`.

    `rho = arccosh(z[0])` recovers the radial geodesic distance to the origin; the
    angular part is `atan2(y, x)` in `d == 2` and the normalized spatial vector in
    `d >= 3`.

    Args:
        z_lorentz (`torch.Tensor` of shape `(..., d + 1)`):
            Ambient Lorentz-Cartesian coordinates with `z[0] >= 1`.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`:
            - `rhos` of shape `(...)`.
            - `theta_or_u`: scalar angle of shape `(...)` for `d == 2`, else unit
              vector of shape `(..., d)`.
    """
    tiny = torch.finfo(z_lorentz.dtype).tiny
    rhos = torch.acosh(z_lorentz[..., 0].clamp_min(1.0))
    spatial = z_lorentz[..., 1:]
    d = spatial.shape[-1]
    if d == 2:
        theta = torch.atan2(spatial[..., 1], spatial[..., 0])
        return rhos, theta
    norms = spatial.norm(dim=-1)
    direction = spatial / norms.unsqueeze(-1).clamp_min(tiny)
    return rhos, direction


@torch.no_grad()
def lorentz_cartesian_to_poincare_cartesian(z_lorentz: torch.Tensor) -> torch.Tensor:
    """Stereographic projection from the upper hyperboloid to the Poincare disk.

    The map is `z -> z[1:] / (1 + z[0])`.

    Args:
        z_lorentz (`torch.Tensor` of shape `(..., d + 1)`):
            Ambient Lorentz-Cartesian coordinates with `z[0] >= 1`.

    Returns:
        `torch.Tensor` of shape `(..., d)`: Poincare-disk Cartesian coordinates.
    """
    spatial = z_lorentz[..., 1:]
    t = z_lorentz[..., 0]
    denom = (1.0 + t).unsqueeze(-1).clamp_min(torch.finfo(z_lorentz.dtype).tiny)
    return spatial / denom


@torch.no_grad()
def sphere_polar_to_cartesian(
    phis: torch.Tensor, thetas_or_u: torch.Tensor
) -> torch.Tensor:
    """Convert spherical polar `(phi, theta_or_u)` to Cartesian on `S^d`.

    `phi` is the colatitude from the north pole `e_1`; the second argument is a
    scalar azimuth for `d == 2` (so spatial dim is `(d,)`) or a unit vector on
    `S^{d-1}` for `d >= 3`.

    Args:
        phis (`torch.Tensor` of shape `(...)`):
            Colatitude in `[0, pi]`.
        thetas_or_u (`torch.Tensor`):
            Azimuthal angle of shape `(...)` (if `d == 2`) or unit vector of shape
            `(..., d)` (if `d >= 3`).

    Returns:
        `torch.Tensor` of shape `(..., d + 1)`: ambient Cartesian coordinates on
        the unit sphere `S^d`.
    """
    direction = _polar_direction(phis, thetas_or_u)
    return torch.cat(
        [
            torch.cos(phis).unsqueeze(-1),
            torch.sin(phis).unsqueeze(-1) * direction,
        ],
        dim=-1,
    )


@torch.no_grad()
def sphere_cartesian_to_polar(
    z_sphere: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Cartesian on `S^d` to polar `(phi, theta_or_u)`.

    Inverse of [`sphere_polar_to_cartesian`].

    Args:
        z_sphere (`torch.Tensor` of shape `(..., d + 1)`):
            Cartesian coordinates on the unit sphere.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`:
            - `phis` of shape `(...)`: colatitude in `[0, pi]`.
            - `theta_or_u`: azimuthal angle of shape `(...)` for `d == 2`, else
              unit vector of shape `(..., d)`.
    """
    tiny = torch.finfo(z_sphere.dtype).tiny
    phis = torch.acos(z_sphere[..., 0].clamp(-1.0, 1.0))
    spatial = z_sphere[..., 1:]
    d = spatial.shape[-1]
    if d == 2:
        theta = torch.atan2(spatial[..., 1], spatial[..., 0])
        return phis, theta
    norms = spatial.norm(dim=-1)
    direction = spatial / norms.unsqueeze(-1).clamp_min(tiny)
    return phis, direction


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def _gruet_radial(ts: torch.Tensor, d: int, kappa: int) -> torch.Tensor:
    """Unified Gruet radial sampler for hyperbolic (kappa=-1) or spherical (kappa=+1).

    Hyperbolic uses log-space `arccosh(v^2 + (1-v^2)*cosh(s))` to keep precision when
    `cosh(s)` would overflow; spherical clamps the argument to `[-1, 1]` for `arccos`.
    """
    if ts.numel() == 0:
        return torch.empty_like(ts)
    rate = ((d - 1) ** 2) * ts / 8.0
    ns = torch.poisson(rate).to(torch.int64)
    ss = ts.sqrt() * FreeBinaryHyperbolicHeatKernel.sample_chi(2 * ns + d + 1, ts.dtype)
    vs = torch.rand_like(ts)
    if kappa == 1:
        arg = vs.square() + (1.0 - vs.square()) * torch.cos(ss)
        return torch.acos(arg.clamp(-1.0, 1.0))
    if kappa != -1:
        raise ValueError(f"kappa must be -1 or +1; got {kappa}")
    # Hyperbolic: arccosh(v^2 + (1-v^2)*cosh(s)) via log-space to avoid cosh overflow.
    tiny = torch.finfo(ss.dtype).tiny
    ln2 = float(torch.log(torch.tensor(2.0, dtype=ss.dtype)).item())
    log_cosh_s = torch.where(ss > 30.0, ss - ln2, torch.log(torch.cosh(ss).clamp_min(1.0)))
    log_a = torch.log((1.0 - vs.square()).clamp_min(tiny)) + log_cosh_s
    log_b = torch.log(vs.square().clamp_min(tiny))
    log_arg = torch.logaddexp(log_a, log_b).clamp_min(0.0)
    inner = (1.0 - torch.exp(-2.0 * log_arg)).clamp_min(0.0)
    return log_arg + torch.log1p(inner.sqrt())


class FreeBinaryHyperbolicHeatKernel:
    """Closed-form free hyperbolic heat kernel and bridge on `H^2` (d=2).

    Implements Gruet's series representation specialized to the disk: a Poisson
    count `n ~ Poisson(t / 8)`, a chi draw `s = sqrt(t) * chi(2n + 3)`, and a
    uniform mixing variable `v` jointly realize
    `rho = arccosh(v^2 + (1 - v^2) cosh(s))` distributed as the radial marginal
    of `H^2` Brownian motion at time `t`. The azimuthal angle is then sampled
    from the conditional Poisson kernel `(cosh rho - sinh rho cos theta)^{-1}`.

    All methods are `@staticmethod` and run under `torch.no_grad()`; they accept
    a `ts` batch of heat times of shape `(batch_size,)` and an optional
    `output_coord` in `{Coordinate.POLAR, Coordinate.CARTESIAN}` selecting the
    return geometry. `d=2` outputs from [`FreeHyperbolicHeatKernel`] dispatch
    here to preserve bit-exact equivalence.
    """

    @staticmethod
    @torch.no_grad()
    def poincare_polar_to_lorentz_cartesian(
        rhos: torch.FloatTensor,
        thetas: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Convert `(rho, theta)` on `H^2` to Lorentz-Cartesian coordinates.

        Specialized `d == 2` analogue of the module-level
        [`poincare_polar_to_lorentz_cartesian`] that consumes a scalar angle.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`):
                Hyperbolic radial coordinate.
            thetas (`torch.FloatTensor` of shape `(batch_size,)`):
                Azimuthal angle.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 3)`: Lorentz-Cartesian
            coordinates `(cosh rho, sinh rho * cos theta, sinh rho * sin theta)`.
        """
        sinh_r = torch.sinh(rhos)
        return torch.stack(
            [torch.cosh(rhos), sinh_r * thetas.cos(), sinh_r * thetas.sin()],
            dim=-1,
        )

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
            POLAR: `(rhos, thetas)`. CARTESIAN: Poincare disk `z` of shape `(B, 2)`.
        """
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * FreeBinaryHyperbolicHeatKernel.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
        if output_coord == Coordinate.CARTESIAN:
            return poincare_polar_to_poincare_cartesian(ps, thetas)
        return ps, thetas

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
                Either `Coordinate.POLAR` (returns `(rho, theta)`) or
                `Coordinate.CARTESIAN` (returns Lorentz-Cartesian coords).

        Returns:
            POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.POLAR
        )
        if output_coord == Coordinate.POLAR:
            return rhos, thetas
        _check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return poincare_polar_to_lorentz_cartesian(rhos, thetas)

    @staticmethod
    @torch.no_grad()
    def binary_poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        """Sample the `H^2` bridge endpoint conditioned on a target embedding.

        The free-heat-kernel angle is rotated by `atan2(e[1], e[0])` so the
        resulting sample is concentrated near the target direction `e =
        word_embedding[targets]`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices into `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, 2)`):
                Word-embedding table. Only the angular part is used.
            output_coord (`str`, *optional*, defaults to `Coordinate.POLAR`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: Poincare-disk coordinates of shape `(batch_size, 2)`.
        """
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.POLAR
        )
        e = word_embedding[targets].to(ts.dtype)
        target_angle = torch.atan2(e[..., 1], e[..., 0])
        thetas = thetas + target_angle
        if output_coord == Coordinate.CARTESIAN:
            return poincare_polar_to_poincare_cartesian(rhos, thetas)
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
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: Lorentz-Cartesian coords of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_poincare_bridge(
            ts=ts,
            targets=targets,
            word_embedding=word_embedding,
            output_coord=Coordinate.POLAR,
        )
        if output_coord == Coordinate.POLAR:
            return rhos, thetas
        _check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return poincare_polar_to_lorentz_cartesian(rhos, thetas)

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t,
        src: Optional[torch.FloatTensor] = None,
        dest: Optional[torch.FloatTensor] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed hyperbolic geodesic on `H^2` from `src` to `dest` at fraction `t`.

        The endpoints are accepted either as Lorentz-Cartesian tensors (`src`,
        `dest`) or as polar pairs (`src_radial`, `src_angular`, `dest_radial`,
        `dest_angular`); exactly one form per endpoint must be provided. The
        intrinsic distance uses the differential form
        `cosh d - 1 = <x - y, x - y>_L / 2` to avoid cancellation at large `d`.

        Args:
            t (`float` or `torch.Tensor`):
                Fraction along the geodesic, broadcastable to the batch shape.
            src (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
                Lorentz-Cartesian source.
            dest (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
                Lorentz-Cartesian destination.
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the source.
            src_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the destination.
            dest_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the destination.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.POLAR`. Defaults to the
                same form used to specify the source.

        Returns:
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, 3)`.
            POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, or
                if a polar input has `rho > _LORENTZ_RHO_MAX`.
        """
        if (src is not None and (src_radial is not None or src_angular is not None)) or (
            src is None and (src_radial is None or src_angular is None)
        ):
            raise ValueError(
                "Only accept one source, either src or (src_radial, src_angular)"
            )
        if (dest is not None and (dest_radial is not None or dest_angular is not None)) or (
            dest is None and (dest_radial is None or dest_angular is None)
        ):
            raise ValueError(
                "Only accept one destination, either dest or (dest_radial, dest_angular)"
            )

        if output_coord is None:
            output_coord = Coordinate.CARTESIAN if src is not None else Coordinate.POLAR

        if src is not None:
            x_amb = src
        else:
            _check_lorentz_rho_bound(src_radial, d=2)
            x_amb = poincare_polar_to_lorentz_cartesian(src_radial, src_angular)
        if dest is not None:
            y_amb = dest
        else:
            _check_lorentz_rho_bound(dest_radial, d=2)
            y_amb = poincare_polar_to_lorentz_cartesian(dest_radial, dest_angular)

        gamma = _geodesic_kernel(x_amb, y_amb, t, kappa=-1)

        if output_coord == Coordinate.CARTESIAN:
            return gamma
        return lorentz_cartesian_to_poincare_polar(gamma)


@torch.no_grad()
def _reflect_to_target(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Householder reflection mapping `e_1 -> x` applied to `u`. `x` must be unit."""
    d = u.shape[-1]
    e1 = torch.zeros(d, dtype=u.dtype, device=u.device)
    e1[0] = 1.0
    v = e1.unsqueeze(0) - x
    v_norm_sq = (v * v).sum(-1, keepdim=True)
    dot = (u * v).sum(-1, keepdim=True)
    reflected = u - 2.0 * dot / v_norm_sq.clamp_min(1e-300) * v
    return torch.where(v_norm_sq > 1e-24, reflected, u)


class FreeHyperbolicHeatKernel:
    """`d`-dimensional free hyperbolic heat kernel sampler.

    Generalizes [`FreeBinaryHyperbolicHeatKernel`] to `H^d` for `d >= 2`. The
    radial marginal is sampled via Gruet's representation
    `rho = arccosh(v^2 + (1 - v^2) cosh(s))` with `n ~ Poisson((d-1)^2 t / 8)`
    and `s = sqrt(t) * chi(2n + d + 1)`; the conditional angular distribution
    `(cosh rho - sinh rho <e_1, u>)^{-(d-1)}` is then sampled by one of three
    interchangeable methods.

    Angular methods (selected via the `method` argument):
        - `METHOD_BOOST` (default): apply the Lorentz boost
          `c -> (1 - 2 b (1 - c) / T) / T` to a uniform direction. Cheapest.
        - `METHOD_ICDF`: draw the polar coordinate `c` from its analytic CDF
          via a symmetric Beta proposal, then pad with a uniform direction
          on `S^{d-2}`.
        - `METHOD_VMF`: alias of `METHOD_ICDF` (algebraically equivalent
          under the `z <-> 1 - z` symmetry of the Beta proposal).

    For `d == 2` every public method dispatches to
    [`FreeBinaryHyperbolicHeatKernel`] for bit-exact parity. All methods are
    `@staticmethod` and run under `torch.no_grad()`.
    """

    METHOD_BOOST: str = "boost"
    METHOD_ICDF: str = "icdf"
    METHOD_VMF: str = "vmf"

    @staticmethod
    @torch.no_grad()
    def sample_radial(ts: torch.FloatTensor, d: int) -> torch.FloatTensor:
        """Sample the radial marginal `rho` of the `H^d` free heat kernel.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            d (`int`):
                Hyperbolic dimension; must be `>= 2`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size,)`: radial samples `rho`.
        """
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            return torch.empty_like(ts)
        return _gruet_radial(ts, d, kappa=-1)

    @staticmethod
    @torch.no_grad()
    def sample_angular(
        rhos: torch.FloatTensor,
        d: int,
        method: str = "boost",
    ) -> torch.FloatTensor:
        """Sample the conditional angular distribution given `rho`.

        For `d == 2` the result is a scalar angle of shape `(batch_size,)`. For
        `d >= 3` it is a unit vector on `S^{d-1}` of shape `(batch_size, d)`.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`):
                Radial coordinates returned by [`sample_radial`].
            d (`int`):
                Hyperbolic dimension; must be `>= 2`.
            method (`str`, *optional*, defaults to `"boost"`):
                One of `METHOD_BOOST`, `METHOD_ICDF`, `METHOD_VMF`. Ignored when
                `d == 2`.

        Returns:
            `torch.FloatTensor`: angle of shape `(batch_size,)` for `d == 2`,
            else unit direction of shape `(batch_size, d)`.
        """
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if rhos.numel() == 0:
            if d == 2:
                return torch.empty_like(rhos)
            return rhos.new_empty(0, d)

        if d == 2:
            us = torch.rand_like(rhos)
            return 2 * torch.atan((-rhos).exp() * torch.tan(torch.pi * (us - 0.5)))

        if method == FreeHyperbolicHeatKernel.METHOD_BOOST:
            return FreeHyperbolicHeatKernel._angular_boost(rhos, d)
        if method == FreeHyperbolicHeatKernel.METHOD_ICDF:
            return FreeHyperbolicHeatKernel._angular_icdf(rhos, d)
        if method == FreeHyperbolicHeatKernel.METHOD_VMF:
            return FreeHyperbolicHeatKernel._angular_vmf(rhos, d)
        raise ValueError(f"unknown method: {method!r}")

    @staticmethod
    @torch.no_grad()
    def free_poincare_heat_kernel(
        ts: torch.FloatTensor,
        d: int,
        method: str = "boost",
        output_coord: Optional[str] = None,
    ):
        """Sample from the free `H^d` heat kernel in Poincare-ball form.

        For `d == 2` this is exactly [`FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            d (`int`):
                Hyperbolic dimension; must be `>= 2`.
            method (`str`, *optional*, defaults to `"boost"`):
                Angular sampler — see [`sample_angular`].
            output_coord (`str`, *optional*, defaults to `Coordinate.POLAR`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: `(rhos, u)` where `rhos` has shape `(batch_size,)` and `u` is
            an angle of shape `(batch_size,)` for `d == 2` or a unit vector of
            shape `(batch_size, d)` for `d >= 3`.
            CARTESIAN: Poincare-ball coordinates of shape `(batch_size, d)`.
        """
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if d == 2:
            return FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
                ts, output_coord=output_coord
            )
        if ts.numel() == 0:
            if output_coord == Coordinate.CARTESIAN:
                return ts.new_empty(0, d)
            return torch.empty_like(ts), ts.new_empty(0, d)
        rhos = FreeHyperbolicHeatKernel.sample_radial(ts, d)
        u = FreeHyperbolicHeatKernel.sample_angular(rhos, d, method=method)
        if output_coord == Coordinate.CARTESIAN:
            return poincare_polar_to_poincare_cartesian(rhos, u)
        return rhos, u

    @staticmethod
    @torch.no_grad()
    def free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        d: int,
        method: str = "boost",
        output_coord: Optional[str] = None,
    ):
        """Sample from the free `H^d` heat kernel in Lorentz-Cartesian form.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            d (`int`):
                Hyperbolic dimension; must be `>= 2`.
            method (`str`, *optional*, defaults to `"boost"`):
                Angular sampler.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: same as [`free_poincare_heat_kernel`].
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, d + 1)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if d == 2:
            return FreeBinaryHyperbolicHeatKernel.binary_free_lorentz_heat_kernel(
                ts, output_coord=output_coord
            )
        if ts.numel() == 0:
            if output_coord == Coordinate.POLAR:
                return torch.empty_like(ts), ts.new_empty(0, d)
            return ts.new_empty(0, d + 1)
        rhos, u = FreeHyperbolicHeatKernel.free_poincare_heat_kernel(
            ts, d=d, method=method, output_coord=Coordinate.POLAR
        )
        if output_coord == Coordinate.POLAR:
            return rhos, u
        _check_lorentz_rho_bound(rhos, d, ts)
        return poincare_polar_to_lorentz_cartesian(rhos, u)

    @staticmethod
    @torch.no_grad()
    def poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        method: str = "boost",
        output_coord: Optional[str] = None,
    ):
        """Sample the `H^d` bridge endpoint conditioned on a target embedding.

        Draws a free-heat-kernel direction `u` and applies a Householder
        reflection mapping `e_1` to the normalized target embedding, so the
        resulting sample concentrates around `word_embedding[targets]`.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, d)`):
                Word-embedding table; `d = word_embedding.shape[1]` selects the
                hyperbolic dimension.
            method (`str`, *optional*, defaults to `"boost"`):
                Angular sampler.
            output_coord (`str`, *optional*, defaults to `Coordinate.POLAR`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: `(rhos, u_rotated)` matching [`free_poincare_heat_kernel`].
            CARTESIAN: Poincare-ball coordinates of shape `(batch_size, d)`.
        """
        d = word_embedding.shape[1]
        if d == 2:
            return FreeBinaryHyperbolicHeatKernel.binary_poincare_bridge(
                ts, targets, word_embedding, output_coord=output_coord
            )
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            if output_coord == Coordinate.CARTESIAN:
                return ts.new_empty(0, d)
            return torch.empty_like(ts), ts.new_empty(0, d)

        rhos, u = FreeHyperbolicHeatKernel.free_poincare_heat_kernel(
            ts, d=d, method=method, output_coord=Coordinate.POLAR
        )
        x = word_embedding[targets].to(ts.dtype)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        u_rotated = _reflect_to_target(u, x)
        if output_coord == Coordinate.CARTESIAN:
            return poincare_polar_to_poincare_cartesian(rhos, u_rotated)
        return rhos, u_rotated

    @staticmethod
    @torch.no_grad()
    def lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        method: str = "boost",
        output_coord: Optional[str] = None,
    ):
        """Lorentz-form `H^d` bridge endpoint conditioned on a target embedding.

        Lorentz analogue of [`poincare_bridge`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, d)`):
                Word-embedding table.
            method (`str`, *optional*, defaults to `"boost"`):
                Angular sampler.
            output_coord (`str`, *optional*, defaults to `Coordinate.CARTESIAN`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: `(rhos, u_rotated)` matching [`free_poincare_heat_kernel`].
            CARTESIAN: Lorentz-Cartesian coords of shape `(batch_size, d + 1)`.
            Raises `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        d = word_embedding.shape[1]
        if d == 2:
            return FreeBinaryHyperbolicHeatKernel.binary_lorentz_bridge(
                ts, targets, word_embedding, output_coord=output_coord
            )
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            if output_coord == Coordinate.POLAR:
                return torch.empty_like(ts), ts.new_empty(0, d)
            return ts.new_empty(0, d + 1)
        rhos, u_rotated = FreeHyperbolicHeatKernel.poincare_bridge(
            ts, targets, word_embedding, method=method, output_coord=Coordinate.POLAR
        )
        if output_coord == Coordinate.POLAR:
            return rhos, u_rotated
        _check_lorentz_rho_bound(rhos, d=d, ts=ts)
        return poincare_polar_to_lorentz_cartesian(rhos, u_rotated)

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t,
        src: Optional[torch.FloatTensor] = None,
        dest: Optional[torch.FloatTensor] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed hyperbolic geodesic on `H^d` from `src` to `dest` at fraction `t`.

        Endpoints are accepted either as Lorentz-Cartesian tensors of shape
        `(batch_size, d + 1)` or as polar pairs `(rho, theta_or_u)`; exactly one
        form per endpoint must be provided.

        Args:
            t (`float` or `torch.Tensor`):
                Fraction along the geodesic, broadcastable to the batch shape.
            src (`torch.FloatTensor` of shape `(batch_size, d + 1)`, *optional*):
                Lorentz-Cartesian source.
            dest (`torch.FloatTensor` of shape `(batch_size, d + 1)`, *optional*):
                Lorentz-Cartesian destination.
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Source radial coordinate.
            src_angular (`torch.FloatTensor`, *optional*):
                Source angular coordinate (scalar for `d == 2`, unit vector of
                shape `(batch_size, d)` for `d >= 3`).
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Destination radial coordinate.
            dest_angular (`torch.FloatTensor`, *optional*):
                Destination angular coordinate.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.POLAR`. Defaults to the
                same form used to specify the source.

        Returns:
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, d + 1)`.
            POLAR: `(rhos, theta_or_u)` matching
            [`lorentz_cartesian_to_poincare_polar`].

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, or
                if a polar input has `rho > _LORENTZ_RHO_MAX`.
        """
        if (src is not None and (src_radial is not None or src_angular is not None)) or (
            src is None and (src_radial is None or src_angular is None)
        ):
            raise ValueError(
                "Only accept one source, either src or (src_radial, src_angular)"
            )
        if (dest is not None and (dest_radial is not None or dest_angular is not None)) or (
            dest is None and (dest_radial is None or dest_angular is None)
        ):
            raise ValueError(
                "Only accept one destination, either dest or (dest_radial, dest_angular)"
            )

        if output_coord is None:
            output_coord = Coordinate.CARTESIAN if src is not None else Coordinate.POLAR

        if src is not None:
            x_amb = src
        else:
            _check_lorentz_rho_bound(src_radial, d=2)
            x_amb = poincare_polar_to_lorentz_cartesian(src_radial, src_angular)
        if dest is not None:
            y_amb = dest
        else:
            _check_lorentz_rho_bound(dest_radial, d=2)
            y_amb = poincare_polar_to_lorentz_cartesian(dest_radial, dest_angular)

        gamma = _geodesic_kernel(x_amb, y_amb, t, kappa=-1)

        if output_coord == Coordinate.CARTESIAN:
            return gamma
        return lorentz_cartesian_to_poincare_polar(gamma)

    # ------------------------------------------------------------------
    # Private angular samplers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _angular_boost(rhos: torch.FloatTensor, d: int) -> torch.FloatTensor:
        """Lorentz-boost angular sampler. Cheapest per-sample cost."""
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device
        tiny = torch.finfo(dtype).tiny
        u0 = torch.randn(B, d, dtype=dtype, device=device)
        u0 = u0 / u0.norm(dim=-1, keepdim=True).clamp_min(tiny)
        c0 = u0[..., 0]
        b = torch.exp(-2.0 * rhos)
        exp_neg_rho = torch.exp(-rhos)
        one_plus_c0 = 1.0 + c0
        one_minus_c0 = 1.0 - c0
        T_half = (one_plus_c0 + b * one_minus_c0).clamp_min(tiny)
        u = torch.empty(B, d, dtype=dtype, device=device)
        u[..., 0] = (1.0 - 2.0 * b * one_minus_c0 / T_half).clamp(-1.0, 1.0)
        u[..., 1:] = (2.0 * exp_neg_rho / T_half).unsqueeze(-1) * u0[..., 1:]
        return u

    @staticmethod
    @torch.no_grad()
    def _angular_icdf(rhos: torch.FloatTensor, d: int) -> torch.FloatTensor:
        """ICDF angular sampler: polar coord from symmetric Beta, then `S^{d-2}` pad."""
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device

        tiny = torch.finfo(dtype).tiny
        alpha = torch.full((B,), (d - 1) / 2.0, dtype=dtype, device=device)
        z = torch.distributions.Beta(alpha, alpha).sample()
        b = torch.exp(-2.0 * rhos)
        denom = (z + b * (1.0 - z)).clamp_min(tiny)
        c = (1.0 - 2.0 * b * (1.0 - z) / denom).clamp(-1.0, 1.0)

        w = torch.randn(B, d - 1, dtype=dtype, device=device)
        w = w / w.norm(dim=-1, keepdim=True).clamp_min(tiny)
        s = (1.0 - c.square()).clamp_min(0.0).sqrt()
        u = torch.empty(B, d, dtype=dtype, device=device)
        u[:, 0] = c
        u[:, 1:] = s.unsqueeze(-1) * w
        return u

    # `_angular_vmf` is algebraically `_angular_icdf` under z <-> 1-z symmetry of the
    # symmetric Beta proposal; keep as alias for API parity with the spec.
    _angular_vmf = _angular_icdf

class FreeSphericalHeatKernel:
    """`d`-dimensional free spherical heat kernel sampler on `S^d` (kappa = +1 dual).

    The kappa-flipped Gruet ansatz `cos phi = v^2 + (1 - v^2) cos(s)` is empirically
    correct only in the small-`t` regime; samplers refuse `max(t) > _SPHERE_T_MAX`
    via [`_check_sphere_t_bound`]. The angular distribution is rotationally
    symmetric around the polar axis, so [`sample_angular`] returns
    `Uniform(S^{d-1})` regardless of the `method` argument (kept for API parity
    with [`FreeHyperbolicHeatKernel`]).

    All methods are `@staticmethod` and run under `torch.no_grad()`.
    """

    METHOD_BOOST: str = "boost"
    METHOD_ICDF: str = "icdf"
    METHOD_VMF: str = "vmf"

    @staticmethod
    @torch.no_grad()
    def sample_radial(ts: torch.FloatTensor, d: int) -> torch.FloatTensor:
        """Sample the radial marginal `phi` (colatitude) of the `S^d` free heat kernel.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times; must satisfy `max(ts) <= _SPHERE_T_MAX`.
            d (`int`):
                Spherical dimension; must be `>= 2`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size,)`: colatitude samples in
            `[0, pi]`.

        Raises:
            ValueError: if `max(ts) > _SPHERE_T_MAX` or `d < 2`.
        """
        if d < 2:
            raise ValueError(f"FreeSphericalHeatKernel requires d >= 2; got d={d}")
        _check_sphere_t_bound(ts, d)
        if ts.numel() == 0:
            return torch.empty_like(ts)
        return _gruet_radial(ts, d, kappa=1)

    @staticmethod
    @torch.no_grad()
    def sample_angular(
        phis: torch.FloatTensor,
        d: int,
        method: str = "boost",
    ) -> torch.FloatTensor:
        """Azimuthal direction on S^{d-1}.

        By rotational symmetry of the free heat kernel around the polar axis, the
        azimuthal direction is `Uniform(S^{d-1})` independent of phi for all methods.
        The `method` argument is preserved for API parity with the hyperbolic class.
        """
        if d < 2:
            raise ValueError(f"FreeSphericalHeatKernel requires d >= 2; got d={d}")
        if method not in (FreeSphericalHeatKernel.METHOD_BOOST,
                          FreeSphericalHeatKernel.METHOD_ICDF,
                          FreeSphericalHeatKernel.METHOD_VMF):
            raise ValueError(f"unknown method: {method!r}")
        if phis.numel() == 0:
            if d == 2:
                return torch.empty_like(phis)
            return phis.new_empty(0, d)
        if d == 2:
            return (torch.rand_like(phis) - 0.5) * (2.0 * torch.pi)
        return _uniform_sphere(phis.shape[0], d, dtype=phis.dtype, device=phis.device)

    @staticmethod
    @torch.no_grad()
    def free_sphere_heat_kernel(
        ts: torch.FloatTensor,
        d: int,
        method: str = "boost",
        output_coord: Optional[str] = None,
    ):
        """Sample from the free `S^d` heat kernel.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times; must satisfy `max(ts) <= _SPHERE_T_MAX`.
            d (`int`):
                Spherical dimension; must be `>= 2`.
            method (`str`, *optional*, defaults to `"boost"`):
                Kept for API parity with [`FreeHyperbolicHeatKernel`]; ignored
                because the angular marginal is `Uniform(S^{d-1})` for every
                method.
            output_coord (`str`, *optional*, defaults to `Coordinate.POLAR`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: `(phis, u)` where `phis` has shape `(batch_size,)` and `u` is
            an azimuth of shape `(batch_size,)` for `d == 2` or a unit vector of
            shape `(batch_size, d)` for `d >= 3`.
            CARTESIAN: ambient sphere coordinates of shape `(batch_size, d + 1)`.
        """
        if d < 2:
            raise ValueError(f"FreeSphericalHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            if output_coord == Coordinate.CARTESIAN:
                return ts.new_empty(0, d + 1)
            if d == 2:
                return torch.empty_like(ts), torch.empty_like(ts)
            return torch.empty_like(ts), ts.new_empty(0, d)
        phis = FreeSphericalHeatKernel.sample_radial(ts, d)
        u = FreeSphericalHeatKernel.sample_angular(phis, d, method=method)
        if output_coord == Coordinate.CARTESIAN:
            return sphere_polar_to_cartesian(phis, u)
        return phis, u

    @staticmethod
    @torch.no_grad()
    def sphere_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        method: str = "boost",
        output_coord: Optional[str] = None,
    ):
        """Sample the `S^d` bridge endpoint conditioned on a target embedding.

        The free-heat-kernel direction is rotated so that the polar axis `e_1`
        aligns with the normalized target embedding (via angle offset for
        `d == 2`, via Householder reflection for `d >= 3`).

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Vocabulary indices.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, d)`):
                Word-embedding table; `d = word_embedding.shape[1]`.
            method (`str`, *optional*, defaults to `"boost"`):
                Ignored — kept for API parity.
            output_coord (`str`, *optional*, defaults to `Coordinate.POLAR`):
                `Coordinate.POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            POLAR: `(phis, u_rotated)` matching [`free_sphere_heat_kernel`].
            CARTESIAN: ambient sphere coords of shape `(batch_size, d + 1)`.
        """
        d = word_embedding.shape[1]
        if d < 2:
            raise ValueError(f"FreeSphericalHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            if output_coord == Coordinate.CARTESIAN:
                return ts.new_empty(0, d + 1)
            if d == 2:
                return torch.empty_like(ts), torch.empty_like(ts)
            return torch.empty_like(ts), ts.new_empty(0, d)

        phis = FreeSphericalHeatKernel.sample_radial(ts, d)
        u = FreeSphericalHeatKernel.sample_angular(phis, d, method=method)
        x = word_embedding[targets].to(ts.dtype)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        if d == 2:
            target_angle = torch.atan2(x[..., 1], x[..., 0])
            u_rotated = u + target_angle
        else:
            u_rotated = _reflect_to_target(u, x)
        if output_coord == Coordinate.CARTESIAN:
            return sphere_polar_to_cartesian(phis, u_rotated)
        return phis, u_rotated

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t,
        src: Optional[torch.FloatTensor] = None,
        dest: Optional[torch.FloatTensor] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        """Constant-speed spherical geodesic (SLERP) on `S^d` from `src` to `dest`.

        Endpoints are accepted either as ambient Cartesian tensors of shape
        `(batch_size, d + 1)` or as polar pairs `(phi, theta_or_u)`; exactly one
        form per endpoint must be provided.

        Args:
            t (`float` or `torch.Tensor`):
                Fraction along the geodesic, broadcastable to the batch shape.
            src (`torch.FloatTensor` of shape `(batch_size, d + 1)`, *optional*):
                Ambient Cartesian source.
            dest (`torch.FloatTensor` of shape `(batch_size, d + 1)`, *optional*):
                Ambient Cartesian destination.
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Source colatitude `phi`.
            src_angular (`torch.FloatTensor`, *optional*):
                Source azimuth (scalar for `d == 2`, unit vector of shape
                `(batch_size, d)` for `d >= 3`).
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Destination colatitude.
            dest_angular (`torch.FloatTensor`, *optional*):
                Destination azimuth.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.POLAR`. Defaults to the
                same form used to specify the source.

        Returns:
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, d + 1)`.
            POLAR: `(phis, theta_or_u)` matching [`sphere_cartesian_to_polar`].

        Raises:
            ValueError: if neither or both forms of an endpoint are provided.
        """
        if (src is not None and (src_radial is not None or src_angular is not None)) or (
            src is None and (src_radial is None or src_angular is None)
        ):
            raise ValueError(
                "Only accept one source, either src or (src_radial, src_angular)"
            )
        if (dest is not None and (dest_radial is not None or dest_angular is not None)) or (
            dest is None and (dest_radial is None or dest_angular is None)
        ):
            raise ValueError(
                "Only accept one destination, either dest or (dest_radial, dest_angular)"
            )

        if output_coord is None:
            output_coord = Coordinate.CARTESIAN if src is not None else Coordinate.POLAR

        if src is not None:
            x_amb = src
        else:
            x_amb = sphere_polar_to_cartesian(src_radial, src_angular)
        if dest is not None:
            y_amb = dest
        else:
            y_amb = sphere_polar_to_cartesian(dest_radial, dest_angular)

        gamma = _geodesic_kernel(x_amb, y_amb, t, kappa=1)

        if output_coord == Coordinate.CARTESIAN:
            return gamma
        return sphere_cartesian_to_polar(gamma)

