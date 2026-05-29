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

    def _binary_polar_direction(
        thetas: torch.Tensor
    ) -> torch.Tensor:
        """Resolve scalar-theta (d=2) direction.

        Unit-vector inputs are renormalized so the Lorentz/sphere invariants are not
        polluted by residual norm error from upstream samplers.

        Args:
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): angles

        Returns:
            CARTESIAN: `(x, y)` on unit ball
        """
        return torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)

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
            rhos (`torch.FloatTensor` of shape `(batch_size,)`): hyperbolic radial
            thetas (`torch.FloatTensor` of shape `(batch_size,)`): angles

        Returns:
            CARTESIAN: `(x, y)`
        """
        direction = GeoUtils._binary_polar_direction(thetas=thetas)
        scale = torch.tanh(rhos / 2)
        one_minus_eps = 1.0 - torch.finfo(scale.dtype).eps
        scale = scale.clamp(max=one_minus_eps)
        return scale.unsqueeze(-1) * direction

    @staticmethod
    @torch.no_grad()
    def binary_hyperbolic_polar_to_lorentz_cartesian(
        rhos: torch.FloatTensor,
        thetas: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Convert `(rho, theta)` on `H^2` to Lorentz-Cartesian coordinates.

        Specialized `d == 2` analogue of the module-level
        [`binary_hyperbolic_polar_to_lorentz_cartesian`] that consumes a scalar angle.

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
    
    @torch.no_grad()
    def binary_lorentz_cartesian_to_hyperbolic_polar(
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Lorentz-Cartesian to polar `(rho, thetas)`.

        `rho = arccosh(z[0])` recovers the radial geodesic distance to the origin; the
        angular part is `atan2(y, x)` in `d == 2`

        Args:
            z (`torch.Tensor` of shape `(..., 2)`):
                Ambient Lorentz-Cartesian coordinates with `z[0] >= 1`.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                - `rhos` of shape `(...)`.
                - `thetas`: scalar angle of shape `(...)` for `d == 2`.
        """
        d = z.shape[-1]
        if d != 2:
            raise ValueError(f"Cartesian dimenstion should be 2, not {d}.")
        theta = torch.atan2(z[..., 1], z[..., 0])
        return rhos, theta

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
    def lorentz_cartesian_to_poincare_cartesian(z: torch.Tensor) -> torch.Tensor:
        """Convert Lorentz-Cartesian to Poincare-disk-Cartesian via stereographic lift.

        The map is `z -> ((1 + ||z||^2) / (1 - ||z||^2), 2 z / (1 - ||z||^2))`, taking
        `B^d` into the upper hyperboloid in `R^{1, d}`.

        Args:
            z (`torch.Tensor` of shape `(..., d)`):
                Lorentz Cartesian coordinates.

        Returns:
            `torch.Tensor` of shape `(..., d + 1)`: ambient Poincare-disk-Cartesian coordinates
            satisfying `-z[0]^2 + sum(z[1:]^2) = -1`.
        """
        pass
        # TODO: Convert lorentz cartesian to Poincare disk cartesian

class FreeBinaryHyperbolicHeatKernel(GeoUtils):
    """Closed-form free hyperbolic heat kernel and bridge on `H^2` (d=2).

    Implements Gruet's series representation specialized to the disk: a Poisson
    count `n ~ Poisson(t / 8)`, a chi draw `s = sqrt(t) * chi(2n + 3)`, and a
    uniform mixing variable `v` jointly realize
    `rho = arccosh(v^2 + (1 - v^2) cosh(s))` distributed as the radial marginal
    of `H^2` Brownian motion at time `t`. The azimuthal angle is then sampled
    from the conditional Poisson kernel `(cosh rho - sinh rho cos theta)^{-1}`.

    All methods are `@staticmethod` and run under `torch.no_grad()`; they accept
    a `ts` batch of heat times of shape `(batch_size,)` and an optional
    `output_coord` in `{Coordinate.HYPERBOLIC_POLAR, Coordinate.CARTESIAN}` selecting the
    return geometry. `d=2` outputs from [`FreeHyperbolicHeatKernel`] dispatch
    here to preserve bit-exact equivalence.
    """

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
    def binary_free_hyperbolic_heat_kernel(
        ts: torch.FloatTensor,
    ):
        r"""Sample (rho, theta) from the free hyperbolic heat kernel on H^2.
        rho is the radial, which is also the area betweeen the hyperbolic curve and projection line.
        thetas is the angular, which is preserved in both Poincare and Lorentz model

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`): heat times.
            output_coord (`str`, *optional*): `"polar"` (default) or `"cartesian"`.

        Returns:
            HYPERBOLIC_POLAR: `(rhos, thetas)`
        """
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * HyperBridge.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        rhos = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = us / ( 2 * torch.pi())
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
            HYPERBOLIC_POLAR: `(rhos, thetas)`. CARTESIAN: Poincare disk `z` of shape `(B, 2)`.
        """
        rhos, thetas = binary_free_hyperbolic_heat_kernel(ts=ts)
        if output_coord == Coordinate.CARTESIAN:
            return binary_hyperbolic_polar_to_poincare_cartesian(ps, thetas)
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
                Either `Coordinate.HYPERBOLIC_POLAR` (returns `(rho, theta)`) or
                `Coordinate.CARTESIAN` (returns Lorentz-Cartesian coords).

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, 3)`. Raises
            `ValueError` if `max(rho) > _LORENTZ_RHO_MAX`.
        """
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, thetas
        _check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)

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
            output_coord (`str`, *optional*, defaults to `Coordinate.HYPERBOLIC_POLAR`):
                `Coordinate.HYPERBOLIC_POLAR` or `Coordinate.CARTESIAN`.

        Returns:
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.
            CARTESIAN: Poincare-disk coordinates of shape `(batch_size, 2)`.
        """
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(
            ts=ts, output_coord=Coordinate.HYPERBOLIC_POLAR
        )
        e = word_embedding[targets].to(ts.dtype)
        target_angle = torch.atan2(e[..., 1], e[..., 0])
        thetas = thetas + target_angle
        if output_coord == Coordinate.CARTESIAN:
            return binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)
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
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_poincare_bridge(
            ts=ts,
            targets=targets,
            word_embedding=word_embedding,
            output_coord=Coordinate.HYPERBOLIC_POLAR,
        )
        if output_coord == Coordinate.HYPERBOLIC_POLAR:
            return rhos, thetas
        _check_lorentz_rho_bound(rhos, d=2, ts=ts)
        return binary_hyperbolic_polar_to_lorentz_cartesian(rhos, thetas)

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
        """Constant-speed hyperbolic geodesic on `H^2` from `src` to `dest` at fraction `t`.

        The endpoints are accepted either as Lorentz-Cartesian tensors (`src`,
        `dest`) or as polar pairs (`src_radial`, `src_angular`, `dest_radial`,
        `dest_angular`); exactly one form per endpoint must be provided. The
        intrinsic distance uses the differential form
        `cosh d - 1 = <x - y, x - y>_L / 2` to avoid cancellation at large `d`.

        Args:
            t (`float` or `torch.Tensor`):
                Fraction along the geodesic, broadcastable to the batch shape.
            src_cartesian (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
                Lorentz-Cartesian source.
            dest_cartesian (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
                Lorentz-Cartesian destination.
            cartesian_model (str)`, *optional*):
                The local chart of the manifold of the Cartesian coordinate
            src_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the source.
            src_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the source.
            dest_radial (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar radial coordinate of the destination.
            dest_angular (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                Polar angle of the destination.
            output_coord (`str`, *optional*):
                `Coordinate.CARTESIAN` or `Coordinate.HYPERBOLIC_POLAR`. Defaults to the
                same form used to specify the source.

        Returns:
            CARTESIAN: `torch.FloatTensor` of shape `(batch_size, 3)`.
            HYPERBOLIC_POLAR: tuple `(rhos, thetas)` each of shape `(batch_size,)`.

        Raises:
            ValueError: if neither or both forms of an endpoint are provided, or
                if a polar input has `rho > _LORENTZ_RHO_MAX`.
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
            output_coord = Coordinate.CARTESIAN if src is not None else Coordinate.HYPERBOLIC_POLAR

        if src is not None:
            if cartesian_model == Geometry.POINCARE:
                x_amb = poincare_cartesian_to_lorentz_cartesian(z=src)
            elif cartesian_model == Geometry.LORENTZ:
                x_amb = src
            else:
                raise ValueError(f"cartesian_model should be ({Geometry.POINCARE}, {Geometry.LORENTZ}), not {cartesian_model}.")
        else:
            _check_lorentz_rho_bound(src_radial, d=2)
            x_amb = binary_hyperbolic_polar_to_lorentz_cartesian(rhos=src_radial, thetas=src_angular)
        if dest is not None:
            y_amb = dest
        else:
            _check_lorentz_rho_bound(dest_radial, d=2)
            y_amb = binary_hyperbolic_polar_to_lorentz_cartesian(rhos=dest_radial, thetas=dest_angular)
        
        interpolate = _geodesic_kernel(x_amb, y_amb, t, kappa=-1)

        if output_coord == Coordinate.CARTESIAN:
            if cartesian_model == Geometry.Lorentz:
                return interpolate
            elif cartesian_model == Geometry.Poincare:
                return lorentz_cartesian_to_poincare_cartesian(z=interpolate)
            else:
                raise ValueError(f"cartesian_model, {cartesian_model}, is not supported, only support ({Geometry.Lorentz}, {Geometry.Poincare}).")
        return binary_lorentz_cartesian_to_hyperbolic_polar(interpolate)