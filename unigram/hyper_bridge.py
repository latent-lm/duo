import torch
from typing import Tuple

@dataclass
class Geometry:
    POINCARE: str = "poincare"
    LORENTZ_POLAR: str = "lorentz_polar"
    LORENTZ_CARTESIAN: str = "lorentz_cartesian"

@dataclass
class Coordinate:
    POLAR: str = "polar"
    CARTESIAN: str = "cartesian"

class FreeBinaryHyperbolicHeatKernel:
    """
    
    """
    @staticmethod
    @torch.no_grad()
    def poincare_polar_to_lorentz_cartesian(
        rhos: torch.FloatTensor,
        thetas: torch.FloatTensor,
    ) -> torch.FloatTensor:
        sinh_r = torch.sinh(rhos)
        return torch.stack(
            [torch.cosh(rhos), sinh_r * thetas.cos(), sinh_r * thetas.sin()],
            dim=-1,
        )

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
    def sample_chi_old(ns, dtype=torch.float64):
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
        """
        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times, strictly positive. The dtype of `ts` is preserved throughout.
            output_coord (`str`, *optional*, defaults to `None`):
                One of `"polar"` or `"cartesian"`.

        Returns:
            `Tuple[torch.FloatTensor, torch.FloatTensor]`: A pair `(rhos, thetas)`:
                - `rhos` (`torch.FloatTensor` of shape `(batch_size,)`): hyperbolic distance
                  from the origin, non-negative, same dtype/device as `ts`.
                - For `d == 2`, `thetas` of shape `(batch_size,)`: wrapped-Cauchy angle in
                  `(-pi, pi]`, bit-exact to the binary closed-form sampler.
        """
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * FreeBinaryHyperbolicHeatKernel.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
        if output_coord == Coordinate.CARTESIAN:
            pass
            # TODO: Finish this
        else:
            return ps, thetas

    @staticmethod
    @torch.no_grad()
    def binary_free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel(ts=ts, output_coord=Coordinate.POLAR)
        if output_coord == Coordinate.POLAR:
            return rhos, thetas
        else:
            return FreeBinaryHyperbolicHeatKernel.poincare_polar_to_lorentz_cartesian(rhos=rhos, thetas=thetas)

    @staticmethod
    @torch.no_grad()
    def binary_poincare_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts=ts)
        # thetas = thetas + (targets.to(ts.dtype) + 0.5) * (2 * torch.pi / V)
        e = word_embedding[targets]                              # (B, 2), unit vectors
        target_angle = torch.atan2(e[..., 1], e[..., 0])         # (B,) in (-π, π]
        thetas = thetas + target_angle
        if output_coord == Coordinate.CARTESIAN:
            pass
            # TODO: Finish this
        else:
            return rhos, thetas

    @staticmethod
    @torch.no_grad()
    def binary_lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        # TODO: Finish this
        if output_coord == Coordinate.POLAR:
            pass
            # TODO: Finish this
        else:
            # TODO: Finish this

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t: torch.FloatTensor,
        src: Optional[torch.FloatTensor] = None,
        dest: Optional[torch.FloatTensor] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        if (src is not None and (src_radial is not None or src_angular is not None)) or (src is None and (src_radial is None or src_angular is None)):
            raise ValueError(f"Only accept one source, either src or (src_radial, src_angular)")
        if (dest is not None and (dest_radial is not None or dest_angular is not None)) or (dest is None and (dest_radial is None or dest_angular is None)):
            raise ValueError(f"Only accept one destination, either src or (dest_radial, dest_angular)")
        
        # Decide the cooridnate of the output
        if output_coord is None:
            if src is not None:
                output_coord = Coordinate.CARTESIAN
            else:
                output_coord = Coordinate.POLAR

        # TODO: Implement the geodesic calculation in a numerical stable and efficient way and convert to correct coordinate

class FreeHyperbolicHeatKernel:
    # Free hyperbolic heat kernel on H^d, starting from origin, in polar coordinates (rho, u).
    #
    # Radial (Gruet generalization):
    #     n   ~ Poisson((d-1)^2 * t / 8)
    #     s   = sqrt(t) * chi_{2n + d + 1}
    #     v   ~ Uniform(0, 1)
    #     rho = arccosh(v^2 + (1 - v^2) * cosh s)
    #
    # Angular (Poisson kernel posterior conditional on x = e_1):
    #     pi(u | rho, x=e_1) propto (cosh rho - sinh rho * <e_1, u>)^{-(d-1)},  u in S^{d-1}
    #
    # Three equivalent samplers (selected by `method`):
    #     "boost" -- Lorentz-boost a uniform S^{d-1} sample by rapidity rho along e_1
    #     "icdf"  -- Mobius transform of a Beta-on-c sample (exact inverse-CDF)
    #     "vmf"   -- Wood-style Beta envelope with x = tanh rho (acceptance = 1, exact)
    #
    # At d=2 every public entry point short-circuits to FreeBinaryHyperbolicHeatKernel for
    # bit-exact parity.
    """d-dimensional free hyperbolic heat kernel sampler with three angular variants."""

    METHOD_BOOST: str = "boost"
    METHOD_ICDF: str = "icdf"
    METHOD_VMF: str = "vmf"

    @staticmethod
    @torch.no_grad()
    def sample_radial(ts: torch.FloatTensor, d: int) -> torch.FloatTensor:
        r"""
        Sample the radial coordinate `rho` of the free hyperbolic heat kernel on `H^d` at
        heat time `ts`, via the Gruet generalization:

            n   ~ Poisson((d - 1)^2 * t / 8)
            s   = sqrt(t) * chi_{2n + d + 1}
            v   ~ Unif(0, 1)
            rho = arccosh(v^2 + (1 - v^2) * cosh(s))

        Reuses [`FreeBinaryHyperbolicHeatKernel.sample_chi`] for the chi draw.

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times, strictly positive. Output dtype/device match `ts`.
            d (`int`):
                Ambient hyperboloid spatial dimension. Must be `>= 2`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size,)`: hyperbolic distance from the origin,
            non-negative, same dtype/device as `ts`.

        Examples:

        ```python
        >>> import torch
        >>> from unigram.hyper_bridge import FreeHyperbolicHeatKernel
        >>> ts = torch.full((1024,), 1.0, dtype=torch.float64)
        >>> rho = FreeHyperbolicHeatKernel.sample_radial(ts, d=5)
        >>> rho.shape, (rho >= 0).all().item()
        (torch.Size([1024]), True)
        ```
        """
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            return torch.empty_like(ts)
        rate = ((d - 1) ** 2) * ts / 8.0
        ns = torch.poisson(rate).to(torch.int64)
        ss = ts.sqrt() * FreeBinaryHyperbolicHeatKernel.sample_chi(2 * ns + d + 1, ts.dtype)
        vs = torch.rand_like(ts)
        return FreeHyperbolicHeatKernel._stable_acosh_gruet(vs, ss)

    @staticmethod
    @torch.no_grad()
    def _stable_acosh_gruet(vs: torch.FloatTensor, ss: torch.FloatTensor) -> torch.FloatTensor:
        r"""Numerically stable `arccosh(v^2 + (1 - v^2) * cosh(s))` for large `s`.

        Uses the identity `arccosh(arg) = log(arg) + log1p(sqrt(1 - 1/arg^2))` evaluated in
        log-space so the `cosh(s)` factor does not overflow when `s` is large.

        Args:
            vs (`torch.FloatTensor`): uniform draws in `[0, 1]`, same shape as `ss`.
            ss (`torch.FloatTensor`): scaled chi draws, non-negative.

        Returns:
            `torch.FloatTensor`: `arccosh(v^2 + (1 - v^2) * cosh(s))`, same shape/dtype.
        """
        ln2 = float(torch.log(torch.tensor(2.0, dtype=ss.dtype)).item())
        log_cosh_s = torch.where(
            ss > 30.0,
            ss - ln2,
            torch.log(torch.cosh(ss).clamp_min(1.0)),
        )
        one_minus_v2 = (1.0 - vs.square()).clamp_min(0.0)
        log_one_minus_v2 = torch.log(one_minus_v2.clamp_min(torch.finfo(ss.dtype).tiny))
        log_a = log_one_minus_v2 + log_cosh_s  # log((1-v^2) cosh s)
        log_b = torch.log(vs.square().clamp_min(torch.finfo(ss.dtype).tiny))
        log_arg = torch.logaddexp(log_a, log_b)
        log_arg = log_arg.clamp_min(0.0)
        inner = (1.0 - torch.exp(-2.0 * log_arg)).clamp_min(0.0)
        return log_arg + torch.log1p(inner.sqrt())

    @staticmethod
    @torch.no_grad()
    def sample_angular(
        rhos: torch.FloatTensor,
        d: int,
        method: str = "boost",
    ) -> torch.FloatTensor:
        r"""
        Sample the angular coordinate conditional on the boundary point `x = e_1`:

            pi(u | rho, x=e_1) propto (cosh rho - sinh rho * <e_1, u>)^{-(d-1)}

        At `d == 2` the call uses the wrapped-Cauchy closed form
        `theta = 2 * atan(exp(-rho) * tan(pi * (u - 0.5)))` (same as
        [`FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel`]). At `d >= 3`
        the call dispatches to one of three statistically equivalent samplers.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`):
                Radial coordinates, non-negative. Dtype/device are preserved.
            d (`int`):
                Ambient hyperboloid spatial dimension. Must be `>= 2`.
            method (`str`, *optional*, defaults to `"boost"`):
                One of `"boost"`, `"icdf"`, `"vmf"`. Ignored when `d == 2`.

        Returns:
            `torch.FloatTensor`: For `d == 2`, scalar angles `theta` of shape `(batch_size,)`
            in `(-pi, pi]`. For `d >= 3`, unit vectors of shape `(batch_size, d)` on `S^{d-1}`.

        Examples:

        ```python
        >>> import torch
        >>> from unigram.hyper_bridge import FreeHyperbolicHeatKernel
        >>> rhos = torch.full((1024,), 1.0, dtype=torch.float64)
        >>> u = FreeHyperbolicHeatKernel.sample_angular(rhos, d=5, method="boost")
        >>> u.shape, torch.allclose(u.norm(dim=-1), torch.ones(1024, dtype=torch.float64))
        (torch.Size([1024, 5]), True)
        ```
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
    def free_hyperbolic_heat_kernel(
        ts: torch.FloatTensor,
        d: int,
        method: str = "boost",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""
        Sample from the free hyperbolic heat kernel on H^d at heat time `ts`, starting from
        the origin. Returns the sample in polar coordinates `(rho, u)`, where `rho` is the
        hyperbolic distance from the origin and `u` is the unit direction on `S^{d-1}` (for
        `d == 2` a scalar angle `theta` is returned instead, matching
        [`FreeBinaryHyperbolicHeatKernel`]).

        The radial coordinate is drawn from the d-dimensional Gruet generalization:

            n   ~ Poisson((d - 1)^2 * t / 8)
            s   = sqrt(t) * chi_{2n + d + 1}
            v   ~ Unif(0, 1)
            rho = arccosh(v^2 + (1 - v^2) * cosh(s))

        The angular coordinate (for `d >= 3`) is drawn from one of three equivalent samplers
        selected by `method`. At `d == 2` the call short-circuits to
        [`FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times, strictly positive. The dtype of `ts` is preserved throughout.
            d (`int`):
                Ambient hyperboloid spatial dimension. Must be `>= 2`.
            method (`str`, *optional*, defaults to `"boost"`):
                Angular sampler. One of `"boost"`, `"icdf"`, `"vmf"`. Ignored when `d == 2`.

        Returns:
            `Tuple[torch.FloatTensor, torch.FloatTensor]`: A pair `(rho, u_or_theta)`:
                - `rho` (`torch.FloatTensor` of shape `(batch_size,)`): hyperbolic distance
                  from the origin, non-negative, same dtype/device as `ts`.
                - For `d >= 3`, `u` of shape `(batch_size, d)`: unit vector on `S^{d-1}`
                  conditional on the boundary point `x = e_1`.
                - For `d == 2`, `theta` of shape `(batch_size,)`: wrapped-Cauchy angle in
                  `(-pi, pi]`, bit-exact to the binary closed-form sampler.

        Examples:

        ```python
        >>> import torch
        >>> from unigram.hyper_bridge import FreeHyperbolicHeatKernel
        >>> ts = torch.full((1024,), 1.0, dtype=torch.float64)
        >>> rho, u = FreeHyperbolicHeatKernel.free_hyperbolic_heat_kernel(ts, d=5, method="boost")
        >>> rho.shape, u.shape
        (torch.Size([1024]), torch.Size([1024, 5]))
        >>> torch.allclose(u.norm(dim=-1), torch.ones_like(rho), atol=1e-10)
        True
        ```
        """
        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if d == 2:
            return FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts)
        if ts.numel() == 0:
            return torch.empty_like(ts), ts.new_empty(0, d)
        rhos = FreeHyperbolicHeatKernel.sample_radial(ts, d)
        u = FreeHyperbolicHeatKernel.sample_angular(rhos, d, method=method)
        return rhos, u

    @staticmethod
    @torch.no_grad()
    def free_poincare_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        # TODO: Implement free heat kernel on Poincare disk model
    
    @staticmethod
    @torch.no_grad()
    def free_lorentz_heat_kernel(
        ts: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        # TODO: Implement free heat kernel on Lorentz model
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.free_poincare_heat_kernel(ts=ts, output_coord=Coordinate.POLAR)
        if output_coord == Coordinate.POLAR:
            return rhos, thetas
        else:
            return FreeBinaryHyperbolicHeatKernel.poincare_polar_to_lorentz_cartesian(rhos=rhos, thetas=thetas)

    @staticmethod
    @torch.no_grad()
    def geodesic(
        t: torch.FloatTensor,
        src: Optional[torch.FloatTensor] = None,
        dest: Optional[torch.FloatTensor] = None,
        src_radial: Optional[torch.FloatTensor] = None,
        src_angular: Optional[torch.FloatTensor] = None,
        dest_radial: Optional[torch.FloatTensor] = None,
        dest_angular: Optional[torch.FloatTensor] = None,
        output_coord: Optional[str] = None,
    ):
        if (src is not None and (src_radial is not None or src_angular is not None)) or (src is None and (src_radial is None or src_angular is None)):
            raise ValueError(f"Only accept one source, either src or (src_radial, src_angular)")
        if (dest is not None and (dest_radial is not None or dest_angular is not None)) or (dest is None and (dest_radial is None or dest_angular is None)):
            raise ValueError(f"Only accept one destination, either src or (dest_radial, dest_angular)")
        
        # Decide the cooridnate of the output
        if output_coord is None:
            if src is not None:
                output_coord = Coordinate.CARTESIAN
            else:
                output_coord = Coordinate.POLAR

        # TODO: Implement the geodesic calculation in a numerical stable and efficient way and convert to correct coordinate

    @staticmethod
    @torch.no_grad()
    def hyperbolic_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        method: str = "boost",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""
        Boundary-conditioned posterior sampler. Draws a free sample `(rho, u_free)` (with
        `u_free` conditioned on `x = e_1`), then applies a per-sample Householder reflection
        that maps `e_1 -> x_target`. The angular conditional density depends only on
        `<u, e_1>` (invariant under `O(d-1)` of the orthogonal complement), so the reflection
        produces the correct marginal even though it has determinant `-1`. The radial
        coordinate is unchanged. At
        `d == 2` the call short-circuits to
        [`FreeBinaryHyperbolicHeatKernel.binary_hyperbolic_bridge`].

        Args:
            ts (`torch.FloatTensor` of shape `(batch_size,)`):
                Heat times, strictly positive.
            targets (`torch.LongTensor` of shape `(batch_size,)`):
                Index of the boundary token, used to look up the boundary direction in
                `word_embedding`.
            word_embedding (`torch.FloatTensor` of shape `(vocab_size, d)`):
                Embedding table whose rows are read as ambient vectors and normalized to
                directions on `S^{d-1}`. At `d == 2` only `vocab_size` is used.
            method (`str`, *optional*, defaults to `"boost"`):
                Angular sampler. Ignored when `d == 2`.

        Returns:
            `Tuple[torch.FloatTensor, torch.FloatTensor]`: A pair `(rho, u_or_theta)`:
                - `rho` (`torch.FloatTensor` of shape `(batch_size,)`).
                - For `d >= 3`, `u_rotated` of shape `(batch_size, d)` on `S^{d-1}` with the
                  Householder reflection applied so that `e_1 -> x_target`.
                - For `d == 2`, `theta` of shape `(batch_size,)`, bit-exact to
                  `binary_hyperbolic_bridge`.

        Examples:

        ```python
        >>> import torch
        >>> from unigram.hyper_bridge import FreeHyperbolicHeatKernel
        >>> ts = torch.full((1024,), 1.0, dtype=torch.float64)
        >>> targets = torch.zeros(1024, dtype=torch.long)
        >>> word_embedding = torch.eye(7, 5, dtype=torch.float64)
        >>> rho, u = FreeHyperbolicHeatKernel.hyperbolic_bridge(ts, targets, word_embedding)
        >>> rho.shape, u.shape
        (torch.Size([1024]), torch.Size([1024, 5]))
        ```
        """
        d = word_embedding.shape[1]
        if d == 2:
            V = word_embedding.shape[0]
            return FreeBinaryHyperbolicHeatKernel.binary_hyperbolic_bridge(ts, targets, V)

        if d < 2:
            raise ValueError(f"FreeHyperbolicHeatKernel requires d >= 2; got d={d}")
        if ts.numel() == 0:
            return torch.empty_like(ts), ts.new_empty(0, d)

        rhos, u = FreeHyperbolicHeatKernel.free_hyperbolic_heat_kernel(ts, d=d, method=method)
        x = word_embedding[targets].to(ts.dtype)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        u_reflected = FreeHyperbolicHeatKernel._reflect_to_target(u, x)
        return rhos, u_reflected

    @staticmethod
    @torch.no_grad()
    def lorentz_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        output_coord: Optional[str] = None,
    ):
        # TODO: Finish this
        if output_coord == Coordinate.POLAR:
            pass
            # TODO: Finish this
        else:
            # TODO: Finish this

    # ------------------------------------------------------------------
    # Private angular samplers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _angular_boost(rhos: torch.FloatTensor, d: int) -> torch.FloatTensor:
        r"""Lorentz-boost a uniform `S^{d-1}` sample by rapidity `rho` along `e_1`.

        Sample `u0 ~ Unif(S^{d-1})`, treat `(1, u0)` as a unit-time-component 4-vector and
        boost along `e_1` by rapidity `rho`. The boosted time component
        `t' = cosh rho + sinh rho * u0[0]` normalizes the boosted spatial vector
        `(sinh rho + cosh rho * u0[0], u0[1:])` back onto the unit sphere. This produces a
        draw from `(cosh rho - sinh rho * c)^{-(d-1)}` on `S^{d-1}`.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`).
            d (`int`): spatial dimension, `>= 3`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, d)` on `S^{d-1}`.
        """
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device
        tiny = torch.finfo(dtype).tiny
        u0 = torch.randn(B, d, dtype=dtype, device=device)
        u0 = u0 / u0.norm(dim=-1, keepdim=True).clamp_min(tiny)
        c0 = u0[..., 0]
        # Stable Lorentz-boost: factor out e^rho so cosh/sinh do not overflow.
        # u_new[0] = 1 - 2 b (1 - c0) / ((1 + c0) + b (1 - c0)); preserves precision of 1-c.
        # u_new[i>=1] = 2 e^{-rho} u0[i] / ((1 + c0) + b (1 - c0)) where b = e^{-2 rho}.
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
        r"""Inverse-CDF on `c = <e_1, u>` against the angular marginal.

        Uses the Mobius-transform identity: if `c0 = 2 X - 1` with `X ~ Beta((d-1)/2, (d-1)/2)`
        (the marginal of a uniform direction on `S^{d-1}`), then
        `c = (cosh rho * c0 + sinh rho) / (cosh rho + sinh rho * c0)` has density
        proportional to `(1 - c^2)^{(d-3)/2} (cosh rho - sinh rho c)^{-(d-1)}` on `[-1, 1]`.
        Sampling `X` from the Beta distribution is an exact inverse-CDF draw, after which the
        Mobius map yields `c` analytically. Then `c` is paired with `w ~ Unif(S^{d-2})` so
        that `u = (c, sqrt(1 - c^2) w)`.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`).
            d (`int`): spatial dimension, `>= 3`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, d)` on `S^{d-1}`.
        """
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device

        tiny = torch.finfo(dtype).tiny
        alpha = torch.full((B,), (d - 1) / 2.0, dtype=dtype, device=device)
        z = torch.distributions.Beta(alpha, alpha).sample()
        # Stable Mobius transform: `c = 1 - 2 b (1 - z) / (z + b (1 - z))` with
        # `b = e^{-2 rho}`. Preserves precision of `1 - c` when `b` is tiny.
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

    @staticmethod
    @torch.no_grad()
    def _angular_vmf(rhos: torch.FloatTensor, d: int) -> torch.FloatTensor:
        r"""Wood-style Beta envelope tuned to the angular conditional.

        Wood (1994)'s vMF envelope is `(1 - x^2)/(1 - x w)^2` with a free tuning parameter
        `x`. For the target `(1 - w^2)^{(d-3)/2} (cosh rho - sinh rho w)^{-(d-1)}` the
        envelope reduces to the target itself when `x = tanh rho` (acceptance probability is
        identically `1`). Set `b = e^{-2 rho} = (1 - tanh rho)/(1 + tanh rho)` and use the
        Wood proposal `w = (1 - (1+b) z) / (1 - (1-b) z)` with `z ~ Beta((d-1)/2, (d-1)/2)`,
        which is exact without rejection. Pair with `v ~ Unif(S^{d-2})`.

        Note: under the symmetry `z <-> 1 - z` of `Beta((d-1)/2, (d-1)/2)`, this Mobius
        transform is algebraically identical to the one used by `_angular_icdf`. The two
        methods therefore produce identically distributed samples and consume the same RNG
        budget per call. Both are kept as separate entry points to expose the boost/icdf/vmf
        triple from the spec, but cross-method KS at d>=3 is by construction trivial.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`).
            d (`int`): spatial dimension, `>= 3`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, d)` on `S^{d-1}`.
        """
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device
        tiny = torch.finfo(dtype).tiny

        b = torch.exp(-2.0 * rhos)
        alpha = torch.full((B,), (d - 1) / 2.0, dtype=dtype, device=device)
        z = torch.distributions.Beta(alpha, alpha).sample()
        # Stable form of `(1 - (1+b)z)/(1 - (1-b)z) = 1 - 2 b z / (1 - z + b z)`; the latter
        # preserves precision of `1 - w` when `b` is tiny (large rho).
        denom = (1.0 - z + b * z).clamp_min(tiny)
        w = (1.0 - 2.0 * b * z / denom).clamp(-1.0, 1.0)

        v = torch.randn(B, d - 1, dtype=dtype, device=device)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
        s = (1.0 - w.square()).clamp_min(0.0).sqrt()
        u = torch.empty(B, d, dtype=dtype, device=device)
        u[:, 0] = w
        u[:, 1:] = s.unsqueeze(-1) * v
        return u

    @staticmethod
    @torch.no_grad()
    def _reflect_to_target(u: torch.FloatTensor, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""Per-sample Householder reflection mapping `e_1 -> x` applied to `u`.

        Use the Householder reflection with reflection axis `v = e_1 - x` (unit):
            H(v) e_1 = e_1 - 2 * ((1 - <e_1, x>) / (2 - 2 <e_1, x>)) * (e_1 - x) = x
        For the angular conditional posterior (which is invariant under O(d-1) rotations
        of the orthogonal complement of `{e_1, x}`), a reflection produces the correct
        marginal distribution since the density depends only on `<e_1, u>` after the map.
        Falls back to identity when `||e_1 - x|| < 1e-12`.

        Args:
            u (`torch.FloatTensor` of shape `(batch_size, d)`).
            x (`torch.FloatTensor` of shape `(batch_size, d)`): unit target direction.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, d)`.
        """
        d = u.shape[1]
        dtype = u.dtype
        device = u.device
        e1 = torch.zeros(d, dtype=dtype, device=device)
        e1[0] = 1.0
        v = e1.unsqueeze(0) - x
        v_norm_sq = (v * v).sum(-1, keepdim=True)
        dot = (u * v).sum(-1, keepdim=True)
        reflected = u - 2.0 * dot / v_norm_sq.clamp_min(1e-300) * v
        safe = v_norm_sq > 1e-24
        return torch.where(safe, reflected, u)
