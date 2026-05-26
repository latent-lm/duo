import torch
from typing import Tuple


class FreeBinaryHyperbolicHeatKernel:
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
    def binary_free_hyperbolic_heat_kernel(ts: torch.FloatTensor):
        ns = torch.poisson(ts/8).to(torch.int64)
        ss = ts.sqrt() * FreeBinaryHyperbolicHeatKernel.sample_chi(2*ns+3, ts.dtype)
        vs = torch.rand_like(ts)
        ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
        us = torch.rand_like(ts)
        thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
        return (ps, thetas)

    @staticmethod
    @torch.no_grad()
    def binary_hyperbolic_bridge(ts: torch.FloatTensor, targets: torch.LongTensor, V: int):
        rhos, thetas = FreeBinaryHyperbolicHeatKernel.binary_free_hyperbolic_heat_kernel(ts=ts)
        # thetas = thetas + (targets.to(ts.dtype) + 0.5) * (2 * torch.pi / V)
        e = word_embedding[targets]                              # (B, 2), unit vectors
        target_angle = torch.atan2(e[..., 1], e[..., 0])         # (B,) in (-π, π]
        thetas = thetas + target_angle
        return rhos, thetas


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
    #     "icdf"  -- grid-based inverse-CDF on c = <e_1, u>, then uniform w perp e_1
    #     "vmf"   -- Gamma-vMF mixture: lambda ~ Gamma(d-1, cosh rho), u ~ vMF(e_1, lambda sinh rho)
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
        arg = vs.square() + (1 - vs.square()) * torch.cosh(ss)
        arg = arg.clamp_min(1.0)
        return torch.acosh(arg)

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
    def hyperbolic_bridge(
        ts: torch.FloatTensor,
        targets: torch.LongTensor,
        word_embedding: torch.FloatTensor,
        method: str = "boost",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""
        Boundary-conditioned posterior sampler. Draws a free sample `(rho, u_free)` (with
        `u_free` conditioned on `x = e_1`), then rotates `u_free` so that `e_1 -> x_target`
        via a per-sample Householder reflection. The radial coordinate is unchanged. At
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
        u_rot = FreeHyperbolicHeatKernel._rotate_to_target(u, x)
        return rhos, u_rot

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
        u0 = torch.randn(B, d, dtype=rhos.dtype, device=rhos.device)
        u0 = u0 / u0.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        ch = torch.cosh(rhos)
        sh = torch.sinh(rhos)
        u0_0 = u0[..., 0]
        time_after = ch + sh * u0_0
        spatial_0 = sh + ch * u0_0
        u = u0.clone()
        u[..., 0] = spatial_0
        u = u / time_after.unsqueeze(-1)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        return u

    @staticmethod
    @torch.no_grad()
    def _angular_icdf(rhos: torch.FloatTensor, d: int) -> torch.FloatTensor:
        r"""Grid-based inverse-CDF on `c = <e_1, u>` against the angular marginal.

        The marginal density of `c` is
        `f(c) propto (1 - c^2)^{(d-3)/2} * (cosh rho - sinh rho * c)^{-(d-1)}` on `[-1, 1]`.
        Build the unnormalized CDF on a 1024-node uniform grid in log-space (to avoid
        overflow at large rho), invert via vectorized binary search, then pair `c` with
        `w ~ Unif(S^{d-2})` to assemble `u = (c, sqrt(1 - c^2) w)`. Falls back to
        `_angular_boost` for `rho > 30` where the integrand concentrates beyond grid
        resolution.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`).
            d (`int`): spatial dimension, `>= 3`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, d)` on `S^{d-1}`.
        """
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device

        large = rhos > 30.0
        small = rhos < 1e-3
        normal = ~(large | small)

        u_out = torch.empty(B, d, dtype=dtype, device=device)

        if large.any():
            u_out[large] = FreeHyperbolicHeatKernel._angular_boost(rhos[large], d)

        # Assemble `c` for small + normal samples; we'll attach w later for both.
        non_large = ~large
        if non_large.any():
            c_non_large = torch.empty(int(non_large.sum().item()), dtype=dtype, device=device)
            # local boolean masks restricted to the non_large block
            small_in_block = small[non_large]
            normal_in_block = normal[non_large]

            if small_in_block.any():
                alpha = torch.full(
                    (int(small_in_block.sum().item()),),
                    (d - 1) / 2.0,
                    dtype=dtype, device=device,
                )
                beta_dist = torch.distributions.Beta(alpha, alpha)
                c_non_large[small_in_block] = 2.0 * beta_dist.sample() - 1.0

            if normal_in_block.any():
                rhos_n = rhos[non_large][normal_in_block]
                Bn = rhos_n.shape[0]
                n_nodes = 1024
                grid = torch.linspace(-1.0, 1.0, n_nodes, dtype=dtype, device=device)
                one_minus_c2 = (1 - grid.square()).clamp_min(1e-300)
                log_factor1 = ((d - 3) / 2.0) * torch.log(one_minus_c2)
                ch = torch.cosh(rhos_n).unsqueeze(-1)
                sh = torch.sinh(rhos_n).unsqueeze(-1)
                arg = (ch - sh * grid.unsqueeze(0)).clamp_min(1e-300)
                log_density = log_factor1.unsqueeze(0) - (d - 1) * torch.log(arg)
                log_density = log_density - log_density.max(dim=-1, keepdim=True).values
                density = torch.exp(log_density)
                dgrid = grid[1:] - grid[:-1]
                cdf_inc = 0.5 * (density[:, 1:] + density[:, :-1]) * dgrid.unsqueeze(0)
                cdf = torch.cat(
                    [torch.zeros(Bn, 1, dtype=dtype, device=device),
                     torch.cumsum(cdf_inc, dim=-1)], dim=-1,
                )
                cdf = cdf / cdf[:, -1:].clamp_min(1e-300)

                q = torch.rand(Bn, dtype=dtype, device=device)
                idx = torch.searchsorted(cdf, q.unsqueeze(-1)).squeeze(-1).clamp(1, n_nodes - 1)
                idx_lo = idx - 1
                cdf_lo = torch.gather(cdf, 1, idx_lo.unsqueeze(-1)).squeeze(-1)
                cdf_hi = torch.gather(cdf, 1, idx.unsqueeze(-1)).squeeze(-1)
                c_lo = grid[idx_lo]
                c_hi = grid[idx]
                frac = (q - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(1e-300)
                c_n = (c_lo + frac * (c_hi - c_lo)).clamp(-1.0, 1.0)
                c_non_large[normal_in_block] = c_n

            n_nl = c_non_large.shape[0]
            w = torch.randn(n_nl, d - 1, dtype=dtype, device=device)
            w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-300)
            s = (1 - c_non_large.square()).clamp_min(0.0).sqrt()
            u_nl = torch.empty(n_nl, d, dtype=dtype, device=device)
            u_nl[:, 0] = c_non_large
            u_nl[:, 1:] = s.unsqueeze(-1) * w
            u_out[non_large] = u_nl

        u_out = u_out / u_out.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        return u_out

    @staticmethod
    @torch.no_grad()
    def _angular_vmf(rhos: torch.FloatTensor, d: int) -> torch.FloatTensor:
        r"""Gamma-vMF mixture: `lambda ~ Gamma(d-1, cosh rho)`, then `u ~ vMF(e_1, kappa)`
        with `kappa = lambda * sinh rho`, drawn via Wood (1994) rejection sampling.

        Wood's algorithm samples the cosine `w = <mu, u>` via a Beta proposal:
            b  = (d - 1) / (2*kappa + sqrt(4*kappa^2 + (d-1)^2))   # conjugate form, stable for large kappa
            x  = (1 - b) / (1 + b);  c0 = kappa * x + (d - 1) * log(1 - x^2)
            z  ~ Beta((d-1)/2, (d-1)/2);  w = (1 - (1+b) z) / (1 - (1-b) z)
            accept if log(U) <= kappa * w + (d - 1) * log(1 - x*w) - c0,  U ~ U(0, 1)
        Cap at 50 iterations; fall back to `_angular_boost` for unaccepted samples.

        Args:
            rhos (`torch.FloatTensor` of shape `(batch_size,)`).
            d (`int`): spatial dimension, `>= 3`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, d)` on `S^{d-1}`.
        """
        B = rhos.shape[0]
        dtype = rhos.dtype
        device = rhos.device

        ch = torch.cosh(rhos)
        sh = torch.sinh(rhos)
        conc = torch.full((B,), float(d - 1), dtype=dtype, device=device)
        lam = torch.distributions.Gamma(conc, ch).sample()
        kappa = (lam * sh).clamp_min(1e-300)

        dm1 = float(d - 1)
        b = dm1 / (2.0 * kappa + torch.sqrt(4.0 * kappa.square() + dm1 ** 2))
        x = (1.0 - b) / (1.0 + b)
        c0 = kappa * x + dm1 * torch.log((1.0 - x.square()).clamp_min(1e-300))

        w_out = torch.zeros(B, dtype=dtype, device=device)
        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        alpha = torch.full((B,), dm1 / 2.0, dtype=dtype, device=device)
        beta_dist = torch.distributions.Beta(alpha, alpha)

        for _ in range(50):
            if accepted.all():
                break
            z = beta_dist.sample()
            denom = (1.0 - (1.0 - b) * z).clamp_min(1e-300)
            w = (1.0 - (1.0 + b) * z) / denom
            log_u = torch.log(torch.rand(B, dtype=dtype, device=device).clamp_min(1e-300))
            test = kappa * w + dm1 * torch.log((1.0 - x * w).clamp_min(1e-300)) - c0
            new_accept = (~accepted) & (log_u <= test)
            w_out = torch.where(new_accept, w, w_out)
            accepted = accepted | new_accept

        u_out = torch.empty(B, d, dtype=dtype, device=device)

        if accepted.any():
            acc_idx = accepted
            n_acc = int(acc_idx.sum().item())
            v = torch.randn(n_acc, d - 1, dtype=dtype, device=device)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-300)
            w_acc = w_out[acc_idx].clamp(-1.0, 1.0)
            s = (1.0 - w_acc.square()).clamp_min(0.0).sqrt()
            u_acc = torch.empty(n_acc, d, dtype=dtype, device=device)
            u_acc[:, 0] = w_acc
            u_acc[:, 1:] = s.unsqueeze(-1) * v
            u_out[acc_idx] = u_acc

        if not accepted.all():
            unacc = ~accepted
            u_out[unacc] = FreeHyperbolicHeatKernel._angular_boost(rhos[unacc], d)

        u_out = u_out / u_out.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        return u_out

    @staticmethod
    @torch.no_grad()
    def _rotate_to_target(u: torch.FloatTensor, x: torch.FloatTensor) -> torch.FloatTensor:
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
