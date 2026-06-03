# Derivation: `d`-dimensional hyperbolic heat kernel, bridge, and coordinate maps

Math reference for the `HyperbolicHeatKernel` (general `H^d`) class and the
general-`d` `GeoUtils` converters in `unigram/geo_bridge.py`. This consolidates
and completes the `H^d` sampling math; the model-level posterior/ELBO math lives
in `unigram/hyper_dm.md` (this note is the *sampling + coordinate* companion).

**Convention pinned by this note (the one intentional divergence from the
reference `hyper_bridge.py`):** the **free** heat kernel from the origin is
rotationally symmetric, so its angular part is **uniform on `S^{d-1}`**. The
target-conditioned **bridge** carries the Poisson-kernel direction. The reference
`FreeHyperbolicHeatKernel.free_*` instead folds the Poisson direction (centred at
`e_1`) into its "free" sampler; `geo_bridge` corrects this, exactly as it does for
`d == 2` (see `geo_bridge_ARCH.md`).

> ## ⚠️ Correctness finding (radial sampler, `d >= 3`)
>
> **The reference `hyper_bridge._gruet_radial` is correct for `d == 2` but WRONG
> for `d >= 3`.** It uses `n ~ Poisson((d-1)^2 t/8)`, `s = sqrt(t) chi(2n+d+1)`,
> `rho = arccosh(v^2 + (1-v^2) cosh s)`, `v ~ U(0,1)`. Verified two ways (§V):
> - **Short-time Euclidean limit.** As `t -> 0`, `H^d` is locally Euclidean, so
>   `rho -> sqrt(t) * chi_d` and `E[rho]/sqrt(t) -> E[chi_d]`. The sampler instead
>   gives `rho ~ sqrt(t) chi(d+1) sqrt(1-v^2)`, i.e. `E[rho]/sqrt(t) =
>   E[chi_{d+1}] * (pi/4)`, which equals `E[chi_d]` **only at `d = 2`**.
>   Measured `E[rho]/sqrt(t)`: d=2 1.2535 vs 1.2533 ✓; d=3 **1.4764 vs 1.5958 ✗**;
>   d=5 **1.8464 vs 2.1277 ✗**.
> - **Exact `H^3` marginal.** The closed form gives `pi(rho) ∝ rho sinh rho
>   e^{-rho^2/2t}` (mean 1.849 at `t=1`); the sampler gives 1.733. Disagree by ~6%.
>
> `hyper_dm.md` only ever derives the **2D** scheme; the `(d-1)^2 t/8` / `2n+d+1`
> generalization was an **unvalidated extrapolation**. §2.1 below gives the
> *correct* general-`d` radial law (derived from Grigor'yan–Noguchi's Millson
> recurrence and **verified** in §V against the exact `H^3` kernel + the Euclidean
> short-time limit). The binary `d == 2` path is unaffected and remains correct.

---

## 0. Notation and models

`H^d` is `d`-dimensional hyperbolic space (constant curvature `-1`). We use three
charts:

- **Polar** `(rho, u)`: `rho = d_H(z, O) >= 0` is the geodesic distance to the
  origin `O`; `u in S^{d-1}` (unit vector, shape `(..., d)`) is the direction.
  For `d == 2`, `u` may be written as a scalar angle `theta` with
  `u = (cos theta, sin theta)`; for `d >= 3` we keep `u` as a unit vector.
- **Poincare ball** `B^d = { z in R^d : ||z|| < 1 }`, point `z`.
- **Lorentz hyperboloid** in `R^{1,d}`: `z = (z_0, z_{1:d})` with the Minkowski
  form `<z, z>_L = -z_0^2 + ||z_{1:d}||^2 = -1` and `z_0 >= 1`. The origin is
  `O = (1, 0, ..., 0)`.

Minkowski inner product: `<x, y>_L = -x_0 y_0 + <x_{1:}, y_{1:}>`. For points on
the hyperboloid, `<x, y>_L = -cosh d_H(x, y)`, so `<x, y>_L <= -1`.

---

## 1. Coordinate conversions (general `d`)

These are pure bijections between charts; no randomness. They back the
`GeoUtils` converters of the same names.

### 1.1 Direction normalization — `_polar_direction(u)`
The sampled direction may carry residual norm error from upstream RNG. Define the
projection to `S^{d-1}`:
```
dir(u) = u / max(||u||, tiny).
```
For `d == 2` the binary path instead lifts a scalar angle,
`dir(theta) = (cos theta, sin theta)` (handled by `_binary_polar_direction`).

### 1.2 Polar -> Poincare ball — `hyperbolic_polar_to_poincare_cartesian`
The Poincare radius of a point at hyperbolic distance `rho` is `tanh(rho/2)`
(half-angle form of `rho = 2 artanh ||z||`). Hence
```
z = tanh(rho/2) * dir(u),     ||z|| = tanh(rho/2) < 1.
```
`tanh(rho/2)` saturates to `1.0` in float64 for `rho >~ 36`; clamp the scale to
`1 - eps` so `||z|| < 1` strictly for arbitrarily large `rho`.

### 1.3 Polar -> Lorentz — `hyperbolic_polar_to_lorentz_cartesian`
The geodesic at distance `rho` in direction `u` lifts to
```
z = (cosh rho,  sinh rho * dir(u)),
```
since `<z,z>_L = -cosh^2 rho + sinh^2 rho ||dir(u)||^2 = -cosh^2 rho + sinh^2 rho = -1`.
**Numerical note:** rather than `sinh rho * dir(u)` directly, rescale the spatial
block so its norm-squared equals the algebraic target `cosh^2 rho - 1` exactly
(the reference's trick), so the on-manifold defect `<z,z>_L + 1` cancels at the
float precision of `cosh^2`:
```
spatial = dir(u) * sqrt( (cosh^2 rho - 1) / ||dir(u)||^2 ),   z = (cosh rho, spatial).
```
Gate `rho <= _LORENTZ_RHO_MAX = 20` on the cartesian path (see §6).

### 1.4 Lorentz -> polar — `lorentz_cartesian_to_hyperbolic_polar`
Invert §1.3:
```
rho = arccosh( clamp(z_0, min=1) ),     u = z_{1:d} / max(||z_{1:d}||, tiny).
```
For `d == 2` the spatial block is 2-D and `u` collapses to `theta = atan2(z_2, z_1)`
(the binary converter), but the general-`d` form returns the unit vector `u`.

### 1.5 Poincare <-> Lorentz (stereographic)
Lift `B^d -> hyperboloid`:
```
z_lor = ( (1 + ||z||^2)/(1 - ||z||^2),  2 z / (1 - ||z||^2) ),
```
and its inverse (projection from `-O`):
```
z_poin = z_lor[1:] / (1 + z_lor[0]).
```
Both are `d`-generic (the existing `poincare_cartesian_to_lorentz_cartesian` /
`lorentz_cartesian_to_poincare_cartesian` already implement them; they need no
change for `d >= 3`).

---

## 2. Free heat kernel on `H^d`

Brownian motion on `H^d` started at `O`, run for time `t`. Because the generator
is rotation-invariant about `O`, the law at time `t` factorizes in polar
coordinates:
```
q_t(rho, u) = P_H(rho; t) * Uniform_{S^{d-1}}(u).            (free)
```
i.e. **`u` is uniform on `S^{d-1}`, independent of `rho`.** This is the corrected
convention used here.

### 2.1 Radial marginal `P_H(rho; t)` — the correct target
With the hyperbolic volume element `dV = sinh^{d-1} rho , drho , dsigma(u)`, the
radial law is the density
```
pi(rho) ∝ sinh^{d-1} rho * P_H(rho; t),      rho >= 0,
```
where `P_H(.; t)` is the `H^d` heat kernel of Brownian motion (generator `½ Δ`).
Two anchors pin it unambiguously (used for verification in §V):

- **Short-time (Euclidean) limit.** As `t -> 0`, curvature is negligible and
  `rho -> sqrt(t) * chi_d` (distance of `R^d` BM at time `t`), so
  `E[rho]/sqrt(t) -> E[chi_d] = sqrt(2) Gamma((d+1)/2)/Gamma(d/2)`.
- **`H^3` closed form (`½ Δ`):** `P_H(rho;t) = (2 pi t)^{-3/2} (rho/sinh rho)
  e^{-t/2 - rho^2/2t}`, so `pi(rho) ∝ rho sinh rho e^{-rho^2/2t}` (Davies 1989;
  Grigor'yan 2009). This is elementary; all odd `d` are elementary via the
  Millson recursion, even `d` are McKean integrals.

**`d == 2` sampler (validated).** Gruet's series specialized to the disk:
```
n  ~ Poisson( t / 8 )
s  = sqrt(t) * chi(2n + 3)
v  ~ Uniform(0, 1)
rho = arccosh( v^2 + (1 - v^2) cosh s )
```
`chi(k)` via `chi^2(k) ~ Gamma(k/2, rate=1/2)` then `sqrt` (`GeoUtils.sample_chi`);
the `arccosh` step is taken in log-space when `cosh s` would overflow.

**`d >= 3` correct law (derived from Grigor'yan–Noguchi, verified in §V).** The
reference's `(d-1)^2 t/8` rate + `2n+d+1` dof extrapolation is **incorrect** (⚠️
callout, §V). Grigor'yan–Noguchi give the exact `H^d` heat kernel (generator `Δ`,
`p_n(rho,t)`) by the **Millson recurrence**
```
p_{n+2}(rho,t) = - exp(-n t) / (2 pi sinh rho) * d/drho p_n(rho,t),     (G-N 1.8)
```
from the bases `p_1 ∝ e^{-rho^2/4t}` (the `R^1` Gaussian), `p_3 ∝ (rho/sinh rho)
e^{-t - rho^2/4t}` (1.6), and `p_2` = McKean integral (1.7). Converting to BM
(`½ Δ`, time `t`) is `t -> t/2` and, crucially, **every `rho`-independent factor
(Gaussian normalizer, the spectral shift `e^{-((n-1)/2)^2 t}`) cancels in the
normalized radial marginal**. Writing the curvature operator `O[f] = -f' / sinh rho`,
the marginal is therefore, up to normalization,
```
ODD  d:  m_d(rho) ∝ sinh^{d-1} rho * O^{(d-1)/2} [ exp(-rho^2 / 2t) ]
EVEN d:  m_d(rho) ∝ sinh^{d-1} rho * O^{(d-2)/2} [ p_2(rho, t/2) ]   (p_2 = McKean)
```
`O^{(d-1)/2}` applies `(d-1)/2` curvature-recurrence steps to the Gaussian core.
For `d = 3`, `O^1[e^{-rho^2/2t}] = (rho/(2t sinh rho)) e^{-rho^2/2t}`, giving
`m_3 ∝ rho sinh rho e^{-rho^2/2t}` — the exact `H^3` marginal. **Sampler:** build
`m_d` on a `rho`-grid (operator steps by analytic recurrence for odd `d`, McKean
quadrature base for even `d`), then inverse-CDF. Verified in §V against the `H^3`
closed form (mean `1.8493` vs `1.8498` at `t=1`) and the Euclidean short-time
limit for `d = 3, 5, 7` (e.g. `d=7`: `2.5532` vs `2.5518`). The `d == 2` Gruet
sampler above remains the (validated) special case.

### 2.2 Angular part (free)
Uniform on `S^{d-1}`: draw `g ~ N(0, I_d)` and set `u = g / ||g||`
(`GeoUtils._uniform_sphere`). No `rho`-dependence.

---

## 3. Posterior / bridge: target-conditioned angle

The forward (bridge) law conditions the time-`t` point `z` on the eventual exit
target `x in H^d` (the normalized word embedding). Dropping the `x`-normalizer,
```
q_{t|inf}(z | x) ∝ q_t(z) * exp( (d-1) <x, z>_H ).
```
The likelihood is the **Poisson kernel**, which in polar coordinates is
```
exp( (d-1) <x, z>_H ) = ( (1 - ||z||^2) / ||x - z||^2 )^{d-1}
                      = ( cosh rho - sinh rho <x, u> )^{-(d-1)}.
```
So the joint posterior in `(rho, u)` is
```
pi(rho, u) ∝ [ P_H(rho; t) sinh^{d-1} rho ] * ( cosh rho - sinh rho <x, u> )^{-(d-1)}.
```
**Key fact (radial unchanged).** The angular factor integrates to a `rho`-independent
constant over `S^{d-1}` (it is harmonic measure w.r.t. uniform):
```
∫_{S^{d-1}} ( cosh rho - sinh rho <x, u> )^{-(d-1)} dsigma(u) = |S^{d-1}|   for all rho.
```
Therefore conditioning on `x` does **not** move the radial law — `rho` is sampled
from the same `P_H(rho; t)` as the free kernel (§2.1) — it only biases the angle:
```
pi(rho)   ∝ sinh^{d-1} rho * P_H(rho; t),
pi(u|rho) ∝ ( cosh rho - sinh rho <x, u> )^{-(d-1)}.        (concentrates u toward x)
```
The exponent `-(d-1) < 0` makes the density large where `cosh rho - sinh rho <x,u>`
is small, i.e. where `<x, u>` is large (`u` aligned with `x`). As `rho -> inf` the
angle concentrates on `x`.

---

## 4. Sampling the Poisson-kernel direction (Lorentz boost), then targeting

We sample the angular law for a **reference target `e_1`** (first axis), then map
`e_1 -> x` by an isometry of `S^{d-1}`.

### 4.1 Boost sampler (`method = "boost"`, the one implemented here)
Draw `u_0` uniform on `S^{d-1}` and let `c_0 = <e_1, u_0> = u_0[0]`. Apply the
hyperbolic boost of rapidity `rho` along `e_1`:
```
u[0]   = (c_0 cosh rho + sinh rho) / (cosh rho + c_0 sinh rho)        # boosted cosine
u[1:]  = u_0[1:] / (cosh rho + c_0 sinh rho)                          # transverse, scaled
```
This is a unit vector: `u[0]^2 + ||u[1:]||^2 = 1` (since
`||u_0[1:]||^2 = 1 - c_0^2`, the numerator collapses to `(cosh rho + c_0 sinh rho)^2`).
The boost is conformal on `S^{d-1}` with factor `1/(cosh rho + c_0 sinh rho)`, and
it pushes the uniform measure forward to the harmonic/Poisson measure
`(cosh rho - sinh rho u[0])^{-(d-1)}`. As `rho -> inf`, `u[0] -> 1`, i.e. `u -> e_1`.

The code form uses `b = e^{-2 rho}`, `T = (1 + c_0) + b (1 - c_0)`:
```
u[0]  = 1 - 2 b (1 - c_0) / T          # = (c_0 cosh rho + sinh rho)/(cosh rho + c_0 sinh rho)
u[1:] = (2 e^{-rho} / T) * u_0[1:]     # = u_0[1:]/(cosh rho + c_0 sinh rho)
```
which is the overflow-safe (`b, e^{-rho} in (0,1]`) rewrite of the formulas above.

For `d == 2` the boost reduces to the scalar inverse-CDF
`theta = 2 atan( e^{-rho} tan(pi (U - 1/2)) )` of the binary class.

### 4.2 Targeting by Householder reflection — `_reflect_to_target(u, x)`
With `x = word_embedding[targets]` normalized to `S^{d-1}`, the reflection that
maps `e_1 -> x` is the Householder transform across `v = e_1 - x`:
```
H(u) = u - 2 <u, v> / ||v||^2 * v,        v = e_1 - x.
```
`H` is an isometry of `S^{d-1}` with `H(e_1) = x`, so it carries the
`e_1`-centred Poisson sample to the `x`-centred one. Guard the degenerate
`x ~ e_1` case (`||v||^2 -> 0`) by returning `u` unchanged.

**Bridge, end to end:** `rho` from §2.1; `u` from §4.1 (centred at `e_1`); then
`u_rotated = H(u)`; then convert `(rho, u_rotated)` to the requested chart (§1).

---

## 5. Geodesic

Constant-speed geodesic between two hyperboloid points `x, y` at fraction
`t in [0, 1]` (the `_geodesic_kernel`, `kappa = -1`):
```
d = d_H(x, y) = arccosh(-<x, y>_L),   computed from
cosh d - 1 = <x - y, x - y>_L / 2     (cancellation-safe at large d),
gamma(t)   = [ sinh((1-t) d) x + sinh(t d) y ] / sinh d,
```
with a Euclidean-lerp fallback for `d < 1e-6`. `gamma(0) = x`, `gamma(1) = y`,
`<gamma, gamma>_L = -1`. Polar / Poincare inputs are lifted to the hyperboloid
first (§1), interpolated, then mapped back to the requested chart. This is the
same kernel used by the binary class; it is `d`-generic already.

---

## 6. Numerical guards

- `_LORENTZ_RHO_MAX = 20`: the Lorentz-cartesian conversion's on-manifold defect
  floats at order `cosh(rho) * eps`; beyond `rho = 20` (`cosh ~ 2.4e8`) the
  cartesian image is untrustworthy to `1e-6`. The cartesian paths gate on this;
  polar output is unbounded. Same constant/rationale as the binary class.
- `tanh(rho/2)` clamp to `1 - eps` keeps `||z|| < 1` strict (§1.2).
- Direction normalization clamps `||.||` by `tiny` (§1.1).
- **Large-`t` overflow (radial sampler):** the inverse-CDF marginal
  `sinh^{d-1}(rho) p_H(rho; t)` is formed in linear space, so it overflows float64
  once `rho ~ 709/(d-1)`, i.e. very large `t` (`E[rho] ~ (d-1) t / 2`). Stable across
  the diffusion regime (`t = O(1)`; verified to `t ~ 50`, `rho ~ 85` at `d = 4`); a
  log-space marginal would extend the range. Cartesian output is gated at `rho <= 20`.

---

## 7. Shape conventions for `HyperbolicHeatKernel` (geo_bridge)

Unlike the flat `(batch,)` layout of the binary class and the reference, the
`HyperbolicHeatKernel` methods use a **sequence layout** with embedding dimension
`d = embedding_size`:

| quantity | shape | invariant |
|---|---|---|
| heat times `ts` | `(batch_size,)` | `> 0` |
| radial `rhos` | `(batch_size, seq_len)` | `>= 0` |
| direction `u` / angular | `(batch_size, seq_len, d)` | `||u|| = 1` (on `S^{d-1}`) |
| Poincare-ball `z` | `(batch_size, seq_len, d)` | `||z|| < 1` |
| Lorentz `z` | `(batch_size, seq_len, d + 1)` | `z_0 >= 1`, `<z,z>_L = -1` |
| `targets` | `(batch_size, seq_len)` long | indices into `word_embedding` |
| `word_embedding` | `(vocab_size, d)` | direction `x = e/||e||` used |

`ts` of shape `(batch_size,)` broadcasts across `seq_len` (each token position
gets an independent draw at the same heat time). `d` is read from
`embedding_size` / `word_embedding.shape[-1]`.

---

## 8. Math -> method map

| method (geo_bridge `HyperbolicHeatKernel`) | math |
|---|---|
| `free_hyperbolic_heat_kernel` | §2: `rho` ~ Gruet (§2.1), `u` ~ Uniform `S^{d-1}` (§2.2) |
| `free_poincare_heat_kernel` / `free_lorentz_heat_kernel` | §2 then §1.2 / §1.3 |
| `poincare_bridge` / `lorentz_bridge` | §2.1 radial; §4.1 boost; §4.2 reflect to target; §1 convert |
| `geodesic` | §5 |
| `GeoUtils.hyperbolic_polar_to_poincare_cartesian` | §1.2 |
| `GeoUtils.hyperbolic_polar_to_lorentz_cartesian` | §1.3 |
| `GeoUtils.lorentz_cartesian_to_hyperbolic_polar` | §1.4 |
| `GeoUtils._polar_direction` | §1.1 |

**Divergence from reference (intentional):** `free_*` use the **uniform** angle
(§2.2); the reference folds the §4.1 Poisson direction into its `free_*`. The
bridge math (§3, §4) is identical to the reference up to the shape layout (§7).
The single angular sampler implemented is the **boost** method (§4.1).

---

## V. Verification (numerical + analytic)

All checks at float64; the binary `d == 2` paths are independently covered by
`unigram/tests/test_geo_bridge.py`.

1. **Angular law (boost), §4.1.** Analytic: the boost output is a unit vector
   (`u[0]^2 + ||u[1:]||^2 = 1`, shown in §4.1) and `<e_1, u>` is the standard
   Lorentz boost of `<e_1, u_0>`. The pushforward of uniform measure is the
   harmonic/Poisson measure `(cosh rho - sinh rho <x,u>)^{-(d-1)}` (Malecki et al.
   2006; matches `hyper_dm.md` §"Sample from Heat Kernel"). For `d == 2` it reduces
   to the verified scalar inverse-CDF; the binary bridge test confirms the first
   circular moment `E[cos] = tanh(rho/2)`.

2. **Radial law, §2.1 — and the `d >= 3` bug.** Measured `E[rho]/sqrt(t)` of the
   reference `_gruet_radial` at `t = 1e-3` (Euclidean regime), vs `E[chi_d]`:

   | `d` | `E[chi_d]` (target) | `_gruet_radial` | verdict |
   |---|---|---|---|
   | 2 | 1.2533 | 1.2535 | ✓ |
   | 3 | 1.5958 | 1.4764 | ✗ (= `E[chi_4] pi/4`) |
   | 5 | 2.1277 | 1.8464 | ✗ |

   Analytic root cause: at small `t`, `n = 0` w.h.p., `s = sqrt(t) chi(d+1)`,
   `arccosh(1 + (1-v^2)s^2/2) ~ s sqrt(1-v^2)`, so the sampler realizes
   `sqrt(t) chi(d+1) sqrt(1-v^2)` with `E = E[chi_{d+1}] (pi/4)`. The identity
   `E[chi_{d+1}] (pi/4) = E[chi_d]` holds iff `d = 2` (ratio `4/pi`). Against the
   exact `H^3` marginal `rho sinh rho e^{-rho^2/2t}` (mean `1.849` at `t=1`) the
   sampler gives `1.733` (~6% low). Hence the `d == 2` sampler is correct and the
   `d >= 3` extrapolation is not.

## References

- M. Gruet, *Semi-groupe du mouvement brownien hyperbolique*, Stochastics &
  Stochastics Reports 56 (1996) — original integral representation of the
  hyperbolic heat kernel.
- L.-J. Cheng, T.-Y. Wang, *Bessel bridge representation for the heat kernel in
  hyperbolic space*, arXiv:1701.01194 (2017) — exact probabilistic representation
  of the radial transition density; basis for a correct general-`d` sampler.
- N. Demni, *Hartman–Watson distribution and hyperbolic-like heat kernels*,
  arXiv:2103.10987 (2021) — Gruet's formula vs Millson induction, parity of `d`.
- A. H. Ould Moustapha, *Poisson semigroup and the Gruet formula for the heat
  kernels on spaces of constant curvature*, arXiv:2601.11596 (2026).
- J. Malecki et al., *Poisson kernel and Green function of the ball in real
  hyperbolic spaces*, arXiv:math/0512294 (2006) — `(cosh rho - sinh rho cos)^{-(d-1)}`
  Poisson kernel / harmonic measure (§3, §4).
- E. B. Davies, *Heat Kernels and Spectral Theory* (CUP, 1989); A. Grigor'yan,
  *Heat Kernel and Analysis on Manifolds* (AMS, 2009) — `H^3` closed form,
  `½ Δ` convention, volume element `sinh^{d-1} rho`.
- McKean (1970) — `H^2` heat-kernel integral (even-`d` radial law).
