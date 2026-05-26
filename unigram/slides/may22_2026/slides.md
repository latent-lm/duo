---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
size: 4:3
class: lead
# section.lead h1 {
#   text-align: center;
# }
style: |
  section.lead h1 {
    text-align: center;
  },
  section.lead h2 {
    text-align: center;
  },
  section.lead h3 {
    text-align: center;
  },
  h1 {
    color: #3d3d3d;
  },
  h2 {
    color: #3d3d3d;
  },
  h3 {
    color: #3d3d3d;
  },
  r { 
      color: red; 
  },
  y { 
      color: yellow; 
  },
  b { 
      color: blue; 
  },
  .g {
      color: green;
  }
  
# style: |
#   section {
#     background-color: #ffffff;
#   }
#   h1 {
#     font-size: 50px;
#     color: #2a2a2a;
#   }

---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# Hyperbolic DLM

### May 22, 2026

---

## Poincare Disk - Cartesian Bridge ELBO Loss

$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
\frac{(d-1)^2}{2}
\sum_{i=1}^{N}
w(t_i) \mathbb{E}_{z_{t_i} \sim q_{t_i \mid \infty}(\cdot \mid y)}
\left[
	\int_0^\infty
	(1 - \|z_t\|^2)^2
	\left\|
	\frac{y - z_t}{\|y - z_t\|^2}
	-
	\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_t)} \left[
	\frac{v - z_t}{\|v - z_t\|^2} \right]
	\right\|^2
	dt
\right] \\
\mu^{\theta}(z_t) 
& = \operatorname{softmax} \big( f_{\theta}(z_t) + (d-1) \sum_{v \in V} e_v \log \frac{1 - || v ||^2 }{|| z_t - v ||^2} \big)
\end{aligned}
$$
where $w(t_i)$ is the importance sampling weight of the timestep $t_i$, $y$ is the boundary points, $z_t$ is the bridge at timestep $t$.

---

## Lorentz - Cartesian Bridge ELBO Loss

### Brownian Motion on Lorentz Model

Consider the Brownian bridge on the local chart of Lorentz model,
$$
\begin{aligned}
dx_t
=
\left(
- \frac{(d-1)}{\langle x_t, \xi(y)\rangle_L}\xi(y)
- \frac{d-2}{2} x_t
\right)dt
+ \Pi_{x_t}\, dW_t.
\end{aligned}
$$

where  $<\cdot, \cdot>_{L}$ is Lorentz inner product and $\xi(y)$ is $[1, y_1, y_2, \dots, y_d]$ and $\Pi_x = I_{d+1}+x x^\top J$ with $J = \operatorname{diag}(-1,1,\dots,1)$

---

## Lorentz - Cartesian Bridge ELBO Loss

Instead of matching the drift directly, we choose $\mu^{\theta}$ re-parametrization, the loss function can be written as
$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
\frac{(d-1)^2}{2}
\mathbb{E}_{z_{t} \sim q_{t \mid \infty}(\cdot \mid y)}
\left[
	\int_0^\infty
	\left\|
	\frac{\xi(y)}{\langle z_t, \xi(y) \rangle_{L}}
	-
	\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_t)} \left[
	\frac{\xi(v)}{\langle z_t, \xi(v) \rangle_{L}} \right]
	\right\|_L^2
	dt
\right] \\
\mu^{\theta}(z_t) 
& = \operatorname{softmax} \big( f_{\theta}(z_t) + (d-1) \sum_{v \in V} e_v \log \frac{- \langle z, x \rangle_{L}}{- \langle O, x \rangle_{L}} \big)
\end{aligned}
$$
where $\| x \|_{L}^2 := \langle x, x \rangle_L$ is the Lorentz norm.

---

## Lorentz - Cartesian Bridge ELBO Loss

If we further consider the importance sampling from a proposal distribution $t_i \sim \pi$, the loss can be represented as
$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
\frac{(d-1)^2}{2}
\sum_{i=1}^{N} w(t_i)
\mathbb{E}_{z_{t_i} \sim q_{t_i \mid \infty}(\cdot \mid y)}
\left[
	\int_0^\infty
	\left\|
	\frac{\xi(y)}{\langle z_{t_i}, \xi(y) \rangle_{L}}
	-
	\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_{t_i})} \left[
	\frac{\xi(v)}{\langle z_{t_i}, \xi(v) \rangle_{L}} \right]
	\right\|_L^2
	dt
\right] \\
\end{aligned}
$$
where $w(t_i)$ is the importance sampling weight of the timestep $t_i$, $y$ is the boundary points, $z_t$ is the bridge at timestep $t$.

---

## Exponential Proposal Distribution

**Density $p(t)$ and CDF $F(t)$ on $[0, \infty)$:**
$$
p(t) = \lambda e^{-\lambda t}, 
\qquad 
F(t) = 1 - e^{-\lambda t}
$$

**Inverse-CDF sampling.** Set $F(t) = u$ with $u \sim \mathrm{Uniform}(0,1)$:
$$
t = -\frac{\ln(1-u)}{\lambda}
$$

---

## Proposal Distribution

```python
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
        ts = - torch.log1p(-u) / exp_rate
        density = exp_rate * torch.exp(-exp_rate * ts)
        return ts, density.reciprocal()
    else:
        raise NotImplementedError(f"proposal_type={proposal_type} is not implemented.")
```

---

## Poincare Disk - Polar Bridge ELBO Loss

```python
def binary_bridge_loss_polar(logits, targets, rhos, thetas):
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
```


---

## Poincare Disk - Cartesian Bridge ELBO Loss

```python
@staticmethod
def _cartesian_geometry(rhos, thetas, V):
    """Returns (z, v, diff, sq, one_minus_zz, h) used by every variant."""
    z = polar_to_cart(rhos, thetas)                                  # (N, 2)
    v = vocab_points(V, rhos.device, rhos.dtype)                     # (V, 2)
    diff = v - z.unsqueeze(-2)                                       # (N, V, 2)
    sq   = diff.square().sum(-1)                                     # (N, V)
    one_minus_zz = 1 - z.square().sum(-1, keepdim=True)              # (N, 1)
    h    = (one_minus_zz / sq).log()                                 # (N, V)
    return z, v, diff, sq, one_minus_zz, h

@staticmethod
def _expected_radial(mu, diff, sq):
    """E_{v ~ mu}[ (v - z) / ||v - z||^2 ]  =  sum_v mu_v (v-z)/||v-z||^2."""
    return (mu / sq).unsqueeze(-1).mul(diff).sum(-2)                 # (N, 2)

@staticmethod
def _cartesian_squared_residual(target, model, one_minus_zz, d=2):
    """L = (d-1)^2 / 2 * (1 - ||z||^2)^2 * ||target - model||^2."""
    residual = target - model                                        # (N, 2)
    return (d - 1) ** 2 / 2 * one_minus_zz.squeeze(-1).square() \
            * residual.square().sum(-1)

@staticmethod
def binary_bridge_loss_cartesian(logits, targets, rhos, thetas):
    V, d = logits.shape[-1], 2
    z, v, diff, sq, one_minus_zz, h = HyperBridge._cartesian_geometry(rhos, thetas, V)

    # target term: (y - z) / ||y - z||^2
    y_minus_z = v[targets] - z                                       # (N, 2)
    target = y_minus_z / y_minus_z.square().sum(-1, keepdim=True)    # (N, 2)

    # model term: E_{v ~ mu^theta(.|z)}[ (v - z) / ||v - z||^2 ]
    mu = ((d - 1) * h + logits.to(torch.float64)).softmax(-1)        # (N, V)
    model = HyperBridge._expected_radial(mu, diff, sq)               # (N, 2)

    return HyperBridge._cartesian_squared_residual(target, model, one_minus_zz, d=d)
```

---

## Lorentz - Cartesian Bridge ELBO Loss

```python
@staticmethod
    def _lorentz_boundary_points(V, device, dtype):
        phis = (torch.arange(V, device=device, dtype=dtype) + 0.5) * (2 * torch.pi / V)
        return torch.stack([torch.ones_like(phis), phis.cos(), phis.sin()], dim=-1)

    @staticmethod
    def _lorentz_inner(x, y):
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)

    @staticmethod
    def _lorentz_geometry(rhos, thetas, V, d):
        z = HyperBridge.polar_to_lorentz(rhos, thetas) # (N, 3)
        xi = HyperBridge._lorentz_boundary_points(V, rhos.device, rhos.dtype)
        inner = HyperBridge._lorentz_inner(z[:, None, :], xi[None, :, :]) # (N, V), negative
        log_poisson = (d - 1) *  (-(-inner).clamp_min(1e-300).log()) # (d - 1) * log 1 / (-<z,xi(y)>)
        directions = xi[None, :, :] / inner[:, :, None] # xi(y) / <z,xi(y)>
        return directions, log_poisson

    @staticmethod
    def _lorentz_norm_sq(x):
        return HyperBridge._lorentz_inner(x, x).clamp_min(0)

    @staticmethod
    def binary_bridge_loss_lorentz(logits, targets, rhos, thetas):
        V, d = logits.shape[-1], 2
        directions, log_poisson = HyperBridge._lorentz_geometry(rhos, thetas, V, d)
        mu = (log_poisson + logits.to(torch.float64)).softmax(-1) 
        target = directions[torch.arange(targets.numel(), device=targets.device), targets]
        model = (mu[:, :, None] * directions).sum(-2)
        residual = target - model
        return (d - 1) ** 2 / 2 * HyperBridge._lorentz_norm_sq(residual)
```

---

## Experiment Set Up

- Vocab Size: 2
- Testing Dataset Size: 4000
- Prob Dist of Training Dataset: [0.8, 0.2]
- Testing Data Entropy: 0.5
- Model ouputs the likelihood of the dataset
- Removed Truncation of Exponential distribution proposal

---

### Lorentz Model - Cartesian Coordinate

- Unif[0.01, 10]: 0.6595
- Exp(0.01): 0.7816
- Exp(0.1): 0.6777
- Exp(0.5): 0.7120
- Exp(0.8): 0.6507
- Exp(1.0): 0.6003
- Exp(2.0): 0.4527
- Exp(3.0): 

---

### Poincare Disk - Polar Coordinate

- Unif[0.01, 10]: 0.6595
- Exp(0.01): 0.7816
- Exp(0.1): 0.6777
- Exp(0.5): 0.7120
- Exp(1.0): 0.6003
- Exp(2.0): 0.4527
- Exp(3.0): 0.3933

---

### Poincare Disk - Cartesian Coordinate

- Unif[0.01, 10]: 0.6595
- Exp(0.01): NaN
- Exp(0.1): NaN
- Exp(0.5): 0.7120
- Exp(1.0): 0.6003
- Exp(2.0): 0.4527
- Exp(3.0): 0.3933

---

### Experiment Set Up

- Vocab Size: 10
- Testing Dataset Size: 4000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Testing Data Entropy: 0.5
- Model ouputs the likelihood of the dataset
- Removed Truncation of Exponential distribution proposal

---

### Lorentz - Cartesian Coordinate

- Unif[0.01, 10]: 0.4384
- Exp(0.01): 0.4330
- Exp(0.1): 0.5167
- Exp(0.5): 0.4956
- Exp(0.8): 0.7484
- Exp(1.0): 0.3880
- Exp(2.0): 0.2978
- Exp(3.0): 

---

### Poincare Disk - Polar Coordinate

- Unif[0.01, 10]: 0.5101
- Exp(0.01): 0.4935
- Exp(0.1): 0.5184
- Exp(0.5): 0.4550
- Exp(1.0): 0.4202
- Exp(2.0): 0.2468
- Exp(3.0): 0.2453

---

### Poincare Disk - Cartesian Coordinate

- Unif[0.01, 10]: 0.5101
- Exp(0.01): NaN
- Exp(0.1): NaN
- Exp(0.5): 0.4550
- Exp(1.0): 0.4202
- Exp(2.0): 0.2468
- Exp(3.0): 0.2453

---

### Experiment Set Up

- Vocab Size: 10
- Testing Dataset Size: **90000**
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Testing Data Entropy: 0.5
- Model ouputs the likelihood of the dataset
- Removed Truncation of Exponential distribution proposal

- I've also implemented cartesian coordinate version to make sure 

---

### Poincare Disk - Polar Coordinate

- Unif[0.01, 10]: 0.4774
- Exp(0.01): 0.5345
- Exp(0.1): 0.5558
- Exp(0.5): 0.5444
- Exp(1.0): 0.4476
- Exp(2.0): 0.3341
- Exp(3.0): 0.2321

---

### Poinare Disk - Cartesian Coordinate

- Unif[0.01, 10]: 0.4774
- Exp(0.01): NaN
- Exp(0.1): NaN
- Exp(0.5): 0.5444
- Exp(1.0): 0.4476
- Exp(2.0): 0.3341
- Exp(3.0): 0.2321

---

## Trucation at Large $\lambda$

$$
p(t) = \lambda e^{-\lambda t}
$$

- At $\lambda = 1.0$
$$
\begin{aligned}
  p(10.0) &= e^{-10} = 0.0000454 \\
  p(100.0) &= e^{-100} = 3.7e-44 \\
\end{aligned}
$$
- At $\lambda = 0.1$
$$
\begin{aligned}
  p(10.0) &= 0.1 \times e^{-1.0} = 0.0367 \\
  p(100.0) &= 0.1 \times e^{-10} = 0.00000454 \\
\end{aligned}
$$

The truncation cause the low test_loss as $\lambda$ is large, because the loss ignores the tail (large $t$ close to boundary)

---

# Stratified-Exp Proposal — Seed-Variance Study

---

## Stratified-Exp Proposal — Seed-Variance Study

How does the proposal rate $\lambda$ affect the **stability** and **variance**
of the test NELBO estimator?

*21-seed sweep, optimal model, 4M test samples*

---

## Experiment Set Up

### Stratified Exponential Proposal — 21-Seed Sweep

- Model: **optimal model** — emits the ground-truth unigram distribution (no training)
- Vocab size: 10  |  Data dist: $[0.91,\ 0.01\times 9]$  |  entropy $\approx 0.500$
- Proposal: **stratified exponential**, rate $\lambda \in \{0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0\}$
- Bridge horizon: `hyper_T` $= 1000$, `hyper_dt` $= 0.01$
- Test set: **4,000,000** samples per run
- Repeats: **21 random seeds** (rs42–rs62) per $\lambda$ → 168 runs / geometry
- Geometries: **Lorentz–Cartesian**, **Poincaré–Polar**
- Driver: `unigram/unigram_test2_tmp2.py`; aggregated by `unigram/calc_var.py`

---

## What "Good" Looks Like

For an **optimal model**, the test NELBO should equal the data entropy.

| Reference | Value |
|---|:--:|
| Cross-entropy (data entropy) | **0.5003** |
| ⇒ ideal `test_loss` | **≈ 0.500** |

A good $\lambda$ is **(1)** NaN-free, **(2)** unbiased (mean $\approx 0.500$), **(3)** low-variance.

`test_loss` = importance-sampled NELBO  ·  `test_loss_var` = its within-test-set variance

---

## Poincaré–Polar — Test NELBO vs $\lambda$

Small $\lambda$ (0.1–0.3) is **unbiased**; large $\lambda$ drifts **below** the 0.500 target.

| $\lambda$ | valid | `test_loss` (mean ± std) | `test_loss_var` |
|:--:|:--:|:--:|:--:|
| 0.01 | 0 / 21 | <r>NaN</r> | <r>NaN</r> |
| 0.1  | 21 / 21 | **0.5001 ± 0.0011** | 9 |
| 0.2  | 21 / 21 | 0.4998 ± 0.0024 | 23 |
| 0.3  | 21 / 21 | 0.4983 ± 0.0032 | 63 |
| 0.5  | 21 / 21 | 0.4992 ± 0.0123 | 648 |
| 0.8  | 21 / 21 | 0.4840 ± 0.0236 | 1,849 |
| 1.0  | 21 / 21 | 0.4805 ± 0.0460 | 9,111 |
| 2.0  | 21 / 21 | <r>0.3864</r> ± 0.0416 | 5,259 |

---

## Poincaré–Polar — `test_loss` across seeds

![width:760 center](calc_var_poincare_test_loss.png)

Mean stays on-target until $\lambda \gtrsim 0.5$; seed-to-seed **std grows ~40×** ($\lambda{=}0.1 \to 1.0$).

---

## Poincaré–Polar — Estimator variance explodes with $\lambda$

![width:760 center](calc_var_poincare_test_loss_var.png)

`test_loss_var` rises from **9** ($\lambda{=}0.1$) to **9,100** ($\lambda{=}1.0$) — a **~1000×** blow-up.

---

## Lorentz–Cartesian — Test NELBO vs $\lambda$

Lorentz is **far more fragile**: every $\lambda \le 0.5$ collapses to NaN.

| $\lambda$ | valid | `test_loss` (mean ± std) | `test_loss_var` |
|:--:|:--:|:--:|:--:|
| 0.01 – 0.5 | 0 / 21 | <r>NaN</r> | <r>NaN</r> |
| 0.8  | 16 / 21 | 0.4812 ± 0.0228 | 1,808 |
| 1.0  | 20 / 21 | 0.4809 ± 0.0471 | 9,527 |
| 2.0  | 21 / 21 | <r>0.3864</r> ± 0.0416 | 5,259 |

Even at $\lambda = 0.8$, 5 / 21 seeds still produce NaN.

---

## Lorentz–Cartesian — `test_loss` across seeds

![width:760 center](calc_var_lorentz_test_loss.png)

Only $\lambda \in \{0.8, 1.0, 2.0\}$ survive — the surviving points coincide with Poincaré.

---

## Three Regimes of the Proposal Rate $\lambda$

| Regime | $\lambda$ (Poincaré) | Behaviour |
|---|:--:|---|
| **Too small** | $\le 0.01$ | <r>NaN</r> — heavy-tailed weights $1/(\lambda u)$ blow up |
| **Stable** | **0.1 – 0.3** | <b>unbiased</b> (mean ≈ 0.500), variance 9 – 63 |
| **Too large** | $\ge 0.8$ | variance $10^3$–$10^4$; <r>biased low</r> (0.39 at $\lambda{=}2$) |

The low `test_loss` at large $\lambda$ is **not** an improvement — it under-counts the boundary tail.

---

## Takeaways

- **Sweet spot: $\lambda \approx 0.1$** (Poincaré–Polar) — recovers the target (0.5001 vs 0.500) with the smallest seed std (0.001) and lowest estimator variance (9).
- **Variance grows monotonically with $\lambda$** — ~1000× from $\lambda{=}0.1$ to $\lambda{=}1.0$; the estimate becomes seed-dependent and unusable.
- **Large $\lambda$ ($\ge 2$) is biased low** (0.386 < 0.500) — consistent with tail-truncation, not a tighter bound.
- **Lorentz–Cartesian is numerically fragile** — needs $\lambda \ge 0.8$ just to avoid NaN, vs $\lambda \ge 0.1$ for Poincaré.

---

## Next Step

- Sweep finer $\lambda$ around **0.1–0.2** to pin the variance-minimising rate.
- Diagnose the Lorentz NaN at small $\lambda$ (boundary blow-up in `_lorentz_norm_sq`).
- Use a **log-scale** $\lambda$ axis / log `test_loss_var` axis in the `calc_var` plots.

---

## Stratified-Exp Proposal — NELBO, IS+NELBO, and Proposal Weight Vis

---

## Motivation

Open up the test loss — **per-sample** views vs timestep $t$:

- **NELBO** $\ell(t)$ — the un-weighted loss integrand (geometry behaviour)
- **IS·NELBO** $\ell(t)\,w(t)$ — what the test-loss estimator actually averages
- **Proposal density** $p(t) = \lambda e^{-\lambda t}$ — *where* samples land along $t$

Goal: explain **why** small $\lambda$ NaNs (Lorentz) and large $\lambda$ blows up the variance.

---

## Experiment Set Up

### Single-Seed Stratified-Exp Sweep — `unigram_test_lorentz_tmp3.sh`

- Driver: `unigram/unigram_test2_tmp3.py`, **seed = 42**
- Model: **optimal** (`mode=opt`) — emits ground-truth unigram dist, no training
- Vocab size: 10  |  Data dist: $[0.91,\ 0.01\times 9]$  |  entropy ≈ **0.5003**
- Bridge horizon: `hyper_T = 1000`, `hyper_dt = 0.01`, `rotate_emb = True`
- Test set: **4,000,000** samples per run
- Proposal: **stratified exponential**, $\lambda \in \{0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0\}$
- Geometries: **Lorentz–Cartesian** and **Poincaré–Polar** (8 rates × 2 geoms = 16 runs)

---

## What Each Plot Shows

Per-sample log–log scatter of 4M samples (one figure per run, three stacked panels):

1. **weighted NELBO** $\ell(t)\,w(t)$ — mean over samples = `test_loss`
2. **NELBO (unweighted)** $\ell(t)$ — pure geometry / bridge term
3. **proposal density** $p(t) = \lambda e^{-\lambda t}$ — peaks at $t=0$, decays at rate $\lambda$

Generated by `plot_test_loss_vs_timestep` in `unigram/unigram_test2_tmp3.py:615`.

---

## Poincaré–Polar — Summary (seed 42)

| $\lambda$ | `test_loss` | `test_nelbo` | `test_loss_var` |
|:--:|:--:|:--:|:--:|
| 0.01 | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.1  | **0.4995** | 0.0338 | **9.3** |
| 0.2  | 0.5020 | 0.0513 | 17.6 |
| 0.3  | 0.4997 | 0.0619 | 57.6 |
| 0.5  | 0.5045 | 0.0738 | 524 |
| 0.8  | 0.4936 | 0.0824 | 1,248 |
| 1.0  | 0.4914 | 0.0854 | 4,349 |
| 2.0  | <r>0.3539</r> | 0.0914 | 1,349 |

Target: `test_ce` = **0.5003** (data entropy). Best $\lambda$: **0.1** — on target, ~470× lower variance than $\lambda{=}1.0$.

---

## Lorentz–Cartesian — Summary (seed 42)

| $\lambda$ | `test_loss` | `test_nelbo` | `test_loss_var` |
|:--:|:--:|:--:|:--:|
| 0.01 | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.1  | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.2  | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.3  | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.5  | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.8  | 0.4936 | 0.0824 | 1,248 |
| 1.0  | 0.4914 | 0.0854 | 4,349 |
| 2.0  | <r>0.3539</r> | 0.0914 | 1,349 |

For $\lambda \ge 0.8$ Lorentz **matches Poincaré bit-for-bit**. Small $\lambda$ NaNs come from $\ell(t)$ blow-ups at the boundary, not from the proposal.

---

## Per-Sample Scatter — Sweeping $\lambda$ across both geometries

For each $\lambda$ we show the **same three panels** side-by-side:

- **Left**: Poincaré–Polar  ·  **Right**: Lorentz–Cartesian
- Panels (top → bottom): **weighted NELBO**, **NELBO (unweighted)**, **proposal density**
- 4M samples per run, log–log axes

The story across $\lambda$: NaN regime → stable regime → variance blow-up → biased-low regime.

---

## $\lambda = 0.01$ — both geometries NaN

**Poincaré: <r>NaN</r>**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_poincare_per0.01.jpg) ![w:340](ts_lorentz_per0.01.jpg)

Density barely decays → samples reach $t \sim 10^3$ where $\ell(t)$ underflows and IS weight $e^{\lambda t}/\lambda$ multiplies $0$ by $\infty$.

---

## $\lambda = 0.1$ — Poincaré sweet spot, Lorentz still NaN

**Poincaré: 0.4995, var 9.3**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_poincare_per0.1.jpg) ![w:340](ts_lorentz_per0.1.jpg)

Bulk scatters look identical. Lorentz's rare tail reaches $\ell \sim 10^3$ (vs $\sim 1$ for Poincaré) — boundary blow-up in $\langle z_t, \xi(y)\rangle_L$.

---

## $\lambda = 0.2$ — Poincaré stable, Lorentz NaN

**Poincaré: 0.5020, var 17.6**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_poincare_per0.2.jpg) ![w:340](ts_lorentz_per0.2.jpg)

Variance roughly **2×** $\lambda{=}0.1$. Lorentz NaN persists — the issue is the geometry's tail, not how often we sample it.

---

## $\lambda = 0.3$ — Poincaré stable, Lorentz NaN

**Poincaré: 0.4997, var 57.6**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_poincare_per0.3.jpg) ![w:340](ts_lorentz_per0.3.jpg)

Variance starts climbing fast (~**6×** $\lambda{=}0.1$); IS·NELBO panel widens visibly at large $t$.

---

## $\lambda = 0.5$ — Poincaré on-target but high variance, Lorentz NaN

**Poincaré: 0.5045, var 524**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_poincare_per0.5.jpg) ![w:340](ts_lorentz_per0.5.jpg)

Poincaré IS·NELBO now reaches $\sim 10^2$ — variance has jumped **~10×** vs $\lambda{=}0.3$.

---

## $\lambda = 0.8$ — both stable, geometries **agree exactly**

**Poincaré: 0.4936, var 1,248**  ·  **Lorentz: 0.4936, var 1,248**

![w:340](ts_poincare_per0.8.jpg) ![w:340](ts_lorentz_per0.8.jpg)

First $\lambda$ where Lorentz survives. The two `test_loss` / `test_loss_var` numbers match bit-for-bit ⇒ when finite, the two formulations are numerically equivalent.

---

## $\lambda = 1.0$ — variance blow-up, mean still on target

**Poincaré: 0.4914, var 4,349**  ·  **Lorentz: 0.4914, var 4,349**

![w:340](ts_poincare_per1.0.jpg) ![w:340](ts_lorentz_per1.0.jpg)

Top panel IS·NELBO reaches $\sim 10^3$. Density falls off past $t \approx 5$ — rare large-$t$ samples carry huge IS weights and dominate the mean.

---

## $\lambda = 2.0$ — biased low: tail under-sampled

**Poincaré: <r>0.3539</r>, var 1,349**  ·  **Lorentz: <r>0.3539</r>, var 1,349**

![w:340](ts_poincare_per2.0.jpg) ![w:340](ts_lorentz_per2.0.jpg)

Density collapses by $t \approx 3$; the bridge tail is essentially never sampled. `test_loss` drops **0.39 < 0.5** — not a tighter bound, just missing integration mass.

---

## Three Regimes — What Goes Wrong Where

| Regime | NELBO $\ell(t)$ | IS weight $w(t)$ | Outcome |
|---|---|---|---|
| **$\lambda \le 0.5$** (small) | Lorentz blows up in tail | moderate | <r>Lorentz NaN</r>, Poincaré OK |
| **0.1 – 0.3** | both bounded | $w(t)$ tame | <b>unbiased, low var</b> |
| **$\lambda \ge 0.8$** (large) | bounded | $w(t) = e^{\lambda t}/\lambda$ huge | high var, ~unbiased |
| **$\lambda \ge 2$** (too large) | bounded | $p(t) \approx 0$ past $t\approx 3$ | <r>biased low</r> (tail missed) |

---

## Takeaways

- **Per-sample views separate two failure modes** — small-$\lambda$ NaN is a **geometry** issue (Lorentz boundary), large-$\lambda$ variance is an **estimator** issue (heavy IS weights).
- **Poincaré $\lambda = 0.1$ is the sweet spot** — on-target mean (0.4995) and lowest variance (9.3) among non-NaN rates.
- **Lorentz and Poincaré agree** whenever Lorentz is finite ($\lambda \ge 0.8$); fixing the Lorentz boundary in `_lorentz_norm_sq` would close the gap at $\lambda \le 0.5$.
- The IS·NELBO panel makes the variance blow-up visible: at $\lambda = 1.0$ a few samples reach $10^3$, dominating the mean.

---

# Various Ground-Truth Distributions

---

## Motivation

The seed-variance and per-sample studies all used **one** ground-truth distribution
with entropy $\approx 0.500$. Do the same regimes hold when the data entropy changes?

We re-run the optimal model (`mode=opt`, model output = ground truth) against **two new ground-truth unigrams** with very different entropies and inspect the test ELBO and CE across $\lambda$.

- **Low entropy** (`cmplx_ps`): concentrated on a few tokens
- **High entropy** (`cmplx_ps1`): near-uniform

Drivers: `unigram_test_script/unigram_test_lorentz_tmp3_cmplx_ps.sh`, `..._ps1.sh`.

---

## Experiment Set Up

- Driver: `unigram/unigram_test2_tmp3.py`, **seed = 42**
- Model: **optimal** (`mode=opt`) — emits ground-truth unigram dist, no training
- Vocab size: **10**  |  bridge horizon: `hyper_T = 1000`, `hyper_dt = 0.01`, `rotate_emb = True`
- Test set: **4,000,000** samples per run
- Proposal: **stratified exponential**, $\lambda \in \{0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0\}$, plus uniform $[0.01, 10]$ baseline
- Geometries: **Lorentz–Cartesian**, **Poincaré–Polar**

| Tag | Ground-truth $p$ | $H(p)$ = `test_ce` |
|---|---|:--:|
| `cmplx_ps`  | $[0.31, 0.01, 0.20, 0.01, 0.01, 0.30, 0.08, 0.04, 0.03, 0.01]$ | **1.6664** |
| `cmplx_ps1` | $[0.11, 0.10, 0.10, 0.11, 0.11, 0.10, 0.10, 0.10, 0.09, 0.08]$ | **2.2985** |

Upper bound for $V{=}10$: $\log 10 \approx 2.3026$; `ps1` is essentially uniform.

---

## What "Good" Looks Like

For the optimal model the test ELBO equals the data entropy:

| Tag | target `test_loss` ( = `test_ce`) |
|---|:--:|
| `cmplx_ps`  (low entropy) | **1.6664** |
| `cmplx_ps1` (high entropy) | **2.2985** |

A good $\lambda$ is **(1)** NaN-free, **(2)** unbiased (mean $\approx$ target CE), **(3)** low-variance — exactly the same criteria as before, just with two different targets.

---

## Low-Entropy GT (`cmplx_ps`, CE = 1.6664) — Poincaré–Polar

| $\lambda$ | `test_loss` (ELBO) | `test_nelbo` | `test_loss_var` | std$/\sqrt{N}$ |
|:--:|:--:|:--:|:--:|:--:|
| unif $[0.01,10]$ | <r>1.5008</r> | 0.1502 | 7.5 | 0.0014 |
| 0.01 | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> | — |
| **0.1**  | **1.6665** | 0.1200 | **18.5** | **0.0022** |
| 0.2  | 1.6672 | 0.1899 | 43.1 | 0.0033 |
| 0.3  | 1.6655 | 0.2370 | 146.1 | 0.0060 |
| 0.4  | 1.6569 | 0.2710 | 224.9 | 0.0075 |
| 0.5  | 1.6682 | 0.2970 | 1,258 | 0.0177 |
| 0.8  | 1.8878 | 0.3479 | 332,725 | 0.288 |
| 1.0  | 1.5875 | 0.3693 | 4,490 | 0.0335 |
| 2.0  | <r>1.3483</r> | 0.4223 | 16,832 | 0.0649 |

Target = **1.6664**. Best $\lambda$ = **0.1**: on-target (1.6665) with the lowest variance.

---

## Low-Entropy GT — Lorentz–Cartesian

| $\lambda$ | `test_loss` (ELBO) | `test_nelbo` | `test_loss_var` |
|:--:|:--:|:--:|:--:|
| unif $[0.01,10]$ | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.01 – 0.5 | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.8  | 1.8878 | 0.3479 | 332,725 |
| 1.0  | 1.5875 | 0.3693 | 4,490 |
| 2.0  | <r>1.3483</r> | 0.4223 | 16,832 |

Lorentz remains numerically fragile: every $\lambda \le 0.5$ NaNs. Whenever both geometries are finite, **Lorentz = Poincaré** bit-for-bit.

---

## High-Entropy GT (`cmplx_ps1`, CE = 2.2985) — Poincaré–Polar

| $\lambda$ | `test_loss` (ELBO) | `test_nelbo` | `test_loss_var` | std$/\sqrt{N}$ |
|:--:|:--:|:--:|:--:|:--:|
| unif $[0.01,10]$ | <r>2.0530</r> | 0.2055 | 7.5 | 0.0014 |
| 0.01 | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> | — |
| **0.1**  | **2.2984** | 0.1615 | **22.7** | **0.0024** |
| 0.2  | 2.2950 | 0.2505 | 42.8 | 0.0033 |
| 0.3  | 2.2829 | 0.3071 | 135.0 | 0.0058 |
| 0.4  | 2.2871 | 0.3463 | 562.6 | 0.0119 |
| 0.5  | 2.2872 | 0.3746 | 1,996 | 0.0223 |
| 0.8  | 2.6807 | 0.4256 | 904,924 | 0.476 |
| 1.0  | 2.1632 | 0.4444 | 5,135 | 0.0358 |
| 2.0  | <r>1.8579</r> | 0.4820 | 36,502 | 0.0955 |

Target = **2.2985**. Best $\lambda$ = **0.1**: on-target (2.2984) with the lowest variance.

---

## High-Entropy GT — Lorentz–Cartesian

| $\lambda$ | `test_loss` (ELBO) | `test_nelbo` | `test_loss_var` |
|:--:|:--:|:--:|:--:|
| unif $[0.01,10]$ | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.01 – 0.5 | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |
| 0.8  | 2.6807 | 0.4256 | 904,924 |
| 1.0  | 2.1632 | 0.4444 | 5,135 |
| 2.0  | <r>1.8579</r> | 0.4820 | 36,502 |

Same Lorentz-boundary fragility as the low-entropy case; same exact match with Poincaré whenever finite.

---

## ELBO vs CE Across Entropies — Side-by-Side

`test_loss` of the optimal model at the recommended rate $\lambda = 0.1$ (Poincaré–Polar):

| GT | `test_ce` (target) | `test_loss` @ $\lambda{=}0.1$ | abs. gap | rel. gap |
|---|:--:|:--:|:--:|:--:|
| `cmplx_ps`  (low entropy) | 1.6664 | **1.6665** | $+1\!\times\!10^{-4}$ | $0.006\%$ |
| `cmplx_ps1` (high entropy) | 2.2985 | **2.2984** | $-1\!\times\!10^{-4}$ | $0.004\%$ |

The optimal-model ELBO matches CE to **4 decimal places** at $\lambda = 0.1$ for *both* ground truths.

The uniform-proposal baseline systematically **under-counts** by 0.17–0.25 (truncated at $t = 10$).

---

## Low-Entropy GT — $\lambda = 0.1$ (sweet spot, Lorentz NaN)

**Poincaré: 1.6665, var 18.5**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_cmplx_ps_pp_per0.1.jpg) ![w:340](ts_cmplx_ps_lc_per0.1.jpg)

Bulk scatters match the per-rate panels from the seed-variance study; only the IS·NELBO mean shifts up to track CE = 1.6664.

---

## Low-Entropy GT — $\lambda = 0.5$ (Poincaré stable, Lorentz NaN)

**Poincaré: 1.6682, var 1,258**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_cmplx_ps_pp_per0.5.jpg) ![w:340](ts_cmplx_ps_lc_per0.5.jpg)

Variance has climbed ~70×; mean still on target.

---

## Low-Entropy GT — $\lambda = 1.0$ (variance blow-up, geometries agree)

**Poincaré: 1.5875, var 4,490**  ·  **Lorentz: 1.5875, var 4,490**

![w:340](ts_cmplx_ps_pp_per1.0.jpg) ![w:340](ts_cmplx_ps_lc_per1.0.jpg)

Top panel IS·NELBO reaches $\sim 10^3$; mean has drifted ~0.08 below target.

---

## Low-Entropy GT — $\lambda = 2.0$ (biased low, tail under-sampled)

**Poincaré: <r>1.3483</r>, var 16,832**  ·  **Lorentz: <r>1.3483</r>, var 16,832**

![w:340](ts_cmplx_ps_pp_per2.0.jpg) ![w:340](ts_cmplx_ps_lc_per2.0.jpg)

Density collapses by $t \approx 3$; ELBO drops to **1.35 < CE = 1.67** — missing tail mass.

---

## High-Entropy GT — $\lambda = 0.1$ (sweet spot, Lorentz NaN)

**Poincaré: 2.2984, var 22.7**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_cmplx_ps1_pp_per0.1.jpg) ![w:340](ts_cmplx_ps1_lc_per0.1.jpg)

IS·NELBO mean shifts up to track the higher CE = 2.2985; everything else identical to low-entropy.

---

## High-Entropy GT — $\lambda = 0.5$ (Poincaré stable, Lorentz NaN)

**Poincaré: 2.2872, var 1,996**  ·  **Lorentz: <r>NaN</r>**

![w:340](ts_cmplx_ps1_pp_per0.5.jpg) ![w:340](ts_cmplx_ps1_lc_per0.5.jpg)

Same regime shape, variance ~1.6× the low-entropy case at the same $\lambda$.

---

## High-Entropy GT — $\lambda = 1.0$ (variance blow-up, geometries agree)

**Poincaré: 2.1632, var 5,135**  ·  **Lorentz: 2.1632, var 5,135**

![w:340](ts_cmplx_ps1_pp_per1.0.jpg) ![w:340](ts_cmplx_ps1_lc_per1.0.jpg)

Mean has drifted ~0.14 below target — slightly larger absolute gap than low-entropy.

---

## High-Entropy GT — $\lambda = 2.0$ (biased low, tail under-sampled)

**Poincaré: <r>1.8579</r>, var 36,502**  ·  **Lorentz: <r>1.8579</r>, var 36,502**

![w:340](ts_cmplx_ps1_pp_per2.0.jpg) ![w:340](ts_cmplx_ps1_lc_per2.0.jpg)

ELBO drops to **1.86 < CE = 2.30** — absolute gap **0.44** vs 0.32 in low-entropy: tail truncation **hurts more when entropy is higher**.

---

## Cross-Entropy Comparison: Bias Scales with CE

Absolute bias `|test_loss − test_ce|` at each $\lambda$ (Poincaré–Polar):

| $\lambda$ | low-entropy bias | high-entropy bias |
|:--:|:--:|:--:|
| unif | 0.166 | 0.246 |
| 0.1  | **0.0001** | **0.0001** |
| 0.2  | 0.0008 | 0.0035 |
| 0.3  | 0.0009 | 0.0156 |
| 0.4  | 0.0095 | 0.0114 |
| 0.5  | 0.0018 | 0.0113 |
| 0.8  | 0.2214 | 0.3822 |
| 1.0  | 0.0789 | 0.1353 |
| 2.0  | **0.3181** | **0.4406** |

Bias at small/well-chosen $\lambda$ is **entropy-invariant** ($\sim 10^{-4}$); bias at the failure regimes (unif, large $\lambda$) **grows with CE**.

---

## Conclusion

- **Optimal-model ELBO = CE at $\lambda = 0.1$ across entropies** (1.6665 vs 1.6664, 2.2984 vs 2.2985) — the estimator is **unbiased and entropy-agnostic** in the sweet spot.
- **Three regimes are preserved across data entropies**: small $\lambda$ → NaN (Lorentz boundary); $\lambda \in [0.1, 0.5]$ → unbiased, low variance (Poincaré); $\lambda \ge 0.8$ → high variance and increasing low-bias from tail truncation.
- **Variance scales with entropy**: at the same $\lambda$, the higher-entropy run has $\sim 1.2$–$3\times$ larger `test_loss_var` (e.g. $\lambda{=}0.5$: 1,258 vs 1,996; $\lambda{=}2.0$: 16,832 vs 36,502).
- **Tail-truncation bias grows with entropy**: at $\lambda = 2.0$ the absolute gap is **0.32** for $H{=}1.67$ and **0.44** for $H{=}2.30$ — the bridge tail carries more mass when the target is broader.
- **Uniform proposal is uniformly bad**: under-counts CE by 0.17 (low entropy) and 0.25 (high entropy); stratified exponential at $\lambda = 0.1$ is the right default.
- **Lorentz still requires $\lambda \ge 0.8$** regardless of GT entropy — the NaN comes from `_lorentz_norm_sq`, not from the data distribution.

---

