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

### May 12, 2026

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

## Experiment Set Up

- Vocab Size: 10
- Training Steps: 20000
- LR: 1e-5
- Training Dataset Size: 20000
- Valid / Test Dataset Size: 4000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model is trainable

---

# Lorentz - Cartesian

---

## Lorentz - Cartesian Overview

- Unif[0.01, 10]: 0.4057
- Exp(0.01): NaN
- Exp(0.1): NaN
- Exp(0.2): 0.5059
- Exp(0.3): 0.4233
- Exp(0.5): 0.3499
- Exp(1.0): 0.3707
- Exp(2.0): 0.2562
- Exp(3.0):

---

### Lorentz, Cartesian, Important Sampling: Unif[0.01, 10]

![width:700](image-25.png)

- Test ELBO: 0.4057

---

### Lorentz, Cartesian, Important Sampling: Exp(0.01)

![width:700](image-26.png)

- Test ELBO: NaN

---

### Lorentz, Cartesian, Important Sampling: Exp(0.1)

![width:700](image-27.png)

- Test ELBO: NaN

---

### Lorentz, Cartesian, Important Sampling: Exp(0.2)

![width:700](image-31.png)

- Test ELBO: 0.5059

---

### Lorentz, Cartesian, Important Sampling: Exp(0.3)

![width:700](image-32.png)

- Test ELBO: 0.4233

---

### Lorentz, Cartesian, Important Sampling: Exp(0.5)

![width:700](image-28.png)

- Test ELBO: 0.3499

---

### Lorentz, Cartesian, Important Sampling: Exp(1.0)

![width:700](image-29.png)

- Test ELBO: 0.3707

---

### Lorentz, Cartesian, Important Sampling: Exp(2.0)

![width:700](image-30.png)

- Test ELBO: 0.2562

---

# Poincare Disk - Polar

---

#  Poincare Disk - Polar Overview

- Unif[0.01, 10]: 0.4595
- Exp(0.01): NaN
- Exp(0.05): 0.7154
- Exp(0.1): 0.6749
- Exp(0.2): 0.8076
- Exp(0.3): 0.9689
- Exp(0.5): 1.1802
- Exp(1.0): 1.0641
- Exp(2.0): 1.2368
- Exp(3.0): 2.7444

---

### Poincare Disk, Polar, Important Sampling: Unif[0.01, 10]

![width:700](image-6.png)

- Test ELBO: 0.4595

---

### Poincare Disk, Polar, Important Sampling: Exp(0.01)

![width:700](image-1.png)

- Test ELBO: NaN

---

### Poincare Disk, Polar, Important Sampling: Exp(0.05)

![width:700](image-16.png)

- Test ELBO: 0.7154

---

<!-- ### Poincare Disk, Polar, Important Sampling: Exp(0.075)

![width:700](image-24.png)

- Test ELBO: 0.6913

---

### Poincare Disk, Polar, Important Sampling: Exp(0.078)

![width:700](image-23.png)

- Test ELBO: 0.5996

---

### Poincare Disk, Polar, Important Sampling: Exp(0.079)

![width:700](image-22.png)

- Test ELBO: 0.6605

---

### Poincare Disk, Polar, Important Sampling: Exp(0.08)

![width:700](image-15.png)

- Test ELBO: 0.5872

---

### Poincare Disk, Polar, Important Sampling: Exp(0.081)

![width:700](image-21.png)

- Test ELBO: 0.6138

---

### Poincare Disk, Polar, Important Sampling: Exp(0.082)

![width:700](image-20.png)

- Test ELBO: 0.6482

---

### Poincare Disk, Polar, Important Sampling: Exp(0.085)

![width:700](image-19.png)

- Test ELBO: 0.7632

---

### Poincare Disk, Polar, Important Sampling: Exp(0.09)

![width:700](image-14.png)

- Test ELBO: 0.6750

--- -->

### Poincare Disk, Polar, Important Sampling: Exp(0.1)

![width:700](image.png)

- Test ELBO: 0.6749

---

### Poincare Disk, Polar, Important Sampling: Exp(0.2)

![width:700](image-17.png)

- Test ELBO: 0.8076

---

### Poincare Disk, Polar, Important Sampling: Exp(0.3)

![width:700](image-18.png)

- Test ELBO: 0.9689

---

### Poincare Disk, Polar, Important Sampling: Exp(0.5)

![width:700](image-2.png)

- Test ELBO: 1.1802

---

### Poincare Disk, Polar, Important Sampling: Exp(1.0)

![width:700](image-3.png)

- Test ELBO: 1.0641

---

### Poincare Disk, Polar, Important Sampling: Exp(2.0)

![width:700](image-4.png)

- Test ELBO: 1.2368

---

### Poincare Disk, Polar, Important Sampling: Exp(3.0)

![width:700](image-5.png)

- Test ELBO: 2.7444

---

## Poincare Disk - Cartesian Coodinate

---

### Poincare Disk, Cart, Important Sampling: Unif[0.01, 10]

![width:700](image-9.png)

- Test ELBO: 0.4595

---

### Poincare Disk, Cart, Important Sampling: Exp(0.01)

![width:700](image-8.png)

- Test ELBO: NaN

---

### Poincare Disk, Cart, Important Sampling: Exp(0.1)

![width:700](image-7.png)


- Test ELBO: NaN

---

### Poincare Disk, Cart, Important Sampling: Exp(0.5)

![width:700](image-10.png)

- Test ELBO: 1.1802

---

### Poincare Disk, Cart, Important Sampling: Exp(1.0)

![width:700](image-11.png)

- Test ELBO: 1.0641

---

### Poincare Disk, Cart, Important Sampling: Exp(2.0)

![width:700](image-12.png)

- Test ELBO: 1.2368

---

### Poincare Disk, Cart, Important Sampling: Exp(3.0)

![width:700](image-13.png)

- Test ELBO: 2.7444

---

## Trucation at Large $\lambda$

The truncation cause the low test_loss as $\lambda$ is large, because the loss ignores the tail (large $t$ close to boundary)

Small $t$ has larger loss variance because it is far away from the boundary.

---

## Next Step

- The implmentation of loss should be correct. 
- Require tuning for $\lambda$


---

## Experiment Set Up

- Vocab Size: 10
- Training Steps: 20000
- LR: 1e-5
- Training Dataset Size: 20000
- Valid / Test Dataset Size: 4000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5

---

# Lorentz - Cartesian Overview

- Unif[0.01, 10]: 0.4057
- Exp(0.01): NaN
- Exp(0.1): NaN
- Exp(0.2): 0.5059
- Exp(0.3): 0.4233
- Exp(0.5): 0.3499
- Exp(1.0): 0.3707
- Exp(2.0): 0.2562
- Exp(3.0):

---

#  Poincare Disk - Polar Overview

- Unif[0.01, 10]: 0.4595
- Exp(0.01): NaN
- Exp(0.05): 0.7154
- Exp(0.1): 0.6749
- Exp(0.2): 0.8076
- Exp(0.3): 0.9689
- Exp(0.5): 1.1802
- Exp(1.0): 1.0641
- Exp(2.0): 1.2368
- Exp(3.0): 2.7444

---

## End

---

## Experiment Set Up

### Fixed Stratified Exp + log

- Vocab Size: 10
- LR: 1e-5
- Test Dataset Size: 400000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model outputs ground-truth training data distribution
```python
u = u.clamp(min=1e-12, max=1 - 1e-12)
ts = - torch.log(u) / exp_rate
```
---

## Lorentz - Cartesian Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.1): 0.5477, Var: 11.0593, CE: 0.5002
- Stratified Exp(0.2): 0.5470, Var: 15.5015, CE: 0.5002
- Stratified Exp(0.3): 0.5516, Var: 46.3914, CE: 0.5002
- Stratified Exp(0.5): 0.5110, Var: 39.1890, CE: 0.5002
- Stratified Exp(0.8): 0.4709, Var: 81.2501, CE: 0.5002
- Stratified Exp(1.0): 0.5473, Var: 4194.7417, CE: 0.5002
- Stratified Exp(2.0): 0.5553, Var: 19920.7353, CE: 0.5002

---

## Poincare - Polar Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5304, Var: 60.0901, CE: 0.5002
- Stratified Exp(0.1): 0.5489, Var: 11.5475, CE: 0.5002
- Stratified Exp(0.2): 0.5469, Var: 22.2538, CE: 0.5002
- Stratified Exp(0.3): 0.5408, Var: 30.2793, CE: 0.5002
- Stratified Exp(0.5): 0.5732, Var: 588.0352, CE: 0.5002
- Stratified Exp(0.8): 1.0936, Var: 120364.8515, CE: 0.5002
- Stratified Exp(1.0): 0.8336, Var: 46185.4688, CE: 0.5002
- Stratified Exp(2.0): 0.4047, Var: 766.7626, CE: 0.5002

---

## Experiment Set Up

### Shuffled Stratified Exp + log

- Vocab Size: 10
- LR: 1e-5
- Test Dataset Size: 400000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model outputs ground-truth training data distribution
```python
u = u.view(-1)[torch.randperm(u.numel())].view(u.shape)
u = u.clamp(min=1e-12, max=1 - 1e-12)
ts = - torch.log(u) / exp_rate
```

---

## Lorentz - Cartesian Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5690, Var: 64.9428, CE: 0.5002
- Stratified Exp(0.1): 0.5558, Var: 11.0196, CE: 0.5002
- Stratified Exp(0.2): 0.5521, Var: 17.7441, CE: 0.5002
- Stratified Exp(0.3): 0.5396, Var: 21.9774, CE: 0.5002
- Stratified Exp(0.5): 0.5165, Var: 58.5893, CE: 0.5002
- Stratified Exp(0.8): 0.4921, Var: 149.1625, CE: 0.5002
- Stratified Exp(1.0): 0.4447, Var: 56.4453, CE: 0.5002
- Stratified Exp(2.0): 0.3584, Var: 206.8494, CE: 0.5002

---

## Poincare - Polar Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5690, Var: 64.9428, CE: 0.5002
- Stratified Exp(0.1): 0.5558, Var: 11.0196, CE: 0.5002
- Stratified Exp(0.2): 0.5521, Var: 17.7441, CE: 0.5002
- Stratified Exp(0.3): 0.5396, Var: 21.9774, CE: 0.5002
- Stratified Exp(0.5): 0.5165, Var: 58.5893, CE: 0.5002
- Stratified Exp(0.8): 0.4921, Var: 149.1625, CE: 0.5002
- Stratified Exp(1.0): 0.4447, Var: 56.4453, CE: 0.5002
- Stratified Exp(2.0): 0.3584, Var: 206.8494, CE: 0.5002

---

## Experiment Set Up

### Shuffled Stratified Exp + log + Shorten Compute

- Vocab Size: 10
- LR: 1e-5
- Test Dataset Size: 400000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model outputs ground-truth training data distribution
```python
u = u.view(-1)[torch.randperm(u.numel(), device=device)].view(u.shape)
u = u.clamp(min=1e-12, max=1 - 1e-12).reshape(shape)
ts = - torch.log(u) / exp_rate
density = exp_rate * u
```

---

## Lorentz - Cartesian Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5690, Var: 64.9428, CE: 0.5002
- Stratified Exp(0.1): 0.5558, Var: 11.0196, CE: 0.5002
- Stratified Exp(0.2): 0.5521, Var: 17.7441, CE: 0.5002
- Stratified Exp(0.3): 0.5396, Var: 21.9774, CE: 0.5002
- Stratified Exp(0.5): 0.5165, Var: 58.5893, CE: 0.5002
- Stratified Exp(0.8): 0.4921, Var: 149.1625, CE: 0.5002
- Stratified Exp(1.0): 0.4447, Var: 56.4453, CE: 0.5002
- Stratified Exp(2.0): 0.3584, Var: 206.8494, CE: 0.5002

---

## Poincare - Polar Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5690, Var: 64.9428, CE: 0.5002
- Stratified Exp(0.1): 0.5558, Var: 11.0196, CE: 0.5002
- Stratified Exp(0.2): 0.5521, Var: 17.7441, CE: 0.5002
- Stratified Exp(0.3): 0.5396, Var: 21.9774, CE: 0.5002
- Stratified Exp(0.5): 0.5165, Var: 58.5893, CE: 0.5002
- Stratified Exp(0.8): 0.4921, Var: 149.1625, CE: 0.5002
- Stratified Exp(1.0): 0.4447, Var: 56.4453, CE: 0.5002
- Stratified Exp(2.0): 0.3584, Var: 206.8494, CE: 0.5002

---

## Experiment Set Up

### Shuffled Stratified Exp + log + Shorten Compute + x10 + Rotate

- Vocab Size: 10
- LR: 1e-5
- Test Dataset Size: 4000000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model outputs ground-truth training data distribution


---

## Experiment Set Up

### Shuffled Stratified Exp + log + Shorten Compute + x10 + Rotate

Add rotation

```python
rhos, thetas = self.bridge.binary_bridge(ts=ts)
thetas = thetas + (
    targets.to(dtype=torch.float64) + 0.5
) * (2 * torch.pi / int(self.config.vocab_size))
```

Stratified Exp Importance Sampling

```python
u = u.view(-1)[torch.randperm(u.numel(), device=device)].view(u.shape)
u = u.clamp(min=1e-12, max=1 - 1e-12).reshape(shape)
ts = - torch.log(u) / exp_rate
weights = 1.0 / (exp_rate * u)
```

---

## Experiment Set Up

### Shuffled Stratified Exp + log + Shorten Compute + x10 + Rotate

Loss Function

```python
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
    thetas = thetas + (
        targets.to(dtype=torch.float64) + 0.5
    ) * (2 * torch.pi / int(self.config.vocab_size))
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
```

---

## Lorentz - Cartesian Overview

- Stratified Exp(0.01): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.1): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.2): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.3): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.5): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.8): 0.4935, Var: 1247.9748, CE: 0.5002
- Stratified Exp(1.0): 0.4914, Var: 4348.8323, CE: 0.5002
- Stratified Exp(2.0): 0.3539, Var: 1349.0525, CE: 0.5002

---

## Poincare - Polar Overview

- Stratified Exp(0.01): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.1): 0.4994, Var: 9.2611, CE: 0.5002
- Stratified Exp(0.2): 0.5020, Var: 17.6186, CE: 0.5002
- Stratified Exp(0.3): 0.4997, Var: 57.6125, CE: 0.5002
- Stratified Exp(0.5): 0.5044, Var: 523.9634, CE: 0.5002
- Stratified Exp(0.8): 0.4935, Var: 1247.9748, CE: 0.5002
- Stratified Exp(1.0): 0.4914, Var: 4348.8323, CE: 0.5002
- Stratified Exp(2.0): 0.3539, Var: 1349.0525, CE: 0.5002

---

## Poincare - Polar Overview

- Stratified Exp(2.0): 0.3539, Var: 1349.0525, CE: 0.5002 (seed 42)
- Stratified Exp(2.0): 0.4975, Var: 28598.2650, CE: 0.5002 (seed 43)

---

## Experiment Set Up

### Fixed Stratified Exp + log1p

- Vocab Size: 10
- LR: 1e-5
- Test Dataset Size: 400000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model outputs ground-truth training data distribution
```python
u = u.clamp(min=1e-12, max=1 - 1e-12)
ts = - torch.log1p(-u) / exp_rate
```

---

## Lorentz - Cartesian Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.1): 0.5477, Var: 11.0593, CE: 0.5002
- Stratified Exp(0.2): 0.5470, Var: 15.5015, CE: 0.5002
- Stratified Exp(0.3): 0.5516, Var: 46.3914, CE: 0.5002
- Stratified Exp(0.5): 0.5110, Var: 39.1890, CE: 0.5002
- Stratified Exp(0.8): 0.4709, Var: 81.2501, CE: 0.5002
- Stratified Exp(1.0): 0.5473, Var: 4194.7417, CE: 0.5002
- Stratified Exp(2.0): 0.5553, Var: 19920.7353, CE: 0.5002

---

## Poincare - Polar Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): NaN, Var: NaN, CE: 0.5002
- Stratified Exp(0.1): 0.5477, Var: 11.0593, CE: 0.5002
- Stratified Exp(0.2): 0.5470, Var: 15.5015, CE: 0.5002
- Stratified Exp(0.3): 0.5516, Var: 46.3914, CE: 0.5002
- Stratified Exp(0.5): 0.5110, Var: 39.1890, CE: 0.5002
- Stratified Exp(0.8): 0.4709, Var: 81.2501, CE: 0.5002
- Stratified Exp(1.0): 0.5473, Var: 4194.7417, CE: 0.5002
- Stratified Exp(2.0): 0.5553, Var: 19920.7353, CE: 0.5002

---

## Experiment Set Up

### Shuffled Stratified Exp + log1p

- Vocab Size: 10
- LR: 1e-5
- Test Dataset Size: 400000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Training Data Entropy: 0.5
- Model outputs ground-truth training data distribution
```python
u = u.view(-1)[torch.randperm(u.numel())].view(u.shape)
u = u.clamp(min=1e-12, max=1 - 1e-12)
ts = - torch.log1p(-u) / exp_rate
```

---

## Lorentz - Cartesian Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5573, Var: 63.1333, CE: 0.5002
- Stratified Exp(0.1): 0.5501, Var: 10.8407, CE: 0.5002
- Stratified Exp(0.2): 0.5496, Var: 14.6595, CE: 0.5002
- Stratified Exp(0.3): 0.5630, Var: 76.9901, CE: 0.5002
- Stratified Exp(0.5): 0.5463, Var: 96.2502, CE: 0.5002
- Stratified Exp(0.8): 0.4725, Var: 49.4189, CE: 0.5002
- Stratified Exp(1.0): 0.4713, Var: 127.0324, CE: 0.5002
- Stratified Exp(2.0): 0.3729, Var: 248.4214, CE: 0.5002

---

## Poincare - Polar Overview

- Unif[0.01, 10]: 
- Stratified Exp(0.01): 0.5573, Var: 63.1333, CE: 0.5002
- Stratified Exp(0.1): 0.5501, Var: 10.8407, CE: 0.5002
- Stratified Exp(0.2): 0.5496, Var: 14.6595, CE: 0.5002
- Stratified Exp(0.3): 0.5630, Var: 76.9901, CE: 0.5002
- Stratified Exp(0.5): 0.5463, Var: 96.2502, CE: 0.5002
- Stratified Exp(0.8): 0.4725, Var: 49.4189, CE: 0.5002
- Stratified Exp(1.0): 0.4713, Var: 127.0324, CE: 0.5002
- Stratified Exp(2.0): 0.3729, Var: 248.4214, CE: 0.5002


---

## Old Optimal Model - Experiment Set Up

- Vocab Size: 10
- Testing Dataset Size: 400000
- Prob Dist of Training Dataset: [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
- Testing Data Entropy: 0.5
- Model ouputs the likelihood of the dataset
- Removed Truncation of Exponential distribution proposal

---

### Lorentz - Cartesian Coordinate

- Stratified Exp(0.01): NaN, Var within Epoch: NaN
- Stratified Exp(0.1): 0.5477, Var within Epoch: 11.0593
- Stratified Exp(0.5): 0.5110, Var within Epoch: 39.1890
- Stratified Exp(0.8): 0.4709, Var within Epoch: 81.2501
- Stratified Exp(1.0): 0.5473, Var within Epoch: 4194.7417
- Stratified Exp(2.0): 0.5553, Var within Epoch: 19920.7353

---

### Poincare Disk - Polar Coordinate

- Stratified Exp(0.01): NaN, Var within Epoch: NaN
- Stratified Exp(0.1): 0.5477, Var within Epoch: 11.0593
- Stratified Exp(0.5): 0.5110, Var within Epoch: 39.1890
- Stratified Exp(0.8): 0.4709, Var within Epoch: 81.2501
- Stratified Exp(1.0): 0.5473, Var within Epoch: 4194.7417
- Stratified Exp(2.0): 0.5553, Var within Epoch: 19920.7353