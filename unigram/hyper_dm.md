
## Poincare Disk Model
Given a Poincare disk $\mathbb{D}^{d} = \{ z \in \mathbb{R}^{d}: || z || < 1 \}$. The hyperbolic distance $d_{\mathbb{H}}$ is defined as
$$
d_{\mathbb{H}} (z_1, z_2) = \operatorname{arcosh} \left( 1 + \frac{2 || z_1 - z_2 ||^2}{(1 - || z_1 ||^2) (1 - || z_2 ||^2)} \right)
$$
while the distance from the origin $O$ is defined as
$$
d_{\mathbb{H}} (z, 0) = \operatorname{arcosh} \left( \frac{1 + || z ||^2}{1 - || z ||^2} \right) = 2 \operatorname{artanh} \left( || z || \right)
$$
and the Busemann inner product $\langle \cdot, \cdot \rangle_{\mathbb{H}}$ between boundary $x$ and disk $z$
$$
\langle x, z \rangle_{\mathbb{H}} = \log \frac{1 - || z ||^2 }{|| x - z ||^2}
$$
, hence, the Poison kernel is
$$
q_{\infty | t}(x | z) = \exp((d - 1) \langle x, z \rangle_{\mathbb{H}}) = \left( \frac{1 - || z ||^2 }{|| x - z ||^2} \right)^{d-1}
$$
The $d$-dimensional hyperbolic heat kernel $P_{\mathbb{H}^{d}}$  with $d = 2$:
$$
P_{\mathbb{H}^2}(r;t)
=
\frac{\sqrt{2}\, e^{-t/4}}{(4\pi t)^{3/2}}
\int_r^\infty
\frac{s e^{-s^2/(4t)}}{\sqrt{\cosh s - \cosh r}}
\, ds.
$$
## Lorentz Model

Given the hyperboloid model
$$
\begin{aligned}
\mathbb H^d
= \left\{
z \in \mathbb R^{d+1} :
\langle z, z \rangle_L=-1,\ z_0>0
\right\},
\end{aligned}
$$
the hyperbolic distance is
$$
d_{\mathbb H}(z_1, z_2)=\operatorname{arcosh} \bigl( -\langle z_1, z_2 \rangle_L \bigr).
$$
where the Lorentz inner product $<\cdot, \cdot>_{L}$
$$
\begin{aligned}
\langle u,v\rangle_L
= - u_0 v_0 + \sum_{i=1}^{d} u_i v_i = u^\top J v,
\qquad
J = \operatorname{diag}(-1,1,\dots,1).
\end{aligned}
$$
The Busemann inner product $\langle \cdot, \cdot \rangle_{\mathbb{H}}$ between boundary $x$ and disk $z$
$$
\langle x, z \rangle_{\mathbb{H}} 
= \log \frac{- \langle z, x \rangle_{L}}{- \langle O, x \rangle_{L}}
$$
, hence, the Poison kernel is
$$
q_{\infty | t}(x | z) = \exp((d - 1) \langle x, z \rangle_{\mathbb{H}}) 
= \left( \frac{- \langle O, x \rangle_{L}}{- \langle z, x \rangle_{L}} \right)^{d-1} \\
= \frac{1}{\left( - \langle z, x \rangle_{L} \right)^{d-1}}
$$
where the origin point $O = (1,0, \dots,0)$

The tangent space and Lorentz-orthogonal projection are
$$
T_x\mathbb H^d
=
\{v \in \mathbb R^{d+1} : \langle x,v\rangle_L = 0\},
\qquad
\Pi_x = I_{d+1}+x x^\top J.
$$
The $d$-dimensional hyperbolic heat kernel $P_{\mathbb{H}^{d}}$  with $d = 2$:
$$
P_{\mathbb{H}^2}(r;t)
=
\frac{\sqrt{2}\, e^{-t/4}}{(4\pi t)^{3/2}}
\int_r^\infty
\frac{s e^{-s^2/(4t)}}{\sqrt{\cosh s - \cosh r}}
\, ds.
$$
## Posterior

Let's define a Brownian motion $z$, the probability density $q_t(z)$ of the Brownian motion is
$$
\begin{aligned}
q_{t}(z) 
& = P_{\mathbb{H}}(d_{\mathbb{H}}(z, 0); t) \\
\end{aligned}
$$
and the posterior $q_{\infty | t}(x | z)$ can be expressed by the Poisson kernel
$$
\begin{aligned}
q_{\infty | t}(x | z) 
& = \exp((d-1) \langle x, z \rangle_{\mathbb{H}}) \\
\end{aligned}
$$
By Byesian rule / Doob-$h$ identity, the likelihood can be expressed as $q_{t | \infty}(z | x)$, considering the denominator $q_{\infty}(x)$ is a uniform distribution, ignored
$$
\begin{aligned}
q_{t | \infty}(z | x) 
& = \frac{q_{\infty | t}(x | z) q_{t}(z)}{q_{\infty}(x)} = P_{\mathbb{H}}(d_{\mathbb{H}}(z, 0); t) \cdot \exp((d-1) \langle x, z \rangle_{\mathbb{H}}) \\
\end{aligned}
$$
The transition $q_{t | s}(z_{t} | z_{s})$ can be represented as
$$
\begin{aligned}
q_{t | s}(z_{t} | z_{s}) 
& = P_{\mathbb{H}}(d_{\mathbb{H}}(z_{t}, z_{s}); t - s) \\
\end{aligned}
$$
The Brownian bridge $q(z_{t} | z_{s}, x)$ can be expressed as
$$
\begin{aligned}
q(z_{t} | z_{s}, x) 
& = \frac{q_{t | s}(z_{t} | z_{s}) q_{\infty | t}(x | z_{t})}{q_{\infty | s}(x | z_{s})} 
= \frac{P_{\mathbb{H}}(d_{\mathbb{H}}(z_t, z_s); t - s) \cdot \exp((d-1) \langle x, z_t \rangle_{\mathbb{H}})}{\exp((d-1) \langle x, z_s \rangle_{\mathbb{H}})}
\end{aligned}
$$

$$
\begin{aligned}
f_{\theta}(z_s) 
& := \operatorname{softmax}\left( f_{\theta}(z_s) + \sum_{v \in V} e_{v}(d-1) \langle v, z_s \rangle_{\mathbb{H}} \right) \\
P_{\theta}(z_t | z_s) 
& = q(z_{t} | z_{s}, x = f_{\theta}(z_s)) \\
& = \sum_{v \in V} \frac{\exp(f_{\theta}(z_s) + (d-1) \langle v, z_s \rangle_{\mathbb{H}})}{\sum_{u \in V} \exp(f_{\theta}(z_s)^{\top} e_{u} + (d-1) \langle u, z_s \rangle_{\mathbb{H}})} \exp((d-1) \langle v, z_t \rangle_{\mathbb{H}} - (d-1) \langle v, z_s \rangle_{\mathbb{H}}) \cdot P_{\mathbb{H}}(d_{\mathbb{H}}(z_t, z_s); t - s) \\
& = \sum_{v \in V} \frac{\exp(f_{\theta}(z_s) + (d-1) \langle v, z_t \rangle_{\mathbb{H}})}{\sum_{u \in V} \exp(f_{\theta}(z_s)^{\top} e_{u} + (d-1) \langle u, z_s \rangle_{\mathbb{H}})} P_{\mathbb{H}}(d_{\mathbb{H}}(z_t, z_s); t - s) \\
\end{aligned}
$$

> Why there is $\exp((d-1) \langle v, z_t \rangle_{\mathbb{H}} - (d-1) \langle v, z_s \rangle_{\mathbb{H}}) \cdot P_{\mathbb{H}}(d_{\mathbb{H}}(z_t, z_s); t - s)$ ?

## Sample from Heat Kernel
Let's define a Brownian motion $z$, the probability density $q_t(z)$ of the Brownian motion is
$$
\begin{aligned}
q_{t}(z) 
& = P_{\mathbb{H}}(d_{\mathbb{H}}(z, 0); t) \\
\end{aligned}
$$
The sampling code for Poincare disk is ``sample_chi``
```python
def sample_chi(ns, dtype=torch.float64):
	# chi(n) = sqrt(chi^2(n)), and chi^2(n) ~ Gamma(shape=n/2, scale=2).
	# Sampling Gamma directly avoids allocating sum(ns) standard normals,
	# which blows up when ns is large.
	concentration = ns.to(dtype) / 2
	rate = torch.tensor(0.5, device=ns.device, dtype=dtype)
	chi2 = torch.distributions.Gamma(concentration, rate).sample()
	return chi2.sqrt()
```

The sampling code for Lorentz model is ``binary_bridge_lorentz``
```python
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
```

---
## Continuous-Time ELBO Loss

### Continuous-Time ELBO

### Parametrization
Since the prior $q_{\infty | t}(x | z_t)$, according to the Bayesian rule, the prior can be re-parametrized by
$$
q_{\infty | t}(x | z_t) = \frac{q_{\infty}(x) q_{t | \infty}(z_t | x)}{q_t(z_t)},
$$
Therefore, $q_{\infty | t}(x | z_t) \propto q_{\infty}(x) q_{t | \infty}(z_t | x)$, for hyperbolic geometric, $q_{t | \infty}(z_t | x)$ can be computed by the Poison kernel. Also, we let the model $f_{\theta}(z_t)$ to output the token distribution $q_{\infty}(x)$. As a result, we set the model parametrization as
$$
\exp(f_{\theta}(z_t) + (d-1) \langle x, z_t \rangle_{\mathbb{H}})
$$
Consider the normalization, we use $\operatorname{softmax}$ to normalize the model re-parametrization, we let $\mu^{\theta}$ as
$$
\mu^{\theta}(z_t) 
:= \operatorname{softmax}(f_{\theta}(z_t) + (d-1) \sum_{v \in V} e_v \langle v, z_t \rangle_{\mathbb{H}})
$$
Advantages of parametrization, compared with drift matching
- Architecture matches standard transformers / LMs
- Numerical stability near the boundary, no need to learn the scale of the drift
- Valid geometric automatically
- Enable variance reduction, e.g. Rao-Blackwellization
### Cartesian ELBO Loss for Poincare Disk
Consider the Brownian bridge on the local chart of Poincare Disk,
$$
\begin{aligned}
dx_t & = \left( \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{1-\|x_t\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$
Instead of matching the drift directly, we choose $\mu^{\theta}$ re-parametrization, by eliminate the parts irrelevant to the boundary points $x$, the loss function can be written as
$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
\frac{(d-1)^2}{2}
\mathbb{E}_{z_{t} \sim q_{t \mid \infty}(\cdot \mid y)}
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
If we further consider the importance sampling from a proposal distribution $t_i \sim \pi$, the loss can be represented as
$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
\frac{(d-1)^2}{2}
\sum_{i=1}^{N}
w(t_i) \mathbb{E}_{z_{t_i} \sim q_{t_i \mid \infty}(\cdot \mid y)}
\left[
	\int_0^\infty
	(1 - \| z_{t_i} \|^2)^2
	\left\|
	\frac{y - z_{t_i}}{\| y - z_{t_i} \|^2}
	-
	\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_t)} \left[
	\frac{v - z_{t_i}}{\| v - z_{t_i} \|^2} \right]
	\right\|^2
	dt
\right] \\
\end{aligned}
$$
where $w(t_i)$ is the importance sampling weight of the timestep $t_i$, $y$ is the boundary points, $z_t$ is the bridge at timestep $t$.
### Cartesian ELBO Loss for Lorentz Model
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
where  $<\cdot, \cdot>_{L}$ is Lorentz inner product and $\xi(y)$ is $[1, y_1, y_2, \dots, y_d]$

Instead of matching the drift directly, we choose $\mu^{\theta}$ re-parametrization, by eliminate the parts irrelevant to the boundary points $x$, the loss function can be written as
$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
-(d-1)
\mathbb{E}_{z_{t} \sim q_{t \mid \infty}(\cdot \mid y)}
\left[
	\int_0^\infty
	\left\|
	\frac{\xi(y)}{\langle z_t, \xi(y) \rangle_{L}}
	-
	\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_t)} \left[
	\frac{\xi(v)}{\langle z_t, \xi(v) \rangle_{L}} \right]
	\right\|^2
	dt
\right] \\
\mu^{\theta}(z_t) 
& = \operatorname{softmax} \big( f_{\theta}(z_t) + (d-1) \sum_{v \in V} e_v \log \frac{- \langle z, x \rangle_{L}}{- \langle O, x \rangle_{L}} \big)
\end{aligned}
$$
If we further consider the importance sampling from a proposal distribution $t_i \sim \pi$, the loss can be represented as
$$
\begin{aligned}
\mathcal{L}(\theta; y)
& =
-(d-1)
\sum_{i=1}^{N} w(t_i)
\mathbb{E}_{z_{t_i} \sim q_{t_i \mid \infty}(\cdot \mid y)}
\left[
	\int_0^\infty
	\left\|
	\frac{\xi(y)}{\langle z_{t_i}, \xi(y) \rangle_{L}}
	-
	\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_{t_i})} \left[
	\frac{\xi(v)}{\langle z_{t_i}, \xi(v) \rangle_{L}} \right]
	\right\|^2
	dt
\right] \\
\end{aligned}
$$
where $w(t_i)$ is the importance sampling weight of the timestep $t_i$, $y$ is the boundary points, $z_t$ is the bridge at timestep $t$.
### Proposal Distribution Generation

### Exponential Distribution
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
Since $1-u \overset{d}{=} u$, you can equivalently use $t = -\ln(u)/\lambda$. That means $t_1$ and $t_2$ follow the same distributions but $t_1$ is different from $t_2$ for a given $u$
$$
t_1 = -\frac{\ln(1-u)}{\lambda}, \quad t_2 = -\frac{\ln(u)}{\lambda}
$$

---
#### Brownian Bridge on Poincare Disk with Local Chart Projection
Given a target point $y$ at the boundary of the Poincare Disk $||y|| = 1$, the Brownian bridge with $t: \infty \to 0$ is described as
$$dx_t = \left( f(x_t, t) + \sigma^2(x_t, t) \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{\sigma^2(x_t, t) d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{\sigma(x_t, t) (1-\|x_t\|^2)}{2} d\bar{W}_t$$
Therefore, $q(y | x_t)$ is
$$
q(y | x_t) = \frac{1}{A_{d-1}} \left( \frac{1 - \|x_t\|^2}{\|x_t - y\|^2} \right)^{d-1}
$$
where $A_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ is the surface area of $(d-1)$-dimensional unit sphere