
Given a Poincare disk $\mathbb{D}^{d} = \{ z \in \mathbb{R}^{d}: || z || < 1 \}$. The hyperbolic distance $d_{\mathbb{H}}$ is defined as
$$
d_{\mathbb{H}} (z_1, z_2) = \operatorname{arcosh} \left( 1 + \frac{2 || z_1 - z_2 ||^2}{(1 - || z_1 ||^2) (1 - || z_2 ||^2)} \right)
$$
while the distance from the origin $O$ is defined as
$$
d_{\mathbb{H}} (z, 0) = \operatorname{arcosh} \left( \frac{1 + || z ||^2}{1 - || z ||^2} \right) = 2 \operatorname{arcosh} \left( || z || \right)
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


#### Brownian Bridge on Poincare Disk with Local Chart Projection
Given a target point $y$ at the boundary of the Poincare Disk $||y|| = 1$, the Brownian bridge with $t: \infty \to 0$ is described as
$$dx_t = \left( f(x_t, t) + \sigma^2(x_t, t) \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{\sigma^2(x_t, t) d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{\sigma(x_t, t) (1-\|x_t\|^2)}{2} d\bar{W}_t$$
Therefore, $q(y | x_t)$ is
$$
q(y | x_t) = \frac{1}{A_{d-1}} \left( \frac{1 - \|x_t\|^2}{\|x_t - y\|^2} \right)^{d-1}
$$
where $A_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ is the surface area of $(d-1)$-dimensional unit sphere

### Poincare Disk Brownian Bridge Diffusion
Given 2 Brownian bridge $q(x_{0:T} | y), \{x_t\} \subset T_{x_t} \mathbb{D}^d$ and $p_{\theta}(x_{0:T} | y), \{x_t\} \subset T_{x_t} \mathbb{D}^d$ on the local chart of Poincare Disk $\mathbb{D}^d$, and conditioned on a target point $y \in \mathbb{R}^d$ drawn from training dataset $y \in Y$, assume 
- (1) in the continuous bridge limit, both bridges start from the origin at time $\infty$
- (2) the discrete schedule has fixed step size $\tau(i) - \tau(i-1) = \Delta t \ll 1$ for all $i$
and target points $|| y || = 1$ are at the boundary of Poincare disk $\mathbb{D}^d = \{ x \in \mathbb{R}^d : \|x\| < 1 \}$. 

The Brownian bridge on the local chart of Poincare Disk,
$$
\begin{aligned}
dx_t & = \left( \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{1-\|x_t\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$

$$
\begin{aligned}
dx_t & = \left( \frac{(d-1) (1-\|x_t\|^2)^2}{2} \mathbb{E}_{y | x_t} \left[ \frac{y - x_t}{||y-x_t ||^2} \right] - \frac{d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{1-\|x_t\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$

$$
\mathbb{E}_{y | x_t} \left[ \frac{y - x_t}{||y-x_t ||^2} \right] 
= \frac{1}{A_{d-1}} \left( \sum_{y \in V} \left( \frac{1 - \|x_t\|^2}{\|x_t - y\|^2} \right)^{d-1} \frac{y - x_t}{||y-x_t ||^2} \right)
$$

$$
q(y | x_t​) \approx \frac{p(y) q(x_t ​| y)}{q(x_t)}
$$
## What the optimal logits actually are
The bridge loss assumes
$$\mu = \mathrm{softmax}\!\bigl(\text{horosphere\_dists}_y + \text{logits}_y\bigr)$$
recovers the true posterior $q(y \mid x_t)$. For $d = 2$, $\text{horosphere\_dists}_y = \log\!\frac{1 - \|x_t\|^2}{\|y - x_t\|^2}$, and the true posterior is $q(y \mid x_t) \propto p(y)\, \|y - x_t\|^{-2(d-1)}$. Equating gives
$$e^{\text{logits}_y} \propto p(y) \quad\Longrightarrow\quad \text{logits}_y = \log p(y).$$
So the **optimal logits are the unconditional log-prior, broadcast across the batch** — the per-$x_t$ Poisson-kernel factor is already supplied by the loss via the horosphere distance term.
### Derivation: why `logits = log_ps` gives $softmax \propto p(y) / || y - x_t ||^2$

Start from the loss's softmax expression and substitute $\text{logits}_y = \log p(y)$:
$$
\mu_y
= \mathrm{softmax}_y\bigl(\text{horosphere\_dists}_y + \log p(y)\bigr)
= \frac{\exp\bigl(\text{horosphere\_dists}_y + \log p(y)\bigr)}
{\sum_{y'} \exp\bigl(\text{horosphere\_dists}_{y'} + \log p(y')\bigr)}.
$$
Plug in the closed-form horosphere distance for $d = 2$, $\text{horosphere\_dists}_y = \log\!\dfrac{1 - \|x_t\|^2}{\|y - x_t\|^2}$, so $\exp(\text{horosphere\_dists}_y) = \dfrac{1 - \|x_t\|^2}{\|y - x_t\|^2}$:
$$
\mu_y
= \frac{\dfrac{1 - \|x_t\|^2}{\|y - x_t\|^2} \cdot p(y)}
{\displaystyle\sum_{y'} \dfrac{1 - \|x_t\|^2}{\|y' - x_t\|^2} \cdot p(y')}.
$$
The factor $1 - \|x_t\|^2$ does not depend on $y$, so it cancels between numerator and denominator:
$$
\mu_y
= \frac{p(y) / \|y - x_t\|^2}
{\sum_{y'} p(y') / \|y' - x_t\|^2}
\;\propto\; \frac{p(y)}{\|y - x_t\|^2}.
$$
This is exactly the Bayes posterior $q(y \mid x_t) \propto p(y)\,\|y - x_t\|^{-2(d-1)}$ for $d = 2$  i.e., feeding `log_ps` as logits makes the loss's `softmax(horosphere + logits)` reproduce the true posterior, with the $y$-independent $(1 - \|x_t\|^2)^{d-1}$ factor absorbed by the softmax normalization.

# Why $q(y \mid x_t) \propto p(y)\, \|y - x_t\|^{-2(d-1)}$?
It's just Bayes plus dropping every factor that doesn't depend on $y$. Three cancellations.
## Step 1 — Bayes
$$
q(y\mid x_t)=\frac{p(y)\,q(x_t\mid y)}{q(x_t)}\;\propto\;p(y)\,q(x_t\mid y),
$$
since $q(x_t)=\sum_{y'}p(y')q(x_t\mid y')$ is a $y$-independent normalizer.
## Step 2 — Express $q(x_t \mid y)$ via the Doob-$h$ identity
The bridge density (= free hyperbolic BM density conditioned on exiting at $y$) factors as
$$
q(x_t\mid y)=\frac{q_t(x_t)\,q_{\infty\mid t}(y\mid x_t)}{q_\infty(y)},
$$where:
- $q_t(x_t)=P_{\mathbb H}(d_{\mathbb H}(x_t,0);t)$ is the free-BM marginal — depends on $\|x_t\|$ and $t$, **not on $y$**;
- $q_\infty(y)$ is the boundary exit measure — by rotational symmetry it is **uniform on the sphere**, so it does not depend on $y$;
- $q_{\infty\mid t}(y\mid x_t)$ is the Poisson kernel,
$$
q_{\infty\mid t}(y\mid x_t)=e^{(d-1)\langle y,x_t\rangle_{\mathbb H}}=\left(\dfrac{1-\|x_t\|^2}{\|y-x_t\|^2}\right)^{d-1}.
$$
Plugging back:
$$
q(y\mid x_t)\;\propto\;p(y)\,q(x_t\mid y)\;=\;p(y)\,\frac{q_t(x_t)}{q_\infty(y)}\,\left(\dfrac{1-\|x_t\|^2}{\|y-x_t\|^2}\right)^{d-1}.
$$
## Step 3 — Drop everything constant in $y$
- $q_t(x_t)$: no $y$ → drop.
- $q_\infty(y)$: uniform, equal for all $y$ → drop.
- $(1-\|x_t\|^2)^{d-1}$: no $y$ → drop.
Only $\|y-x_t\|^{-2(d-1)}$ remains:
$$
\boxed{\;q(y\mid x_t)\;\propto\;p(y)\,\|y-x_t\|^{-2(d-1)}.\;}
$$
## Why "$y$-independent" really means independent
A common slip: $q_t(x_t)$ "looks like it depends on the bridge endpoint $y$ because we conditioned on $y$," but it doesn't — it's the **free** hyperbolic BM marginal, the un-conditioned density. The conditioning on $y$ is entirely captured by the Poisson-kernel factor $q_{\infty\mid t}(y\mid x_t)/q_\infty(y)$, which is the Doob-$h$ correction. Once you separate that off, every other factor is the same for all $y \in V$ and gets absorbed into the proportionality.

That is also why the **softmax** form
$$
\mu_v(x_t)=\operatorname{softmax}_v\!\big(\log p_v+(d-1)\langle v,x_t\rangle_{\mathbb H}\big)
$$
gives the *exact* Bayes posterior: the softmax denominator implicitly performs all three drops above, regardless of any common $y$-independent factor inside the exponent.

---


$$
\mathcal{L}(\theta; y)
=
\frac{(d-1)^2}{2}
\mathbb{E}_{z_t \sim q_{t \mid \infty}(\cdot \mid y)}
\int_0^\infty
(1 - \|z_t\|^2)^2
\left\|
\frac{y - z_t}{\|y - z_t\|^2}
-
\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_t)} \left[ 
\frac{v - z_t}{\|v - z_t\|^2} \right]
\right\|^2
dt
$$
where $\mu_v(x_t)=\operatorname{softmax}_v\!\big(\log p_v+(d-1)\langle v,x_t\rangle_{\mathbb H}\big)$

> Why taking expectation over model prediction $\mathbb{E}_{v \sim \mu^\theta(\cdot \mid z_t)} \left[ \frac{v - z_t}{\|v - z_t\|^2} \right]$ ?

It looks like a parametrization.


$$
\mathrm{softmax}_y\bigl(\text{horosphere\_dists}_y + \log p(y)\bigr) 
\propto \frac{p(y)}{\|y - x_t\|^2}
\propto q(y\mid x_t)
\propto \mathbb{E}_{v \sim p(y)} \left[ \frac{v - z_t}{\|v - z_t\|^2} \right]
$$
