### Brownian Motion on Manifold
Given a d-dimensional SDE as $d z_t = f(z_t, t) dt + \sigma(t) dw$ on the hyperbolic space $\mathbb{H}^d$, the SDE on the local chart of Poincare Disk $\mathbb{D}^d$ is described as
$$dx_t = \left( f(x_t, t) + \frac{\sigma^2 (d-2) (1 - \|x_t\|^2)}{4} x_t \right) dt + \frac{\sigma (1 - \|x_t\|^2)}{2} dW_t$$
> Proof: Derive the SDE on the local chart of Poincare Disk

Given a target point $y$ at the boundary of the Poincare Disk $||y|| = 1$, the Brownian bridge is described as
$$dx_t = \left( f(x_t, t) + \sigma^2(t) \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{\sigma^2(t) d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{\sigma(t) (1-\|x_t\|^2)}{2} d\bar{W}_t$$
Therefore, $q(y | x_t)$ is
$$
q(y | x_t) = \frac{1}{A_{d-1}} \left( \frac{1 - \|x_t\|^2}{\|x_t - y\|^2} \right)^{d-1}
$$
where $A_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ is the surface area of $(d-1)$-dimensional unit sphere
> Proof: derive the conditional probability q(y | x_t) by using horospherical distance
### NELBO in Continuous Space
For a probability model $p_{\theta}$ and a Brownian motion $x_{1:T}$, the NELBO of the Brownian bridge diffusion conditioned on $x_0$ is following.
$$
\begin{aligned}
- \log p_{\theta}(x_0) 
& \leq \mathbb{E}_{q(x_{0:T})} \Big[ \log\frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \Big] \\
& = \mathbb{E}_q [\underbrace{D_\text{KL}(q(x_T \vert x_0) \parallel p_\theta(x_T))}_{\text{Prior}} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(x_{t-1} \vert x_t, x_0) \parallel p_\theta(x_{t-1} \vert x_t))}_{\text{Diffusion}} \underbrace{- \log p_\theta(x_0 \vert x_1)}_{\text{Reconst}} ]
\end{aligned}
$$
### NELBO in Discrete Space

Note that the KL divergence between isotropic Gaussian distribution $p=\mathcal N(\mu_0,\sigma_0^2 I)$ and $q=\mathcal N(\mu_1,\sigma_1^2 I)$ is
$$
\boxed{ D_{\mathrm{KL}}(p\|q) = \frac12 \left[ d\log\frac{\sigma_1^2}{\sigma_0^2} -d +d\frac{\sigma_0^2}{\sigma_1^2} +\frac{\|\mu_1-\mu_0\|^2}{\sigma_1^2} \right]. }
$$
### Poincare Disk Brownian Bridge Diffusion
Given 2 Brownian bridge $\{x_t\} \subset T_{x_t} \mathbb{D}^d$ and $\{x_t'\} \subset T_{x_t} \mathbb{D}^d$ on the local chart of Poincare Disk $\mathbb{D}^d$, and conditioned on a target point $y \in \mathbb{R}^d$ drawn from training dataset $y \in Y$, assume 
- (1) in the continuous bridge limit, both bridges start from the origin at time $\infty$
- (2) the discrete schedule has fixed step size $\tau(i) - \tau(i-1) = \Delta t \ll 1$ for all $i$
and target points $|| y || = 1$ are at the boundary of Poincare disk $\mathbb{D}^d = \{ x \in \mathbb{R}^d : \|x\| < 1 \}$. 
$$
\begin{aligned}
dx_t & = \left( \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{1-\|x_t\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$
Consider a model $f_{\theta}:  T_{x_t} \mathbb{D}^d \times \mathbb{R} \to \mathbb{R}^d$ would take the state $x_t'$ of the Brownian bridge $\{x_t'\}$ and predict the final target $y$, the bridge learned by the model $f_{\theta}$ is
$$
\begin{aligned}
dx_t' & = \left( \frac{d-1}{2} \frac{(1-\|x_t'\|^2)^2}{\|f_{\theta}(x_t', t)-x_t'\|^2} (f_{\theta}(x_t', t)-x_t') - \frac{d}{4} (1-\|x_t'\|^2) x_t' \right) dt + \frac{1-\|x_t'\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$
Both bridges can be expressed as $dx_t = f(t, x_t, y) dt + g(t, x_t) d \bar{W}_t$ with $f: \mathbb{R} \times T_{x_t} \mathbb{D}^{d} \times \mathbb{R}^{d} \to T_{x_t} \mathbb{D}^d$ and $g: \mathbb{R} \times T_{x_t} \mathbb{D}^{d} \to \mathbb{R}$. Let $\{\tau(i)\}_{i=0}^{T}$ be a strictly increasing discretization of reverse time with
$$
0 = \tau(0) < \tau(1) < \cdots < \tau(T) < \infty, \qquad \tau(i) - \tau(i-1) = \Delta t \text{ for all } i.
$$
Since $\tau(T) = T \Delta t$ under the uniform-step schedule, the continuous bridge limit is recovered by letting $T \to \infty$, $\Delta t \to 0$, and hence $\tau(T) = T \Delta t \to \infty$. Then the local Euler-Gaussian approximation of the one-step posterior is
$$
\begin{aligned}
q(x_{\tau(i-1)} \mid x_{\tau(i)}, y)
& \approx \mathcal{N}\!\big(x_{\tau(i)} - f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 \Delta t I\big), \\
p_\theta(x_{\tau(i-1)} \mid x_{\tau(i)})
& \approx \mathcal{N}\!\big(x_{\tau(i)} - f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 \Delta t I\big).
\end{aligned}
$$
where $f(t, x_t, y) = \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t$ and $g(t, x_t) = \frac{1-\|x_t\|^2}{2}$.
### Local-Chart Approximated NELBO in Continuous Space with Small Step
This derivation is a local-chart, small-step approximate NELBO induced by the Euler-Gaussian approximation above; it is not the exact manifold ELBO. The resulting approximation to the NLL $-\log p_{\theta}(y)$ is
$$
\boxed{
\mathcal{L}_{\mathrm{NELBO}}(y)
\approx
L_T + \sum_{i=2}^{T} L_{i-1} + L_0
}
$$
with
$$
\begin{aligned}
L_T
& := D_{\mathrm{KL}}(q(x_{\tau(T)} \mid y) \parallel p_\theta(x_{\tau(T)})), \\
L_{i-1}
& := \mathbb{E}_{q(x_{\tau(i)} \mid y)}
\left[
\frac{\Delta t}{2 g(\tau(i), x_{\tau(i)})^2}
\Big\| f(\tau(i), x_{\tau(i)}, y) - f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Big\|^2
\right], \\
L_0
& := \mathbb{E}_{q(x_{\tau(1)} \mid y)}
\left[
\frac{d}{2} \log 2 \pi
+ d \log g(\tau(1), x_{\tau(1)})
+ \frac{d}{2} \log \Delta t
+ \frac{1}{2 g(\tau(1), x_{\tau(1)})^2 \Delta t}
\Big\| y - x_{\tau(1)} + f(\tau(1), x_{\tau(1)}, f_\theta(x_{\tau(1)}, \tau(1))) \Delta t \Big\|^2
\right].
\end{aligned}
$$
For $g(t, x_t) = \frac{1-\|x_t\|^2}{2}$, the terms become
$$
\boxed{
L_T = 0
}
$$
$$
\boxed{
L_{i-1}
=
\mathbb{E}_{q(x_{\tau(i)} \mid y)}
\left[
\frac{\Delta t}{2} (d-1)^2 (1-\|x_{\tau(i)}\|^2)^2
\left\|
\frac{y-x_{\tau(i)}}{\|y-x_{\tau(i)}\|^2}
-
\frac{f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}}{\|f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}\|^2}
\right\|^2
\right].
}
$$
$$
\begin{aligned}
\boxed{
L_{0} = 
\frac{d}{2} \log 2 \pi 
+ d \log \left( \frac{1-\|x_{\tau(1)}\|^2}{2} \right)
+ \frac{d}{2} \log \Delta t
  + \frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta t}
\Bigg\| y - x_{\tau(1)}
+ \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2}
\big(f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta t \Bigg\|^2
}
\end{aligned}
$$
#### Prior $L_T$
The prior term is zero only under an explicit terminal-law assumption. If we set
$$
q(x_{\tau(T)} \mid y) = p_\theta(x_{\tau(T)}) = \delta_0,
$$
then
$$
L_T = D_\text{KL}(q(x_{\tau(T)} \mid y) \parallel p_\theta(x_{\tau(T)})) = 0.
$$
#### Diffusion $\sum_{i=2}^{T} L_{i-1}$
Using the provided KL divergence formula for isotropic Gaussians $p$ and $q$, because the variances are identical ($\sigma_0^2 = \sigma_1^2$), the logarithmic and trace terms exactly cancel out ($d \log(1) - d + d(1) = 0$). This leaves only the scaled squared Euclidean distance between the means:
$$D_{\mathrm{KL}}(q \parallel p_\theta) = \frac{1}{2} \frac{\|\mu_p - \mu_q\|^2}{\sigma^2}$$
Substituting the means and variance:
$$
\begin{aligned}
D_{\mathrm{KL}}(q \parallel p_\theta) 
& = \frac{1}{2 g(\tau(i), x_{\tau(i)})^2 \Delta t}
\Big\|
\big(x_{\tau(i)} - f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Delta t\big)
- \big(x_{\tau(i)} - f(\tau(i), x_{\tau(i)}, y) \Delta t\big)
\Big\|^2 \\
& = \frac{\Delta t}{2 g(\tau(i), x_{\tau(i)})^2}
\Big\| f(\tau(i), x_{\tau(i)}, y) - f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Big\|^2 \\
& = \frac{2 \Delta t}{(1-\|x_{\tau(i)}\|^2)^2}
\left[
\frac{(d-1)^2}{4} (1-\|x_{\tau(i)}\|^2)^4
\left\|
\frac{y-x_{\tau(i)}}{\|y-x_{\tau(i)}\|^2}
- \frac{f_{\theta}(x_{\tau(i)}, \tau(i))-x_{\tau(i)}}{\|f_{\theta}(x_{\tau(i)}, \tau(i))-x_{\tau(i)}\|^2}
\right\|^2
\right] \\
& = \frac{\Delta t}{2} (d-1)^2 (1-\|x_{\tau(i)}\|^2)^2
\left\|
\frac{y-x_{\tau(i)}}{\|y-x_{\tau(i)}\|^2}
- \frac{f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}}{\|f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}\|^2}
\right\|^2
\end{aligned}
$$
#### Reconst $L_0$
Consider the multivariate Gaussian distribution as
$$
\mathcal N(y;\mu,\Sigma) = \frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}} \exp\!\left( -\frac12 (y-\mu)^\top \Sigma^{-1}(y-\mu) \right).
$$
As we assume the distribution is a isotropic Gaussian, the negative log is
$$
\begin{aligned}
- \log \mathcal N(y;\mu, \sigma^2 I)
& = - \log \frac{1}{(2\pi)^{d/2} \sigma^d} \exp \left( -\frac{1}{2 \sigma^2} (y-\mu)^\top (y-\mu) \right) \\
& = \frac{d}{2} \log 2 \pi + d \log \sigma + \frac{1}{2 \sigma^2} || y-\mu ||^2. \\
\end{aligned}
$$
Therefore, the reconst term should be
$$
\begin{aligned}
- \log p_\theta(y \vert x_{\tau(1)})
& = - \log \mathcal{N}\!\big(y;\, x_{\tau(1)} - f(\tau(1), x_{\tau(1)}, f_{\theta}(x_{\tau(1)}, \tau(1))) \Delta t,\; g(\tau(1), x_{\tau(1)})^2 \Delta t I\big) \\
& = \frac{d}{2} \log 2 \pi
+ d \log g(\tau(1), x_{\tau(1)})
+ \frac{d}{2} \log \Delta t \\
& \quad + \frac{1}{2 g(\tau(1), x_{\tau(1)})^2 \Delta t}
\Big\| y - x_{\tau(1)} + f(\tau(1), x_{\tau(1)}, f_{\theta}(x_{\tau(1)}, \tau(1))) \Delta t \Big\|^2 \\
& = \frac{d}{2} \log 2 \pi 
+ d \log \left( \frac{1-\|x_{\tau(1)}\|^2}{2} \right)
+ \frac{d}{2} \log \Delta t \\
& \quad + \frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta t}
\Bigg\| y - x_{\tau(1)}
+ \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2}
\big(f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta t \Bigg\|^2
\end{aligned}
$$
### NELBO in Discrete Space

### Training Loss
4 different surrogate loss that can try
- Cross Entropy surrogate
$$
\mathcal{L}_{\mathrm{CE}}(Y, \theta)
=
\mathbb{E}_{i \sim \mathrm{Unif}(\{2,\ldots,T\}),\, y \sim Y,\, x_{\tau(i)} \sim q(\cdot \mid y)}
\left[
\mathrm{CE}(y \mathbf{E}^{\top}, \mathrm{softmax}(f_\theta(x_{\tau(i)}, \tau(i)) \mathbf{E}^{\top}))
\right]
$$
- L2 surrogate
$$
\mathcal{L}_{\mathrm{L2}}(Y, \theta)
=
\mathbb{E}_{i \sim \mathrm{Unif}(\{2,\ldots,T\}),\, y \sim Y,\, x_{\tau(i)} \sim q(\cdot \mid y)}
\left[
\left\| y - f_\theta(x_{\tau(i)}, \tau(i)) \right\|_2^2
\right]
$$
- Diffusion Loss Surrogate
$$
\widehat{\mathcal{L}}_{\mathrm{diff}}(Y, \theta)
=
(T-1)\,
\mathbb{E}_{i \sim \mathrm{Unif}(\{2,\ldots,T\}),\, y \sim Y,\, x_{\tau(i)} \sim q(\cdot \mid y)}
\left[
\frac{\Delta t}{2} (d-1)^2 (1-\|x_{\tau(i)}\|^2)^2
\left\|
\frac{y-x_{\tau(i)}}{\|y-x_{\tau(i)}\|^2}
-
\frac{f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}}{\|f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}\|^2}
\right\|_2^2
\right]
$$
- NELBO
$$
\mathcal{L}_{\mathrm{NELBO}}(Y, \theta) = \mathcal{L}_{\mathrm{diff}}(Y, \theta) + \mathcal{L}_{0}(Y, \theta)
$$
	- $\mathcal{L}_0$ is Reconstruction term
$$
\mathcal{L}_{0}(Y, \theta)
=
\mathbb{E}_{y \sim Y,\, x_{\tau(1)} \sim q(\cdot \mid y)}
\left[
\frac{d}{2} \log 2 \pi 
+ d \log \left( \frac{1-\|x_{\tau(1)}\|^2}{2} \right)
+ \frac{d}{2} \log \Delta t
+ \frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta t}
\Bigg\| y - x_{\tau(1)}
+ \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2}
\big(f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta t \Bigg\|^2
\right]
$$
### Training Algorithm
We choose $T = 1000$ and $\Delta t = 0.01$
Given training dataset $Y$, a discrete schedule $\{\tau(i)\}_{i=0}^{T}$, and a word embedding $E \in \mathbb{R}^{V \times d}$:
For each data point $y \in Y$ drawn from the training dataset $Y$,
- Sample or simulate bridge states $\{x_{\tau(i)}\}_{i=1}^{T}$ from $q(\cdot \mid y)$. One Euler-Maruyama step is
$$
x_{\tau(i-1)} =
x_{\tau(i)}
- f(\tau(i), x_{\tau(i)}, y) \Delta t
+ g(\tau(i), x_{\tau(i)}) \sqrt{\Delta t}\,\varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N(0, I).
$$
- If using a surrogate objective, sample $i \in \{2, \ldots, T\}$ and compute one of the following:
	- Cross Entropy
$$
\mathcal{L}_{\mathrm{CE}}(Y, \theta)
=
\mathbb{E}_{i \sim \mathrm{Unif}(\{2,\ldots,T\}),\, y \sim Y,\, x_{\tau(i)} \sim q(\cdot \mid y)}
\left[
\mathrm{CE}(y \mathbf{E}^{\top}, \mathrm{softmax}(f_\theta(x_{\tau(i)}, \tau(i)) \mathbf{E}^{\top}))
\right]
$$
	- L2 Norm
$$
\mathcal{L}_{\mathrm{L2}}(Y, \theta)
=
\mathbb{E}_{i \sim \mathrm{Unif}(\{2,\ldots,T\}),\, y \sim Y,\, x_{\tau(i)} \sim q(\cdot \mid y)}
	\left[
	\left\| y - f_\theta(x_{\tau(i)}, \tau(i)) \right\|_2^2
	\right]
	$$
	- Diffusion
	$$
	\mathcal{L}_{\mathrm{diff}}(Y, \theta)
	=
	(T-1)\,
	\mathbb{E}_{i \sim \mathrm{Unif}(\{2,\ldots,T\}),\, y \sim Y,\, x_{\tau(i)} \sim q(\cdot \mid y)}
	\left[
	\frac{\Delta t}{2} (d-1)^2 (1-\|x_{\tau(i)}\|^2)^2
	\left\|
	\frac{y-x_{\tau(i)}}{\|y-x_{\tau(i)}\|^2}
	-
	\frac{f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}}{\|f_\theta(x_{\tau(i)}, \tau(i))-x_{\tau(i)}\|^2}
	\right\|_2^2
	\right]
	$$
	- NELBO
$$
\mathcal{L}_{\mathrm{NELBO}}(Y, \theta) = \mathcal{L}_{\mathrm{diff}}(Y, \theta) + \mathcal{L}_{0}(Y, \theta)
$$
- Update the model weight with $\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}_{*}(Y, \theta)$, where $\mathcal{L}_{*}$ is either a surrogate objective.

### Inference Algorithm
- If the terminal law is fixed as $p_\theta(x_{\tau(T)}) = \delta_0$, initialize $x_{\tau(T)}' = 0$.
- For $i = T, T-1, \ldots, 1$, predict a boundary point
$$
\hat{y}_i = \frac{f_\theta(x_{\tau(i)}', \tau(i))}{\|f_\theta(x_{\tau(i)}', \tau(i))\|},
$$
and take the same reverse Euler-Maruyama step
$$
\begin{aligned}
x_{\tau(i-1)}'
& =
x_{\tau(i)}'
- \left(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(i)}'\|^2)^2}{\|\hat{y}_i-x_{\tau(i)}'\|^2}
(\hat{y}_i-x_{\tau(i)}')
- \frac{d}{4} (1-\|x_{\tau(i)}'\|^2) x_{\tau(i)}'
\right) \Delta t \\
& \qquad + \frac{1-\|x_{\tau(i)}'\|^2}{2} \sqrt{\Delta t}\,\varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N(0, I).
\end{aligned}
$$
- After the sampling loop, normalize the final state to the boundary:
$$
y' := \frac{x_{\tau(0)}'}{\|x_{\tau(0)}'\|}.
$$