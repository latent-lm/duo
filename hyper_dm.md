### Brownian Motion on Manifold
#### Brownian Motion on Poincare Disk with Local Chart Projection
Given a $d$-dimensional SDE as $d z_t = f(z_t, t) dt + \sigma(z_t, t) dw$ on the hyperbolic space $\mathbb{H}^d$, the SDE with $t: 0 \to \infty$ on the local chart of Poincare Disk $\mathbb{D}^d$ is described as
$$dx_t = \left( f(x_t, t) + \frac{\sigma(x_t, t)^2 (d-2) (1 - \|x_t\|^2)}{4} x_t \right) dt + \frac{\sigma(x_t, t) (1 - \|x_t\|^2)}{2} dW_t$$
> Proof: Derive the SDE on the local chart of Poincare Disk

#### Brownian Bridge on Poincare Disk with Local Chart Projection
Given a target point $y$ at the boundary of the Poincare Disk $||y|| = 1$, the Brownian bridge with $t: \infty \to 0$ is described as
$$dx_t = \left( f(x_t, t) + \sigma^2(x_t, t) \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{\sigma^2(x_t, t) d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{\sigma(x_t, t) (1-\|x_t\|^2)}{2} d\bar{W}_t$$
Therefore, $q(y | x_t)$ is
$$
q(y | x_t) = \frac{1}{A_{d-1}} \left( \frac{1 - \|x_t\|^2}{\|x_t - y\|^2} \right)^{d-1}
$$
where $A_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ is the surface area of $(d-1)$-dimensional unit sphere
> Proof: derive the conditional probability q(y | x_t) by using horospherical distance

- Use heat kernel to derive base motion
- Use Poisson kernel $K(x, y)$ for boundary condition
- Use Doob's h-transform for bridge drift
- Bridge marginal is $p(x_t | O) K(x, y)$

#### Brownian Motion on the Hyperboloid / Lorentz Model with Local Chart Projection
We use the hyperboloid model
$$
\begin{aligned}
\mathbb H^d
= \left\{
x \in \mathbb R^{d+1} :
\langle x,x\rangle_L=-1,\ x_0>0
\right\}
\end{aligned}
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
The tangent space and Lorentz-orthogonal projection are
$$
T_x\mathbb H^d
=
\{v \in \mathbb R^{d+1} : \langle x,v\rangle_L = 0\},
\qquad
\Pi_x = I_{d+1}+x x^\top J.
$$
The hyperbolic distance is
$$
d_{\mathbb H}(x,y)=\operatorname{arcosh}\!\bigl(-\langle x,y\rangle_L\bigr).
$$
If $f(x_t,t)\in T_{x_t}\mathbb H^d$, then the Ito SDE on $\mathbb H^d$ with generator
$$
L=\langle f,\nabla_{\mathbb H}\rangle+\frac{\sigma(x_t, t)^2}{2}\Delta_{\mathbb H}
$$
has the ambient-space form
$$
dx_t
=
\left(f(x_t,t)+\frac{d}{2}\sigma(x_t, t)^2 x_t\right)dt
+ \sigma(x_t, t)\Pi_{x_t}\, dW_t.
$$
This is the Ito form. The corresponding Stratonovich form is geometrically cleaner, and converting to Ito produces the curvature drift $+\frac d2 \sigma(t)^2 x_t$.

> Proof: Derive the ambient Ito form on the hyperboloid

#### Brownian Bridge on the Hyperboloid / Lorentz Model with Local Chart Projection
Refer to [derivation](https://claude.ai/share/53c00a93-a442-42a5-bf08-c1a8fe927191). Let $y \in S^{d-1}$ and represent the corresponding ideal boundary point by the null vector
$$
\xi(y)=(1,y),
\qquad
\|y\|=1,
\qquad
\langle \xi(y),\xi(y)\rangle_L = 0.
$$
So boundary points are represented by null vectors on the light cone, not by points on $\mathbb H^d$ itself. The Poisson kernel, equivalently the harmonic measure density with respect to the uniform surface measure on $S^{d-1}$, is
$$
q(y \mid x)
=
\frac{1}{A_{d-1}}
\left(-\langle x,\xi(y)\rangle_L\right)^{-(d-1)},
\qquad
A_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}.
$$
The Doob $h$-transform of hyperbolic Brownian motion conditioned to converge to the ideal boundary point $y$ has SDE
$$
dx_t
=
\left(
f(x_t,t)
- \frac{(d-1) \sigma(x_t, t)^2}{\langle x_t,\xi(y)\rangle_L}\xi(y)
- \frac{d-2}{2}\sigma(x_t, t)^2 x_t
\right)dt
+ \sigma(x_t, t)\Pi_{x_t}\, dW_t.
$$
Equivalently, if $h_{\xi}(x)\propto \left(-\langle x,\xi\rangle_L\right)^{-(d-1)}$, then the added drift is $\sigma(x_t, t)^2 \nabla_{\mathbb H}\log h_\xi(x)$.

> Proof: Derive the boundary-conditioned drift from the Poisson kernel via a Doob $h$-transform
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
Consider a model $f_{\theta}:  T_{x_t} \mathbb{D}^d \times \mathbb{R} \to \mathbb{R}^d$ would take the state $x_t$ of the Brownian bridge $p_{\theta}(x_{0:T} | y)$ and predict the final target $y$, the bridge learned by the model $f_{\theta}$ is
$$
\begin{aligned}
dx_t & = \left( \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|f_{\theta}(x_t, t)-x_t\|^2} (f_{\theta}(x_t, t)-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t \right) dt + \frac{1-\|x_t\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$
Both bridges can be expressed as $dx_t = f(t, x_t, y) dt + g(t, x_t) d \bar{W}_t$ with $f: \mathbb{R} \times T_{x_t} \mathbb{D}^{d} \times \mathbb{R}^{d} \to T_{x_t} \mathbb{D}^d$ and $g: \mathbb{R} \times T_{x_t} \mathbb{D}^{d} \to \mathbb{R}$. Let $\{\tau(i)\}_{i=0}^{T}$ be a strictly increasing discretization of reverse time with
$$
0 = \tau(0) < \tau(1) < \cdots < \tau(T) < \infty, \qquad \tau(i) - \tau(i-1) = \Delta t \text{ for all } i.
$$
Then the **local Euler-Gaussian approximation** of the one-step posterior is
$$
\begin{aligned}
q(x_{\tau(i-1)} \mid x_{\tau(i)}, y)
& \approx \mathcal{N}\!\big(x_{\tau(i)} + f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 \Delta t I\big), \\
p_\theta(x_{\tau(i-1)} \mid x_{\tau(i)})
& \approx \mathcal{N}\!\big(x_{\tau(i)} + f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 \Delta t I\big).
\end{aligned}
$$
where $f(t, x_t, y) = \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t$ and $g(t, x_t) = \frac{1-\|x_t\|^2}{2}$. Under this reverse-time bridge convention, the one-step map from the noisier state $x_{\tau(i)}$ to the cleaner state $x_{\tau(i-1)}$ uses the $+ f(\tau(i), x_{\tau(i)}, \cdot)\Delta t$ mean shown above, and the reconstruction residual is therefore $y - x_{\tau(1)} - f(\tau(1), x_{\tau(1)}, \cdot)\Delta t$.
### Lorentz Model Brownian Bridge Diffusion
Given 2 Brownian bridge $q(x_{0:T} | y), \{x_t\} \subset T_{x_t} \mathbb{L}^d$ and $p_{\theta}(x_{0:T} | y), \{x_t\} \subset T_{x_t} \mathbb{L}^d$ on the ambient space $\mathbb{R}^{d+1}$, and conditioned on a target point $y \in \mathbb{R}^{d+1}$ drawn from training dataset $y \in Y$, assume 
- (1) in the continuous bridge limit, both bridges start from the origin at time $\infty$
- (2) the discrete schedule has fixed step size $\tau(i) - \tau(i-1) = \Delta t \ll 1$ for all $i$
and target points $|| y || = 1$ are at the boundary of  $\mathbb{R}^{d+1} = \{ x \in \mathbb{R}^{d+1} : \|x\| = 1 \}$. 
$$
dx_t
=
\left(
- \frac{(d-1)}{\langle x_t,\xi(y)\rangle_L}\xi(y)
- \frac{d-2}{2} x_t
\right)dt
+ \Pi_{x_t}\, dW_t.
$$
Consider a model $f_{\theta}:  T_{x_t} \mathbb{L}^d \times \mathbb{R} \to \mathbb{R}^d$ would take the state $x_t$ of the Brownian bridge $p_{\theta}(x_{0:T} | y)$ and predict the final target $y$, the bridge learned by the model $f_{\theta}$ is
$$
\begin{aligned}
dx_t
=
\left(
- \frac{(d-1)}{\langle x_t,\xi(f_{\theta}(x_t, t))\rangle_L}\xi(f_{\theta}(x_t, t))
- \frac{d-2}{2} x_t
\right)dt
+ \Pi_{x_t}\, dW_t.\\
\end{aligned}
$$
Both bridges can be expressed as $dx_t = f(t, x_t, y) dt + g(t, x_t) d \bar{W}_t$ with $f: \mathbb{R} \times T_{x_t} \mathbb{D}^{d} \times \mathbb{R}^{d} \to T_{x_t} \mathbb{L}^d$ and $g: \mathbb{R} \times T_{x_t} \mathbb{L}^{d} \to \mathbb{R}$. Let $\{\tau(i)\}_{i=0}^{T}$ be a strictly increasing discretization of reverse time with
$$
0 = \tau(0) < \tau(1) < \cdots < \tau(T) < \infty, \qquad \tau(i) - \tau(i-1) = \Delta t \text{ for all } i.
$$
Since $\tau(T) = T \Delta t$ under the uniform-step schedule, the continuous bridge limit is recovered by letting $T \to \infty$, $\Delta t \to 0$, and hence $\tau(T) = T \Delta t \to \infty$. Then the **local Euler-Gaussian approximation** of the one-step posterior is
### Poincare Disk Brownian Bridge Diffusion on Simplex Space
According to the Diffusion Duality paper, the discrete ELBO provided in the paper is evaluating the Argmax Gaussian diffusion on the USDM ELBO. However, it requires equivalent marginal distribution between USDM and Argmax Gaussian diffusion with linear drift $\alpha(t) x_t$. The drift of Poincare Disk Brownian Bridge Diffusion doesn't satisfy this criteria.
#### Brownian Bridge on Simplex Space with Softmax Projection
Therefore, we apply Ito's lemma to derive the dynamic of the bridge on the simplex space with ``softmax`` function. Based on Ito's lemma, given a bridge SDE $d x_t = \bar{\mu}(t, x_t, y) dt + \bar{\sigma}(t, x_t) d \bar{W}_t$ conditioned on target $y \in \mathbb{R}^d$, where $x_t \in \mathbb{R}^d$, drift $\bar{\mu}: \mathbb{R} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$, and diffusion $\bar{\sigma}: \mathbb{R} \times \mathbb{R}^d \to \mathbb{R}^{d \times d}$, the dynamic of $h_{\mathcal{T}}(x_t) := softmax(\frac{E x_t}{\mathcal{T}}), E \in \mathbb{R}^{K \times d}$ is
$$
\begin{aligned}
d h_{\mathcal{T}}(x_t) 
=  \left[ J_h(x_t) \bar{\mu}(t, x_t, y) + \frac{1}{2} b_{Ito}(t, x_t) \right] dt + J_h(x_t) \bar{\sigma}(t, x_t) d \bar{W}_t
\end{aligned}
$$
where 
- $J_h(x_t) = \frac{1}{\mathcal{T}} \left( \text{Diag}(h_{\mathcal{T}}(x_t)) - h_{\mathcal{T}}(x_t) h_{\mathcal{T}}(x_t)^{\top} \right) E$ is the Jacobian matrix of the temperature softmax.
- $\Sigma(t, x_t) = \bar{\sigma}(t, x_t) \bar{\sigma}(t, x_t)^T$ is the covariance (diffusion) tensor.
- $[b_{Ito}(t, x_t)]_k = Tr(\Sigma(t, x_t) \nabla_{x_t}^2 [h_{\mathcal{T}}(x_t)]_{k})$ is the $k$-th component of Ito correction $k = 1, \dots,  K$

> Proof: The Jacobian matrix of temperature softmax $J_h(x_t) = \frac{1}{\mathcal{T}} \left( \text{Diag}(h_{\mathcal{T}}(x_t)) - h_{\mathcal{T}}(x_t) h_{\mathcal{T}}(x_t)^{\top} \right) E$

Let
$$
u(x_t) := \frac{E x_t}{\mathcal T},
\qquad
h_{\mathcal T}(x_t) = \operatorname{softmax}(u(x_t)).
$$
For each coordinate,
$$
[h_{\mathcal T}(x_t)]_i
=
\frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}}.
$$
Differentiating with respect to $u_k$, for $k \neq i$, gives
$$
\begin{aligned}
\frac{\partial [h_{\mathcal T}(x_t)]_i}{\partial u_k}
& =
\frac{\partial}{\partial u_k} \frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}} 
 = e^{u_i} \frac{\partial}{\partial u_k} \left( \sum_{\ell=1}^K e^{u_\ell} \right)^{-1} \\
& = - e^{u_i} \left( \sum_{\ell=1}^K e^{u_\ell} \right)^{-2} \frac{\partial}{\partial u_k} \sum_{\ell=1}^K e^{u_\ell} 
= - e^{u_i} \left( \sum_{\ell=1}^K e^{u_\ell} \right)^{-2} e^{u_k} \\
& = - [h_{\mathcal T}(x_t)]_i [h_{\mathcal T}(x_t)]_k 
= - [h_{\mathcal T}(x_t)]_i ( 0 - [h_{\mathcal T}(x_t)]_k). \\
\end{aligned}
$$
Differentiating with respect to $u_i$ gives
$$
\begin{aligned}
\frac{\partial [h_{\mathcal T}(x_t)]_i}{\partial u_i}
& =
\frac{\partial}{\partial u_i} \frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}} 
=
\frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}} + e^{u_i} \frac{\partial}{\partial u_i} \left( \sum_{\ell=1}^K e^{u_\ell} \right)^{-1} \\
& =
\frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}} - e^{u_i} \left( \sum_{\ell=1}^K e^{u_\ell} \right)^{-2} \frac{\partial}{\partial u_i} \sum_{\ell=1}^K e^{u_\ell} 
=
\frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}} - e^{u_i} \left( \sum_{\ell=1}^K e^{u_\ell} \right)^{-2} e^{u_i} \\
& =
\frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}}\left(1-\frac{e^{u_i}}{\sum_{\ell=1}^K e^{u_\ell}}\right) 
= [h_{\mathcal T}(x_t)]_i \bigl(1-[h_{\mathcal T}(x_t)]_i\bigr). \\
\end{aligned}
$$
Combining the two cases,
$$
\frac{\partial [h_{\mathcal T}(x_t)]_i}{\partial u_k}
=
[h_{\mathcal T}(x_t)]_i (\delta_{ik}-[h_{\mathcal T}(x_t)]_k).
$$
where $\delta_{ik}$ is the Kronecker delta: $\delta_{ik}=1$ if $i=k$, and $\delta_{ik}=0$ otherwise. Therefore the Jacobian of softmax with respect to $u$ is
$$
\frac{\partial h_{\mathcal T}(x_t)}{\partial u}
=
\operatorname{Diag}(h_{\mathcal T}(x_t)) - h_{\mathcal T}(x_t) h_{\mathcal T}(x_t)^\top.
$$
Also,
$$
\frac{\partial u}{\partial x_t} = \frac{1}{\mathcal T} E.
$$
Applying the chain rule,
$$
J_h(x_t)
=
\frac{\partial h_{\mathcal T}(x_t)}{\partial x_t}
=
\frac{\partial h_{\mathcal T}(x_t)}{\partial u}\frac{\partial u}{\partial x_t}
=
\frac{1}{\mathcal T}\left(\operatorname{Diag}(h_{\mathcal T}(x_t)) - h_{\mathcal T}(x_t) h_{\mathcal T}(x_t)^\top\right)E.
$$
#### Poincare Disk Brownian Bridge on Simplex Space with Softmax Projection
For the Poincare disk Brownian bridge, take
$$
\bar{\mu}(t, x_t, y) = f(t, x_t, y),
\qquad
\bar{\sigma}(t, x_t) = g(t, x_t) I_d.
$$
Then the pushed process on simplex space is
$$
\begin{aligned}
d h_{\mathcal{T}}(x_t) 
= \hat f(t, x_t, y) dt + \hat g(t, x_t) d \bar{W}_t
\end{aligned}
$$
where 
- $f(t, x_t, y) = \frac{d-1}{2} \frac{(1-\|x_t\|^2)^2}{\|y-x_t\|^2} (y-x_t) - \frac{d}{4} (1-\|x_t\|^2) x_t$, which is the same as the Brownian bridge in continuous space
- $g(t, x_t) =  \frac{1-\|x_t\|^2}{2}$, which is the same as the Brownian bridge in continuous space
- $\hat f(t, x_t, y) = J_h(x_t) f(t, x_t, y) + \frac{1}{2} b_{Ito}(t, x_t)$
- $\hat g(t, x_t) = J_h(x_t) \bar{\sigma}(t, x_t) = g(t, x_t) J_h(x_t)$
##### Euler-Gaussian Approximation
Additionally, to get the bridge learned by the model $f_{\theta}$ on simplex space, normalize the model output to a boundary point and plug in $y=\hat y_\theta(x_t,t)$, yielding
$$
\hat y_\theta(x_t,t) := \frac{f_\theta(x_t,t)}{\|f_\theta(x_t,t)\|}.
$$
Then
$$
\begin{aligned}
d h_{\mathcal{T}}(x_t) 
= \hat f(t, x_t, \hat y_\theta(x_t,t)) dt + \hat g(t, x_t) d \bar{W}_t
\end{aligned}
$$
 Then the **local Euler-Gaussian approximation** of the one-step posterior on simplex space is
$$
\begin{aligned}
q(h_{\mathcal{T}}(x_{\tau(i-1)}) \mid x_{\tau(i)}, y)
& \approx \mathcal{N} \Big(h_{\mathcal{T}}(x_{\tau(i)}) + \hat f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 \Delta t\Big), \\
p_\theta(h_{\mathcal{T}}(x_{\tau(i-1)}) \mid x_{\tau(i)})
& \approx \mathcal{N} \Big(h_{\mathcal{T}}(x_{\tau(i)}) + \hat f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 \Delta t\Big).
\end{aligned}
$$
This covariance is generally singular and supported on the simplex tangent space.
##### Softmax Pushforward Approximation 
If one wants a softmax-pushforward approximation on the simplex, it is cleaner to work in logit space
$$
z_t := E x_t.
$$
Then the **local logistic-normal approximation** is
$$
\begin{aligned}
z_{\tau(i-1)} \mid x_{\tau(i)}, y
& \approx \mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big), \\
z_{\tau(i-1)} \mid x_{\tau(i)}
& \approx \mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big).
\end{aligned}
$$
and
$$
\begin{aligned}
w_{\tau(i-1)} \mid x_{\tau(i)}, y
& \approx \mathcal{S}_{\mathcal{T} \#}\mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big), \\
w_{\tau(i-1)} \mid x_{\tau(i)}
& \approx \mathcal{S}_{\mathcal{T} \#}\mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big).
\end{aligned}
$$
Here $\mathcal{S}_{\mathcal{T} \#}$ is the temperature-softmax pushforward: if $z \sim p$, then $\mathcal{S}_{\mathcal{T} \#} p$ is the distribution of $\operatorname{softmax}(\frac{z}{\mathcal{T}})$. Equivalently,
$$
(\mathcal{S}_{\mathcal{T} \#} p)(A)
:=
p\!\left(\left\{z : \operatorname{softmax}\!\left(\frac{z}{\mathcal{T}}\right) \in A\right\}\right).
$$
##### Natural Gradient Approximation
If one wants a **local natural-gradient approximation** directly on the simplex, let
$$
G(w_t) := \operatorname{Diag}(w_t) - w_t w_t^\top
$$
be the Fisher metric tensor for softmax projection and let $G(w_t)^+$ denote its Moore-Penrose pseudoinverse. With $w_{\tau(i)} = h_{\mathcal T}(x_{\tau(i)})$, define the first-order log-coordinate increment
$$
\delta_i := G(w_{\tau(i)})^+ \hat f(\tau(i), x_{\tau(i)}, y)\Delta t.
$$
Then
$$
w_{\tau(i-1)} \mid x_{\tau(i)}, y
\approx
\operatorname{softmax}\!\big(\log w_{\tau(i)} + \delta_i\big),
$$
and similarly for $p_\theta$ by replacing $y$ with $\hat y_\theta(x_{\tau(i)},\tau(i))$. This updates the state in local log-coordinates, which consider the geometric of simplex. Although $\log w_{\tau(i)}$ is not centered logit coordinate, $\delta_i$ is the minimum-norm solution of the approximation on $f\,\Delta t$, it still stays on the centered logit coordinate. However, such update ignore the second-order diffusion information $\hat{g}$, making it just an first-order approximation.

> Proof: Fisher Information Matrix $G(w_t)$ for Categorical Distribution

For a categorical variable $i \sim \mathrm{Cat}(w)$ with logit parameter $\eta \in \mathbb{R}^{K}$, let $e_i \in \mathbb R^K$ denote the $i$-th standard basis vector, i.e.
$$
e_i = (0,\dots,0,\underbrace{1}_{i\text{-th entry}},0,\dots,0)^\top,
\qquad
[e_i]_j = \delta_{ij}.
$$
Then
$$
\eta_i = e_i^\top \eta,
\qquad
\nabla_\eta \eta_i = e_i,
$$
because
$$
\left[\nabla_\eta \eta_i\right]_j = \frac{\partial \eta_i}{\partial \eta_j} = \delta_{ij}.
$$
Let $w=\operatorname{softmax}(\eta), w \in \Delta^{K-1}$. Explicitly,
$$
p(i \mid \eta) = \frac{e^{\eta_i}}{\sum_{j=1}^K e^{\eta_j}},
\qquad
\log p(i \mid \eta) = \eta_i - \log \sum_{j=1}^K e^{\eta_j}.
$$
where $\eta_i \in \mathbb{R}$ is the $i$-th entry of $\eta$.
Therefore,
$$
\begin{aligned}
\nabla_\eta \log p(i \mid \eta) 
& = \nabla_{\eta} \eta_i - \nabla_{\eta} \log \sum_{j=1}^K e^{\eta_j} \\
& = \nabla_{\eta} \eta_i -  \frac{1}{\sum_{j=1}^K e^{\eta_j}} \nabla_{\eta} \sum_{j=1}^K e^{\eta_j} \\
& = e_i -  \frac{1}{\sum_{j=1}^K e^{\eta_j}} \sum_{j=1}^K e^{\eta_j} \nabla_{\eta} \eta_{j} \\
& = e_i -  \frac{1}{\sum_{j=1}^K e^{\eta_j}} \sum_{j=1}^K e^{\eta_j} e_j \\
& = e_i - w.
\end{aligned}
$$
Therefore the Fisher information is
$$
\begin{aligned}
F(\eta)
&= \mathbb{E}_{i \sim \mathrm{Cat}(\operatorname{softmax}(\eta))} \left[\nabla_{\eta} \log p(i \mid \eta) \nabla_{\eta} \log p(i \mid \eta)^{\top} \right] \\
&= \mathbb{E}_{i \sim \mathrm{Cat}(\operatorname{softmax}(\eta))} \left[(e_i-w)(e_i-w)^\top\right] \\
&= \sum_{i=1}^K w_i (e_i-w)(e_i-w)^\top \\
&= \sum_{i=1}^K w_i \left( e_i e_i^{\top} - e_i w^{\top} - w e_{i}^\top + w w^\top \right) \\
&= \underbrace{\sum_{i=1}^K w_i e_i e_i^\top}_{=\operatorname{Diag}(w)}
- \underbrace{\sum_{i=1}^K w_i e_i}_{=\,w} w^\top
- w \underbrace{\sum_{i=1}^K w_i e_i^\top}_{=\,w^\top}
+ w w^\top \underbrace{\sum_{i=1}^K w_i}_{=\,1} \\
& = \operatorname{Diag}(w) - w w^{\top} - w w^{\top} + ww^{\top} \\
&= \operatorname{Diag}(w) - w w^\top.
\end{aligned}
$$
If $\eta = z/\mathcal T$, then the Fisher with respect to $z$ is $\mathcal T^{-2}(\operatorname{Diag}(w)-ww^\top)$. Up to this scalar factor, the simplex Fisher metric is
$$
G(w)=\operatorname{Diag}(w)-ww^\top.
$$
It is singular because $G(w)\mathbf 1 = 0$, which matches the fact that the simplex has dimension $K-1$. Additionally, **the FIM of categorical distribution is identical to the Jacobian of softmax**

> Proof: The natural gradient approximation on simplex

Set $w=w_{\tau(i)}$ and $w^+=w_{\tau(i-1)}$, and let $\phi(z)=\operatorname{softmax}(z)$. Since $\phi(\log w)=w$, a small perturbation $\delta$ in log-coordinates gives
$$
\phi(\log w+\delta)
=
\phi(\log w) + J_\phi(\log w)\delta + O(\|\delta\|^2).
$$
Also, for any $z$,
$$
J_\phi(z)=\operatorname{Diag}(\phi(z))-\phi(z)\phi(z)^\top.
$$
Evaluating at $z=\log w$ gives $\phi(\log w)=w$, so
$$
J_\phi(\log w)=\operatorname{Diag}(w)-ww^\top = G(w),
$$
so
$$
w^+ - w \approx G(w)\delta.
$$
Now suppose the desired first-order tangent displacement is $\hat f\,\Delta t$. We therefore choose $\delta$ so that
$$
G(w)\delta \approx \hat f\,\Delta t.
$$
Also,
$$
G(w)\mathbf 1
=
(\operatorname{Diag}(w)-ww^\top)\mathbf 1
=
w-w(\mathbf 1^\top w)
=
w-w
=
0,
$$
since $\mathbf 1^\top w=1$. So the all-ones direction lies in the nullspace of $G(w)$. Intuitively, adding the same constant to every logit does not change the softmax, so $\delta$ and $\delta+c\mathbf 1$ produce the same first-order change in $w$. Therefore the solution is not unique, and the Moore-Penrose pseudoinverse picks the unique minimum-norm one, meaning the update with no unnecessary common shift:
$$
\delta \approx G(w)^+\hat f\,\Delta t.
$$
Substituting this into the first-order expansion gives
$$
w^+ \approx \operatorname{softmax}\!\big(\log w + G(w)^+\hat f\,\Delta t\big),
$$
which is the stated natural-gradient approximation. Intuitively, $G(w)^+$ converts a tangent drift on the simplex into the corresponding first-order drift in local log-coordinates.

> Proof: The log space of logistic normal distribution is normal distribution

More precisely, the Gaussian object is the **log-ratio** coordinate, not the raw vector $\log w$. If
$$
z \sim \mathcal N(\mu,\Sigma),
\qquad
w = \operatorname{softmax}(z/\mathcal T), \qquad \ w \in \Delta^{d-1}, z\in \mathbb{R}^d
$$
then
$$
\log \frac{w_i}{w_K}
=
\log \frac{\frac{e^{z_i / \mathcal{T}}}{\sum_{j=1}^{K} e^{z_j / \mathcal{T}}}}{\frac{e^{z_K / \mathcal{T}}}{\sum_{j=1}^{K} e^{z_j / \mathcal{T}}}}
=
\log \frac{e^{z_i / \mathcal{T}}}{e^{z_K / \mathcal{T}}}
=
\frac{z_i-z_K}{\mathcal T},
\qquad i=1,\dots,K-1.
$$
The right-hand side $\frac{z_i-z_K}{\mathcal T}$ is an affine transform of the Gaussian vector $z$, so it is Gaussian. Equivalently, the centered log-ratio coordinate satisfies
$$
\begin{aligned}
\operatorname{clr}(w)
& :=
\log w - \frac{1}{K}\mathbf 1 \mathbf 1^\top \log w
=
(I - \frac{1}{K}\mathbf 1 \mathbf 1^\top) \log w \\
& =
\left(I-\frac{1}{K}\mathbf 1\mathbf 1^\top\right)
\left(
\frac{z}{\mathcal T}
- \left(\log \sum_{j=1}^K e^{z_j/\mathcal T}\right)\mathbf 1
\right) \\
& =
\frac{1}{\mathcal{T}} (I - \frac{1}{K}\mathbf 1 \mathbf 1^\top) z
- \underbrace{
\left(\log \sum_{j=1}^{K} e^{z_j / \mathcal{T}}\right)
\left(I-\frac{1}{K}\mathbf 1\mathbf 1^\top\right)\mathbf 1
}_{=\,0\text{ since }(I-\frac{1}{K}\mathbf 1\mathbf 1^\top)\mathbf 1=0}
=
\frac{1}{\mathcal T}\left(I-\frac{1}{K}\mathbf 1\mathbf 1^\top\right) z,
\end{aligned}
$$
which is again Gaussian. This is why the logistic-normal distribution is naturally Gaussian in logit / log-ratio space.

> Lemma - Natural-gradient CTMC realization

Define the potential $\psi_t$
$$
\psi_t := G(w_t)^+ \hat f(t,x_t,y).
$$
Define the generator $Q_t \in \mathbb R^{K\times K}$ of the CTMC $\frac{d w_t}{d t} = Q_t^\top w_t$ by
$$
[Q_t]_{ij}
:=
[w_t]_j\big([\psi_t]_j-[\psi_t]_i\big)_+,
\qquad i \neq j,
\qquad
(a)_+ := \max(a,0),
$$
and
$$
[Q_t]_{ii} := -\sum_{j\neq i} [Q_t]_{ij}.
$$
The matrix $Q_t \in \mathbb{R}^{K\times K}$ is a valid CTMC generator and satisfies
$$
Q_t^\top w_t = \hat f(t,x_t,y).
$$
Assume
$$
w_t = h_{\mathcal T}(x_t) \in \operatorname{int}\Delta^{K-1},
\qquad
\hat f(t,x_t,y) \in T_{w_t}\Delta^{K-1},
\qquad
\mathbf 1^\top \hat f(t,x_t,y) = 0.
$$
> Proof of Lemma - Natural-gradient CTMC realization: $Q_t$ is a valid CTMC generator and satisfies $Q_t^\top w_t = \hat f(t,x_t,y)$

**The Necessary Conditions of a Valid Generator of CTMC**
Given a generator $Q_t \in \mathbb R^{K\times K}$ of the CTMC $\frac{d w}{d t} = Q_t^\top w$, the generator is valid if and only if
- The un-change portion at $i$ is equal to the negative out-flow from $i$
$$
[Q_t]_{ii} = - \sum_{i \neq j} [Q_t]_{ij}
$$
**$Q_t$ is a Valid Generator for CTMC**
For $i\neq j$, $[Q_t]_{ij}\ge 0$. Also,
$$
[Q_t]_{ii} = -\sum_{j\neq i}[Q_t]_{ij},
$$
so each row of $Q_t$ sums to zero. Hence $Q_t$ is a valid CTMC generator.

**The Dual Potential $\psi_t = G(w_t)^+ \hat f(t,x_t,y)$**
Considering the exact first-order infinitestimal of $d h_{\mathcal{T}}(x_t)$ is $\hat{f}(t, x_t, y)$, while the increment $\Delta t$ is sufficiently small, then
$$
\lim_{\Delta t \to 0} \mathbb{E}_{q(h_{\mathcal{T}}(x_{\tau(i-1)}) \mid x_{\tau(i)}, y)} \left[ h_{\mathcal{T}}(x_{\tau(i-1)}) - h_{\mathcal{T}}(x_{\tau(i)}) \right] 
= \hat f(\tau(i), x_{\tau(i)}, y) \Delta t.
$$
We aim to find a natural gradient update direction $\psi_t \in \mathbb{R}^d$ that matches exact first-order infinitestimal.
$$
G(w_t) \psi_t = \hat{f}(t, x_t, y)
$$
Therefore, a straight forward choice for $\psi_t$ is
$$
\psi_t := G(w_t)^+ \hat f(t,x_t,y).
$$
where $G(w_t)^+$ is the Moore-Penrose pseudoinverse of $G(w)$.
**Natural Gradient Update is Already the First Order Approximation on Simplex**
Let $w = \operatorname{softmax}(x)$, the simplex after small perturbation $\varepsilon \psi, \varepsilon \in \mathbb{R}, \psi \in \mathbb{R}^{d}$ is
$$
\begin{aligned}
w^+ = \operatorname{softmax}(x + \varepsilon \psi)
\end{aligned}
$$
Let $J_{\mathrm{softmax}}(z)$ be the Jacobian matrix of Softmax function, expand $w^+$ by Taylor expansion
$$
w^+ = w + \varepsilon J_{\mathrm{softmax}}(z)\psi + O(\varepsilon^2).
$$
Since the Jacobian of softmax is the same as the Fisher metric tensor for softmax projection.
$$
J_{\mathrm{softmax}}(z) = \text{Diag}(w) - w w^{\top} = G(w)
$$
Therefore, the first-order Taylor approximation is the same as the natural gradient update
$$
w^+ - w \approx \varepsilon J_{\mathrm{softmax}}(w) \psi = \varepsilon G(w) \psi.
$$
**The Dynamic of CTMC $Q_t^\top w_t$ is equal to Natural Gradient Update $G(w_t) \psi_t$** 
To finish the proof, compute the drift induced by $Q_t$ coordinatewise. For each $i$,
$$
\begin{aligned}
\left[(Q_t^\top w_t)\right]_i
&=
\sum_{j\neq i} [Q_t]_{ji}[w_t]_j - \sum_{j\neq i}[Q_t]_{ij}[w_t]_i \\
&=
\sum_{j\neq i}[w_t]_i[w_t]_j
\Big(
([\psi_t]_i-[\psi_t]_j)_+
-([\psi_t]_j-[\psi_t]_i)_+
\Big) \\
&=
\sum_{j\neq i}[w_t]_i[w_t]_j\big([\psi_t]_i-[\psi_t]_j\big) \\
&=
[w_t]_i[\psi_t]_i - [w_t]_i\sum_j [w_t]_j[\psi_t]_j.
\end{aligned}
$$
Here we used $(a)_+ - (-a)_+ = a$. Also,
$$
[\operatorname{Diag}(w_t)\psi_t]_i = [w_t]_i[\psi_t]_i,
\qquad
[w_t(w_t^\top\psi_t)]_i = [w_t]_i\sum_j [w_t]_j[\psi_t]_j.
$$
So the previous coordinate identity is exactly the $i$-th component of
$$
\begin{aligned}
Q_t^\top w_t
= \operatorname{Diag}(w_t)\psi_t - w_t(w_t^\top\psi_t)
= \operatorname{Diag}(w_t)\psi_t - (w_t w_t^\top) \psi_t
= ( \operatorname{Diag}(w_t) - w_t w_t^{\top} ) \psi_t
= G(w_t) \psi_t
\end{aligned}
$$
**The CTMC Dynamic is equal to the Exact First-Order Infinitestimal**
Now substitute the definition $\psi_t := G(w_t)^+\hat f(t,x_t,y)$. Since $w_t \in \operatorname{int}\Delta^{K-1}$,
$$
\operatorname{Range}(G(w_t)) = T_{w_t}\Delta^{K-1}
=
\{v\in\mathbb R^K : \mathbf 1^\top v = 0\}.
$$
Because $\hat f(t,x_t,y) \in T_{w_t}\Delta^{K-1}$, it lies in the range of $G(w_t)$, so
$$
G(w_t)G(w_t)^+\hat f(t,x_t,y) = \hat f(t,x_t,y).
$$
Hence
$$
Q_t^\top w_t
=
G(w_t)\psi_t
=
G(w_t)G(w_t)^+\hat f(t,x_t,y)
=
\hat f(t,x_t,y).
$$
This proves the lemma.

**Finite-step consequence**
For a reverse-time step $\Delta t$, define
$$
Q_i := Q\!\big(w_{\tau(i)},\hat f(\tau(i),x_{\tau(i)},y)\big).
$$
Considering $\frac{d w_t}{d t} = Q_t^\top w_t$, the exact Markov update is
$$
w_{\tau(i-1)}
=
\exp\!\big(\Delta t\,Q_i^\top\big)\,w_{\tau(i)}.
$$
Since $\exp(\Delta t\,Q_i^\top)$ is column-stochastic for every $\Delta t \ge 0$, this update stays on the simplex exactly:
$$
w_{\tau(i-1)} \ge 0,
\qquad
\mathbf 1^\top w_{\tau(i-1)} = 1.
$$
Its first-order expansion is
$$
w_{\tau(i-1)}
=
w_{\tau(i)} + \Delta t\,Q_i^\top w_{\tau(i)} + O(\Delta t^2)
=
w_{\tau(i)} + \Delta t\,\hat f(\tau(i),x_{\tau(i)},y) + O(\Delta t^2).
$$
Hence the CTMC step is exact on the simplex, and only first-order equivalent to the earlier softmax retraction
$$
\operatorname{softmax}\!\big(
\log w_{\tau(i)} + \Delta t\,G(w_{\tau(i)})^+\hat f(\tau(i),x_{\tau(i)},y)
\big),
$$
which is an approximation in $\Delta t$. The same construction applies to $p_\theta$ by replacing $y$ with $\hat y_\theta(x_t,t)$. Since $\hat f$ depends on $x_t$, this is generally a conditional Markov transport given $x_t$, not an autonomous CTMC in $w_t$ alone.
#### Categorical Process Yielded by Softmax Projection
##### Softmax Pushforward Approximation
If one wants a softmax-pushforward approximation on the simplex, it is cleaner to work in logit space
$$
z_t := E x_t.
$$
Then the **local logistic-normal approximation** is
$$
\begin{aligned}
z_{\tau(i-1)} \mid x_{\tau(i)}, y
& \approx \mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big), \\
z_{\tau(i-1)} \mid x_{\tau(i)}
& \approx \mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big).
\end{aligned}
$$
and
$$
\begin{aligned}
w_{\tau(i-1)}^{q} \mid x_{\tau(i)}, y
& \approx \mathcal{S}_{\mathcal{T} \#}\mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big), \\
w_{\tau(i-1)}^{\theta} \mid x_{\tau(i)}
& \approx \mathcal{S}_{\mathcal{T} \#}\mathcal{N} \Big(z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big).
\end{aligned}
$$
Here $\mathcal{S}_{\mathcal{T} \#}$ is the temperature-softmax pushforward: if $z \sim p$, then $\mathcal{S}_{\mathcal{T} \#} p$ is the distribution of $\operatorname{softmax}(\frac{z}{\mathcal{T}})$. Equivalently,
$$
(\mathcal{S}_{\mathcal{T} \#} p)(A)
:=
p\!\left(\left\{z : \operatorname{softmax}\!\left(\frac{z}{\mathcal{T}}\right) \in A\right\}\right).
$$
Corresponding categorical process $k_{\tau(i-1)}$ ,
$$
\begin{aligned}
\mathcal{k}_{\tau(i-1)} | x_{\tau(i)}, y 
& \approx \text{Cat}(w_{\tau(i-1)}^{q}), w_{\tau(i-1)}^{q} \sim \mathcal{S}_{\mathcal{T} \#}\mathcal{N} \Big( z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, y) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big), \\
k_{\tau(i-1)} \mid x_{\tau(i)}
& \approx \text{Cat}(w_{\tau(i-1)}^{\theta}), \ w_{\tau(i-1)}^{\theta} \sim \mathcal{S}_{\mathcal{T} \#}\mathcal{N} \Big( z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i))) \Delta t,\; g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t\Big).
\end{aligned}
$$
The KL Divergence
$$
\begin{aligned}
D_{KL}(q(\mathcal{k}_{\tau(i-1)} \mid x_{\tau(i)}, y) \,\|\, p_{\theta}(k_{\tau(i-1)} \mid x_{\tau(i)}))
& = \sum_{j=1}^{K} q(\mathcal{k}_{\tau(i-1), j} \mid x_{\tau(i)}, y) \log \frac{q(\mathcal{k}_{\tau(i-1), j} \mid x_{\tau(i)}, y)}{p_{\theta}(k_{\tau(i-1), j} \mid x_{\tau(i)})} \\
& = \sum_{j=1}^{K} w_{\tau(i-1), j}^{q} \log \frac{w_{\tau(i-1), j}^{q}}{w_{\tau(i-1), j}^{\theta}} \\
& = \sum_{j=1}^{K} w_{\tau(i-1), j}^{q} ( \log w_{\tau(i-1), j}^{q} - \log w_{\tau(i-1), j}^{\theta}) \\
& = \sum_{j=1}^{K} w_{\tau(i-1), j}^{q} (( z_{\tau(i-1), j}^{q} - \log \sum_{l=1}^{K} z_{\tau(i-1), l}^{q}) - ( z_{\tau(i-1), j}^{\theta} - \log \sum_{l=1}^{K} z_{\tau(i-1), l}^{\theta})) \\
\end{aligned}
$$
is the KL divergence between the corresponding marginal categorical distributions. Let
$$
\mu_q := z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, y)\Delta t,
\qquad
\mu_\theta := z_{\tau(i)} + E f(\tau(i), x_{\tau(i)}, \hat y_\theta(x_{\tau(i)},\tau(i)))\Delta t,
$$
and
$$
\Sigma_i := g(\tau(i), x_{\tau(i)})^2 E E^\top \Delta t.
$$
Define
$$
\bar w^{(q)}_{\tau(i-1)}
:=
\mathbb E_{w \sim \mathcal S_{\mathcal T \#}\mathcal N(\mu_q,\Sigma_i)}[w],
\qquad
\bar w^{(\theta)}_{\tau(i-1)}
:=
\mathbb E_{w \sim \mathcal S_{\mathcal T \#}\mathcal N(\mu_\theta,\Sigma_i)}[w].
$$
Then, for each class $m$,
$$
\begin{aligned}
q(\mathcal k_{\tau(i-1)} = m \mid x_{\tau(i)}, y)
&=
\int q(\mathcal k_{\tau(i-1)} = m \mid w)\, q(w \mid x_{\tau(i)}, y)\, dw \\
&=
\int [w]_m\, q(w \mid x_{\tau(i)}, y)\, dw
=
[\bar w^{(q)}_{\tau(i-1)}]_m,
\end{aligned}
$$
and similarly
$$
p_\theta(k_{\tau(i-1)} = m \mid x_{\tau(i)})
=
[\bar w^{(\theta)}_{\tau(i-1)}]_m.
$$
Therefore
$$
\boxed{
D_{KL}(q(\mathcal{k}_{\tau(i-1)} \mid x_{\tau(i)}, y) \,\|\, p_{\theta}(k_{\tau(i-1)} \mid x_{\tau(i)}))
=
\sum_{m=1}^{K}
[\bar w^{(q)}_{\tau(i-1)}]_m
\log
\frac{[\bar w^{(q)}_{\tau(i-1)}]_m}{[\bar w^{(\theta)}_{\tau(i-1)}]_m}
}.
$$
In general, $\bar w^{(q)}_{\tau(i-1)}$ and $\bar w^{(\theta)}_{\tau(i-1)}$ are expectations of logistic-normal random vectors, so they typically do not admit a simple closed form.
### NELBO in Continuous Space
For a probability model $p_{\theta}$ and a Brownian motion $x_{1:T}$, the NELBO of the Brownian bridge diffusion conditioned on $x_0$ is following.
$$
\begin{aligned}
- \log p_{\theta}(x_0) 
& \leq \mathbb{E}_{q(x_{0:T})} \Big[ \log\frac{q(x_{1:T} | x_0)}{p_\theta(x_{0:T})} \Big] \\
& = \mathbb{E}_q [\underbrace{D_\text{KL}(q(x_T \vert x_0) \parallel p_\theta(x_T))}_{\text{Prior}} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(x_{t-1} \vert x_t, x_0) \parallel p_\theta(x_{t-1} \vert x_t))}_{\text{Diffusion}} \underbrace{- \log p_\theta(x_0 \vert x_1)}_{\text{Reconst}} ]
\end{aligned}
$$
Note that the KL divergence between isotropic Gaussian distribution $p=\mathcal N(\mu_0,\sigma_0^2 I)$ and $q=\mathcal N(\mu_1,\sigma_1^2 I)$ is
$$
\boxed{ D_{\mathrm{KL}}(p\|q) = \frac12 \left[ d\log\frac{\sigma_1^2}{\sigma_0^2} -d +d\frac{\sigma_0^2}{\sigma_1^2} +\frac{\|\mu_1-\mu_0\|^2}{\sigma_1^2} \right]. }
$$

### NELBO in Simplex / Dicrete Space
To convert the continuous state $x_t$ to discrete state $w_t$, we project the $d$-dimensional continuous state $x_t$ to $K$-dimensional simplex or discrete state by applying a projection matrix $E \in \mathbb{R}^{K \times d}$ and a softmax function. Considering projection to simplex / discrete states, we use 
- Simplex: Deterministic softmax $w_t = softmax ( \frac{E x_t}{\mathcal{T}})$, 
- Discrete States: Categorical distribution $\text{Cat}(w_t; softmax ( \frac{E x_t}{\mathcal{T}}))$. 

Additionally, $x_{0:T}$ is a Markov chain and $w_t$ only depends on the $x_t$. Therefore, we choose $q(w_{1:T}, x_{0:T} | w_0)$ as the proposal distribution because $w_0$ is observable while both $w_{1:T}$ and $x_{0:T}$ are latents. Denote the learned probability model as $p_{\theta}$ and a Brownian motion as $x_{1:T}$.

#### Decomposition $q(w_{t-1}, x_{t-1} | x_{t}, x_{0}) = q(w_{t-1} | x_{t-1}) q(x_{t-1} | x_{t}, x_{0})$
Let
$$
\begin{aligned}
-\log p_\theta(w_0)
&\le \mathcal L_{\text{NELBO}}(w_0)
:= \mathbb E_{q(w_{1:T},x_{0:T}\mid w_0)}
\left[
\log \frac{q(w_{1:T},x_{0:T}\mid w_0)}{p_\theta(w_{0:T},x_{0:T})}
\right].
\end{aligned}
$$
Assume the reverse generative model factorizes as
$$
\begin{aligned}
p_\theta(w_{0:T}, x_{0:T})
& =
p_{\theta}(x_T)\prod_{t=1}^T p_\theta(x_{t-1} | x_t)\prod_{t=0}^T p_\theta(w_t | x_t) \\
& =
p_{\theta}(x_T) p_\theta(w_0 | x_0) p_\theta(x_{0} | x_1) \prod_{t=2}^T p_\theta(x_{t-1} | x_t) \prod_{t=1}^T p_\theta(w_t | x_t) \\
\end{aligned}
$$
and the proposal factorizes as
$$
\begin{aligned}
q(w_{1:T},x_{0:T}\mid w_0)
&= q(x_{0:T} \mid w_0) q(w_{1:T}\mid x_{0:T},w_0) \\
&= q(x_0 \mid w_0) q(x_{1:T} | x_0, w_0) \prod_{t=1}^T q(w_t\mid x_{0:T}, w_0, w_{1:t-1}) \\
&= q(x_0 \mid w_0) \underbrace{q(x_T\mid x_0) \prod_{t=2}^T q(x_{t-1}\mid x_t,x_0)}_{q(x_{1:T} | x_0, w_0) = q(x_T\mid x_0) \prod_{t=2}^T q(x_{t-1}\mid x_t,x_0)} \prod_{t=1}^T q( w_t\mid x_t).
\end{aligned}
$$
Note that the detailed derivation of factorized proposal $q(x_{1:T} | x_0, w_0)$
$$
\begin{aligned}
q(x_{1:T} | x_0, w_0)
& = q(x_{1:T} | x_0) \\
& = \prod_{t=1}^T q(x_{t} \mid x_{t-1}) \\
& = q(x_{1} \mid x_{0}) \prod_{t=2}^T q(x_{t} \mid x_{t-1}) \\
& = q(x_{1} \mid x_{0}) \left( \prod_{t=2}^T q(x_{t-1} \mid x_{t}, x_0) \frac{q(x_t | x_0)}{q(x_{t-1} | x_0)} \right), \ \because \text{Markov property } q(x_{t} \mid x_{t-1}) = q(x_{t} \mid x_{t-1}, x_0) \\
& = q(x_{1} \mid x_{0}) \prod_{t=2}^T q(x_{t-1} \mid x_{t}, x_0) \prod_{t=2}^T \frac{q(x_t | x_0)}{q(x_{t-1} | x_0)} \\
& = q(x_{1} \mid x_{0}) \frac{q(x_T | x_0)}{q(x_{1} | x_0)} \prod_{t=2}^T q(x_{t-1} \mid x_{t}, x_0)  \\
& = q(x_T\mid x_0) \prod_{t=2}^T q(x_{t-1}\mid x_t,x_0) \\
\end{aligned}
$$

Therefore, with factorization,
$$
\begin{aligned}
p_\theta(w_{0:T}, x_{0:T})
& = p_{\theta}(x_T) p_\theta(w_0 | x_0) p_\theta(x_{0} | x_1) \prod_{t=2}^T p_\theta(x_{t-1} | x_t) \prod_{t=1}^T p_\theta(w_t | x_t) \\
q(w_{1:T},x_{0:T}\mid w_0) 
& = q(x_0 \mid w_0) q(x_T | x_0) \prod_{t=2}^T q(x_{t-1}\mid x_t,x_0) \prod_{t=1}^T q( w_t\mid x_t),
\end{aligned}
$$
the discrete NELBO on the state $w_{t}$ of the Brownian bridge conditioned on $w_0$ is 
$$
\begin{aligned}
\mathcal L_{\text{NELBO}}(w_0)
&=
\mathbb E_q \Bigg[
\log q(x_0\mid w_0)-\log p_\theta(w_0\mid x_0)
+
\sum_{t=1}^T \log \frac{q(w_t\mid x_t)}{p_\theta(w_t\mid x_t)}
\\
&\qquad\qquad\qquad
+
\log \frac{q(x_T\mid x_0)}{p_{\theta}(x_T)}
+
\sum_{t=2}^T \log \frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}
-\log p_\theta(x_0\mid x_1)
\Bigg]
\\
&=
\underbrace{
\mathbb E_{q(x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(x_T\mid x_0)\,\|\,p_{\theta}(x_T)\bigr)
}_{L_{\text{Prior}}}
+
\underbrace{
\sum_{t=2}^T
\mathbb E_{q(x_t,x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)
}_{L_{\text{Diffusion}}}
\\
&\quad+
\underbrace{
\mathbb E_{q(x_0,x_1\mid w_0)}
\bigl[-\log p_\theta(x_0\mid x_1)\bigr]
}_{L_{\text{Reconstruction}}}
+
\underbrace{
\sum_{t=1}^T
\mathbb E_{q(x_t\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(w_t\mid x_t)\,\|\,p_\theta(w_t\mid x_t)\bigr)
}_{L_{\text{Projection}}}
\\
&\quad+
\underbrace{
\mathbb E_{q(x_0\mid w_0)}
\bigl[\log q(x_0\mid w_0)-\log p_\theta(w_0\mid x_0)\bigr]
}_{L_{\text{Init}}}..
\end{aligned}
$$

#### Decomposition $q(w_{t-1}, x_{t-1} | x_{t}, x_{0}) = q(w_{t-1} | x_{t}, x_{0}) q(x_{t-1} | w_{t-1}, x_{t}, x_{0})$
However, for bot discrete and simplex cases, $D_{KL}( q(w_{t} | x_{t}) || p_{\theta}(w_{t} | x_t)) = 0$, which is correct but meaningless. Therefore, to obtain a meaningful discrete ELBO on $q(w_{t-1} | x_{t}, x_{0})$, assume the proposal factorizes as
$$
\begin{aligned}
q(w_{1:T},x_{0:T}\mid w_0)
&= q(x_0 \mid w_0) \underbrace{q(x_T\mid x_0) \prod_{t=2}^T q(x_{t-1}\mid x_t,x_0)}_{q(x_{1:T} | x_0, w_0) = q(x_T\mid x_0) \prod_{t=2}^T q(x_{t-1}\mid x_t,x_0)} \prod_{t=1}^T q( w_t\mid x_t) \\
&= q(x_0 \mid w_0) q(x_T\mid x_0) q(w_{T} | x_{T}) \prod_{t=2}^T \underbrace{q(w_{t-1} | x_{t-1}) q(x_{t-1}\mid x_t,x_0)}_{q(w_{t-1}, x_{t-1} | x_{t}, x_{0}) = q(w_{t-1} | x_{t}, x_{0}) q(x_{t-1} | w_{t-1}, x_{t}, x_{0})} \\
&= q(x_0 \mid w_0) q(x_T\mid x_0) q(w_{T} | x_{T}) \prod_{t=2}^T q(w_{t-1} | x_{t}, x_{0}) q(x_{t-1} | w_{t-1}, x_{t}, x_{0}) \\
&= q(x_0 \mid w_0) q(x_T\mid x_0) q(w_{T} | x_{T}) \prod_{t=2}^T q(x_{t-1} | w_{t-1}, x_{t}, x_{0}) \prod_{t=2}^T q(w_{t-1} | x_{t}, x_{0}) \\
\end{aligned}
$$
and reverse generative model factorizes as
$$
\begin{aligned}
p_\theta(w_{0:T}, x_{0:T})
& =
p_{\theta}(x_T) p_\theta(w_0 | x_0) p_\theta(x_{0} | x_1) p_\theta(w_T | x_T) \prod_{t=2}^T p_\theta(x_{t-1} | x_t) \prod_{t=1}^{T-1} p_\theta(w_t | x_t) \\
& =
p_{\theta}(x_T) p_\theta(w_{T} | x_{T}) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)\prod_{t=1}^{T} p_\theta(w_{t-1} | x_{t-1}) \\
& =
p_{\theta}(x_T) p_\theta(w_{T} | x_{T}) \prod_{t=1}^T p_\theta(w_{t-1} | x_{t-1}) p_\theta(x_{t-1} | x_t) \\
& =
p_{\theta}(x_T) p_\theta(w_{T} | x_{T}) \prod_{t=1}^T p_\theta(w_{t-1}, x_{t-1} | x_{t}) \\
& =
p_{\theta}(x_T) p_\theta(w_{T} | x_{T}) \prod_{t=1}^T p_\theta(w_{t-1} | x_{t}) p_{\theta}(x_{t-1} | w_{t-1}, x_{t}) \\
& =
p_{\theta}(x_T) p_\theta(w_{T} | x_{T}) p_{\theta}(x_{0} | w_{0}, x_{1}) p_\theta(w_{0} | x_{1}) \prod_{t=2}^T p_{\theta}(x_{t-1} | w_{t-1}, x_{t}) \prod_{t=2}^T p_\theta(w_{t-1} | x_{t}) \\
\end{aligned}
$$

Therefore, with factorization,
$$
\begin{aligned}
p_\theta(w_{0:T}, x_{0:T})
& = p_{\theta}(x_T) p_\theta(w_{T} | x_{T}) p_{\theta}(x_{0} | w_{0}, x_{1}) p_\theta(w_{0} | x_{1}) \prod_{t=2}^T p_{\theta}(x_{t-1} | w_{t-1}, x_{t}) \prod_{t=2}^T p_\theta(w_{t-1} | x_{t}) \\
q(w_{1:T},x_{0:T}\mid w_0) 
& = q(x_0 \mid w_0) q(x_T\mid x_0) q(w_{T} | x_{T}) \prod_{t=2}^T q(x_{t-1} | w_{t-1}, x_{t}, x_{0}) \prod_{t=2}^T q(w_{t-1} | x_{t}, x_{0}),
\end{aligned}
$$
the discrete NELBO on the state $w_{t}$ of the Brownian bridge conditioned on $w_0$ is 
$$
\begin{aligned}
\mathcal L_{\text{NELBO}}(w_0)
&=
\mathbb E_q \Bigg[
\underbrace{
\log \frac{q(x_T\mid x_0)}{p_\theta(x_T)}
}_{\text{Prior}}
+
\underbrace{
\log \frac{q(w_T\mid x_T)}{p_\theta(w_T\mid x_T)}
}_{\text{Terminal Emission}} \\
& \quad +
\sum_{t=2}^T
\underbrace{
\log \frac{q(w_{t-1}\mid x_t,x_0)}{p_\theta(w_{t-1}\mid x_t)}
}_{\text{Discrete Denoising}}
+
\sum_{t=2}^T
\underbrace{
\log \frac{q(x_{t-1}\mid w_{t-1},x_t,x_0)}{p_\theta(x_{t-1}\mid w_{t-1},x_t)}
}_{\text{Continuous Refinement}} \\
&\quad
\underbrace{
-
\log p_\theta(w_0\mid x_1)
}_{\text{Reconstruction}}
+
\underbrace{
\log \frac{q(x_0\mid w_0)}{p_\theta(x_0\mid w_0,x_1)}
}_{\text{Initial}}
\Bigg] \\
&=
\underbrace{
\mathbb E_{q(x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(x_T\mid x_0)\,\|\,p_\theta(x_T)\bigr)
}_{\text{Prior}}
\\[0.5em]
&\quad+
\underbrace{
\mathbb E_{q(x_T\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(w_T\mid x_T)\,\|\,p_\theta(w_T\mid x_T)\bigr)
}_{\text{Terminal Emission }(=0\text{ if both softmax)}}
\\[0.5em]
&\quad+
\sum_{t=2}^T
\underbrace{
\mathbb E_{q(x_t,x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(
q(w_{t-1}\mid x_t,x_0)\,\|\,p_\theta(w_{t-1}\mid x_t)
\bigr)
}_{\text{Discrete Denoising}} \\
&\quad+
\sum_{t=2}^T
\underbrace{
\mathbb E_{q(w_{t-1},x_t,x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(
q(x_{t-1}\mid w_{t-1},x_t,x_0) || p_\theta(x_{t-1}\mid w_{t-1},x_t)
\bigr)
}_{\text{Continuous Refinement}} \\
&\quad+
\underbrace{
\mathbb{E}_{q(x_1 | w_0)} \Bigl[ 
- \log p_\theta(w_0 | x_1)
\Bigr]}_{\text{Reconstruction}}
+ 
\underbrace{
\mathbb{E}_{q(x_1 | w_0)} D_{\mathrm{KL}} \bigl(q(x_0 | w_0) || p_\theta(x_0\mid w_0,x_1) \bigr)
}_{\text{Initial}}.
\end{aligned}
$$

##### KL Divergence for Categorical Distribution
Note that the KL divergence between categorical distribution $p = \text{Cat}(\{p_i\}_{i=1}^{K})$ and $q = \text{Cat}(\{q_i\}_{i=1}^{K})$ is
$$
D_{\mathrm{KL}}(p\|q) = CE(p,q) - H(p) = -\sum_{i=1}^{K} p_i \log q_i + \sum_{i}^{K} p_i \log p_i
$$
##### Probit Approximation
$q(t, x) = \text{Cat}(w; \text{softmax}(z)) \mathcal{N}(z; tx, (1-t)^2 I)$
Additionally, the categorical-Gaussian distribution parameterized by softmax 
$$
q(w) := \text{Cat}(w; \text{softmax}(z)) \mathcal{N}(z; \mu, \sigma^2 I)
$$
where $\mu \in \mathbb{R}^{d}$  and $\sigma \in \mathbb{R}$ is isotropic. Then, according to MacKay probit approximation, the marginal distribution over $w$ can be approximated by
$$
q(w) \approx \text{Cat}(w; \operatorname{softmax}(\frac{\mu}{\sqrt{1 + \frac{\pi}{8} \sigma^2}})).
$$
Note that it only holds for isotropic Gaussian. The error bound can be derived for 1D case since the softmax reduces to the logistic sigmoid $\text{sigmoid}(z)$. The probit approximation relies on the fact that the sigmoid closely matches the Gaussian CDF $\Phi(z)$ when scaled: $\text{sigmoid}(z) \approx \Phi(\lambda z)$ where $\lambda^2 = \frac{\pi}{8}$. The the worst-case absolute error for probit approximation for Gaussian CDF $\Phi(z)$ is approximately 0.043.

Furthermore, for non isotropic Gaussian $q(w) := \text{Cat}(w; \text{softmax}(z)) \mathcal{N}(z; \mu, \Sigma)$ with $\Sigma \in \mathbb{R}^{K \times K}$
$$
q(w) \approx \text{Cat}\!\left(w; \operatorname{softmax}\!\left(\mu \oslash \sqrt{1 + \frac{\pi}{8}\operatorname{diag}(\Sigma)}\right)\right),
$$
where $\oslash$ is the Hadamard division, a element-wise division operator. To be more specific, the probit approximation for multivariate distribution can be expressed as
$$
q(w=c) \approx \frac{\exp\left(\mu_c \left(1 + \frac{\pi}{8} \Sigma_{c,c}\right)^{-1/2}\right)}{\sum_{j} \exp\left(\mu_j \left(1 + \frac{\pi}{8} \Sigma_{j,j}\right)^{-1/2}\right)}
$$
##### KL Divergence for Categorical Gaussian Distribution
Given $q(w) := \text{Cat}(w_q; \text{softmax}(z_q)) \mathcal{N}(z_q; \mu_q, \Sigma_q)$ and $p(w) := \text{Cat}(w_p; \text{softmax}(z_p)) \mathcal{N}(z_p; \mu_p, \Sigma_p)$, here provides 2 methods.
###### Probit Approximation
 Let the $i$-th class probability of $q$ be
$$
q_i := \frac{\exp\left(\mu_{q,i} \left(1 + \frac{\pi}{8} \Sigma_{q,i,i}\right)^{-1/2}\right)}{\sum_{j} \exp\left(\mu_{q,j} \left(1 + \frac{\pi}{8} \Sigma_{q,j,j}\right)^{-1/2}\right)}
$$
and the $i$-th class probability of $p$ be
$$
p_i := \frac{\exp\left(\mu_{p,i} \left(1 + \frac{\pi}{8} \Sigma_{p,i,i}\right)^{-1/2}\right)}{\sum_{j} \exp\left(\mu_{p,j} \left(1 + \frac{\pi}{8} \Sigma_{p,j,j}\right)^{-1/2}\right)}.
$$
###### MC Estimation
Drawing $L$ samples from Gaussian distributions, denote the $l$-th samples as $z_q^{(l)}$ and $z_p^{(l)}$
$$
\begin{aligned}
z_q^{(l)} \sim \mathcal{N}(\mu_q, \Sigma_q), \quad z_p^{(l)} \sim \mathcal{N}(\mu_p, \Sigma_p)
\end{aligned}
$$
Therefore, the $i$-th logit of $q$ as $q_i = \left[ \frac{1}{L} \sum_{l=1}^L \text{softmax}(z_q^{(l)}) \right]_i$ and the $i$-th logit of $p$ as $p_i = \left[ \frac{1}{L} \sum_{l=1}^L \text{softmax}(z_p^{(l)}) \right]_i$

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
& = 0, \\
L_{i-1}
& = \mathbb{E}_{q(x_{\tau(i)} \mid y)}
\left[
\frac{\Delta t}{2 g(\tau(i), x_{\tau(i)})^2}
\Big\| f(\tau(i), x_{\tau(i)}, y) - f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Big\|^2
\right], \\
L_0
& = \mathbb{E}_{q(x_{\tau(1)} \mid y)}
\left[
\frac{d}{2} \log 2 \pi
+ d \log g(\tau(1), x_{\tau(1)})
+ \frac{d}{2} \log \Delta t
+ \frac{1}{2 g(\tau(1), x_{\tau(1)})^2 \Delta t}
\Big\| y - x_{\tau(1)} - f(\tau(1), x_{\tau(1)}, f_\theta(x_{\tau(1)}, \tau(1))) \Delta t \Big\|^2
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
L_{0} = 
& \frac{d}{2} \log 2 \pi 
+ d \log \left( \frac{1-\|x_{\tau(1)}\|^2}{2} \right)
+ \frac{d}{2} \log \Delta t
  + \frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta t}
\Bigg\| y - x_{\tau(1)} \\
& - \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2}
\big(f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta t \Bigg\|^2
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
\big(x_{\tau(i)} + f(\tau(i), x_{\tau(i)}, f_\theta(x_{\tau(i)}, \tau(i))) \Delta t\big)
- \big(x_{\tau(i)} + f(\tau(i), x_{\tau(i)}, y) \Delta t\big)
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
As we assume the distribution is a isotropic Gaussian with variance $\sigma^2$, the negative log is
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
& = - \log \mathcal{N}\!\big(y;\, x_{\tau(1)} + f(\tau(1), x_{\tau(1)}, f_{\theta}(x_{\tau(1)}, \tau(1))) \Delta t,\; g(\tau(1), x_{\tau(1)})^2 \Delta t I\big) \\
& = \frac{d}{2} \log 2 \pi
+ d \log g(\tau(1), x_{\tau(1)})
+ \frac{d}{2} \log \Delta t \\
& \quad + \frac{1}{2 g(\tau(1), x_{\tau(1)})^2 \Delta t}
\Big\| y - x_{\tau(1)} - f(\tau(1), x_{\tau(1)}, f_{\theta}(x_{\tau(1)}, \tau(1))) \Delta t \Big\|^2 \\
& = \frac{d}{2} \log 2 \pi 
+ d \log \left( \frac{1-\|x_{\tau(1)}\|^2}{2} \right)
+ \frac{d}{2} \log \Delta t \\
& \quad + \frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta t}
\Bigg\| y - x_{\tau(1)}
- \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\|^2}
\big(f_{\theta}(x_{\tau(1)}, \tau(1)) - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta t \Bigg\|^2
\end{aligned}
$$
### Local-Chart Approximated NELBO in Simplex Space with Small Step
This derivation is a local-chart, small-step approximate NELBO induced by the Euler-Gaussian approximation above together with the Simplex projection $w_t = softmax(\frac{E x_t}{\mathcal{T}})$; it is not the exact manifold ELBO on discrete states. Recall the NELBO on the simplex space is

%% # ELBO Ver. 2 %%
$$
\begin{aligned}
\mathcal L_{\text{NELBO}}(w_0)
&=
\underbrace{
\mathbb E_{q(x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(x_T\mid x_0)\,\|\,p_{\theta}(x_T)\bigr)
}_{L_{\text{Prior}}}
+
\underbrace{
\sum_{t=2}^T
\mathbb E_{q(x_t,x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)
}_{L_{\text{Diffusion}}}
\\
&\quad+
\underbrace{
\mathbb E_{q(x_0,x_1\mid w_0)}
\bigl[-\log p_\theta(x_0\mid x_1)\bigr]
}_{L_{\text{Reconstruction}}}
+
\underbrace{
\sum_{t=1}^T
\mathbb E_{q(x_t\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(w_t\mid x_t)\,\|\,p_\theta(w_t\mid x_t)\bigr)
}_{L_{\text{Projection}}}
\\
&\quad+
\underbrace{
\mathbb E_{q(x_0\mid w_0)}
\bigl[\log q(x_0\mid w_0)-\log p_\theta(w_0\mid x_0)\bigr]
}_{L_{\text{Init}}}.
\end{aligned}
$$
#### Prior $:L_{Prior}$
$$
\begin{aligned}
L_{\text{Prior}} 
& = \mathbb E_{q(x_0\mid w_0)} \left[ D_{\mathrm{KL}}\!\bigl(q(x_T\mid x_0)\,\|\,p_{\theta}(x_T)\bigr) \right] = 0
\end{aligned}
$$
#### Diffusion $:L_{Diffusion}$
Let
$$
\begin{aligned}
x_i &:= x_{\tau(i)}, \qquad
w_i := h_{\mathcal T}(x_i), \qquad
z_i := E x_i, \\
\hat y_{\theta,i}
&:= \frac{f_\theta(x_i,\tau(i))}{\|f_\theta(x_i,\tau(i))\|}, \\
\hat f_i^q
&:= \hat f(\tau(i),x_i,y), \qquad
\hat f_i^\theta := \hat f(\tau(i),x_i,\hat y_{\theta,i}), \\
C_i &:= C_w(\tau(i),x_i), \qquad
G_i := G(w_i), \qquad
\Gamma_i := G_i^+ C_i G_i^+,
\end{aligned}
$$
and for the softmax-pushforward approximation let
$$
A := \begin{bmatrix} I_{K-1} & -\mathbf 1 \end{bmatrix} \in \mathbb R^{(K-1)\times K}.
$$
Then:
##### **Euler-Gaussian**
$$
\begin{aligned}
L_{\text{Diffusion}}^{\mathrm{EG}}
& \approx \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[ D_{\mathrm{KL}}\!\left(
\mathcal N(w_i+\hat f_i^q\Delta t,\; C_i\Delta t)
\;\middle\|\;
\mathcal N(w_i+\hat f_i^\theta\Delta t,\; C_i\Delta t)
\right)\right] \\
& = \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[
\frac{\Delta t}{2}(\hat f_i^\theta-\hat f_i^q)^\top C_i^+ (\hat f_i^\theta-\hat f_i^q)
\right].
\end{aligned}
$$
For the original local-chart bridge drift, the Euler-Gaussian term reduces to
$$
\begin{aligned}
L_{\text{Diffusion}}^{\mathrm{EG}}
& = \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[ \frac{\Delta t}{2} (d-1)^2 (1-\|x_i\|^2)^2
\left\|
\frac{y-x_i}{\|y-x_i\|^2}
- \frac{\hat y_{\theta,i}-x_i}{\|\hat y_{\theta,i}-x_i\|^2}
\right\|^2 \right].
\end{aligned}
$$
##### **Softmax Pushforward**
Let
$$
\delta_i
:=
\frac{y-x_i}{\|y-x_i\|^2}
-
\frac{\hat y_{\theta,i}-x_i}{\|\hat y_{\theta,i}-x_i\|^2}.
$$
$$
\begin{aligned}
m_i^{q,\mathrm{SP}}
&:= \frac{1}{\mathcal T} A\big(z_i + E f(\tau(i),x_i,y)\Delta t\big), \\
m_i^{\theta,\mathrm{SP}}
&:= \frac{1}{\mathcal T} A\big(z_i + E f(\tau(i),x_i,\hat y_{\theta,i})\Delta t\big), \\
\Sigma_i^{\mathrm{SP}}
&:= \frac{g(\tau(i),x_i)^2\Delta t}{\mathcal T^2} A E E^\top A^\top.
\end{aligned}
$$
Then
$$
\begin{aligned}
L_{\text{Diffusion}}^{\mathrm{SP}}
& \approx \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[
D_{\mathrm{KL}}\!\left(
\mathcal N(m_i^{q,\mathrm{SP}}, \Sigma_i^{\mathrm{SP}})
\;\middle\|\;
\mathcal N(m_i^{\theta,\mathrm{SP}}, \Sigma_i^{\mathrm{SP}})
\right)\right] \\
& = \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[
\frac{1}{2}(m_i^{\theta,\mathrm{SP}}-m_i^{q,\mathrm{SP}})^\top (\Sigma_i^{\mathrm{SP}})^{-1} (m_i^{\theta,\mathrm{SP}}-m_i^{q,\mathrm{SP}})
\right].
\end{aligned}
$$
For the original local-chart bridge drift, this reduces to
$$
\begin{aligned}
L_{\text{Diffusion}}^{\mathrm{SP}}
& = \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \Bigg[
\frac{\Delta t}{2}(d-1)^2(1-\|x_i\|^2)^2 \,
\delta_i^\top E^\top A^\top (A E E^\top A^\top)^+ A E \, \delta_i
\Bigg].
\end{aligned}
$$

##### **Natural Gradient**
$$
\begin{aligned}
\nu_i^q &:= G_i^+ \hat f_i^q \Delta t, \qquad
\nu_i^\theta := G_i^+ \hat f_i^\theta \Delta t.
\end{aligned}
$$
Then
$$
\begin{aligned}
L_{\text{Diffusion}}^{\mathrm{NG}}
& \approx \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[
D_{\mathrm{KL}}\!\left(
\mathcal N_T(\nu_i^q,\Gamma_i\Delta t)
\;\middle\|\;
\mathcal N_T(\nu_i^\theta,\Gamma_i\Delta t)
\right)\right] \\
& = \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \left[
\frac{1}{2}(\nu_i^\theta-\nu_i^q)^\top (\Gamma_i\Delta t)^+ (\nu_i^\theta-\nu_i^q)
\right].
\end{aligned}
$$
Using $\hat f_i^\theta-\hat f_i^q = - \frac{d-1}{2}(1-\|x_i\|^2)^2 J_h(x_i)\delta_i$, the natural-gradient term becomes
$$
\begin{aligned}
L_{\text{Diffusion}}^{\mathrm{NG}}
& = \sum_{i=2}^{T} \mathbb E_{q(x_i, x_{\tau(0)} \mid w_{\tau(0)})} \Bigg[
\frac{\Delta t}{2}(d-1)^2(1-\|x_i\|^2)^2 \,
\delta_i^\top J_h(x_i)^\top G_i^+ \big(G_i^+ J_h(x_i)J_h(x_i)^\top G_i^+\big)^+ G_i^+ J_h(x_i)\,\delta_i
\Bigg].
\end{aligned}
$$
#### Reconstruction $:L_{Reconstruction}$
Let
$$
\hat{y}_{\theta,1}
:=
\frac{f_\theta(x_{\tau(1)}, \tau(1))}{\|f_\theta(x_{\tau(1)}, \tau(1))\|}.
$$

##### **Euler-Gaussian**
$$
\begin{aligned}
L_{\text{Reconstruction}}^{\mathrm{EG}}
& = \mathbb E_{q(x_{\tau(0)},x_{\tau(1)}\mid w_{\tau(0)})}
\left[
-\log p_\theta(x_{\tau(0)}\mid x_{\tau(1)})
\right] \\
& \approx
\mathbb E_{q(x_{\tau(0)},x_{\tau(1)}\mid w_{\tau(0)})}
\Bigg[
\frac{d}{2}\log 2\pi
+ d \log\left(\frac{1-\|x_{\tau(1)}\|^2}{2}\right)
+ \frac{d}{2}\log \Delta t \\
& \qquad\qquad
+ \frac{2}{(1-\|x_{\tau(1)}\|^2)^2\Delta t}
\Bigg\|
x_{\tau(0)} - x_{\tau(1)}
- \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|\hat{y}_{\theta,1}-x_{\tau(1)}\|^2}
\big(\hat{y}_{\theta,1}-x_{\tau(1)}\big)
- \frac{d}{4}(1-\|x_{\tau(1)}\|^2)x_{\tau(1)}
\Bigg)\Delta t
\Bigg\|^2
\Bigg].
\end{aligned}
$$

##### **Softmax Pushforward**
Let
$$
\begin{aligned}
r(w_{\tau(0)}) &:= A \log w_{\tau(0)}, \\
m_1^{\theta,\mathrm{SP}}
&:= \frac{1}{\mathcal T} A\big(z_1 + E f(\tau(1),x_{\tau(1)},\hat y_{\theta,1})\Delta t\big), \\
\Sigma_1^{\mathrm{SP}}
&:= \frac{g(\tau(1),x_{\tau(1)})^2\Delta t}{\mathcal T^2} A E E^\top A^\top.
\end{aligned}
$$
Then
$$
\begin{aligned}
L_{\text{Reconstruction}}^{\mathrm{SP}}
& \approx \mathbb E_{q(x_{\tau(0)},x_{\tau(1)}\mid w_{\tau(0)})}
\left[
-\log \Big(\mathcal S_{\mathcal T\#}\mathcal N\big(z_1 + E f(\tau(1),x_{\tau(1)},\hat y_{\theta,1})\Delta t,\; g(\tau(1),x_{\tau(1)})^2 E E^\top \Delta t\big)\Big)(w_{\tau(0)})
\right] \\
& = \mathbb E_{q(x_{\tau(0)},x_{\tau(1)}\mid w_{\tau(0)})}
\left[
\frac{1}{2}\log \operatorname{pdet}(2\pi \Sigma_1^{\mathrm{SP}})
+ \frac{1}{2}\big(r(w_{\tau(0)})-m_1^{\theta,\mathrm{SP}}\big)^\top (\Sigma_1^{\mathrm{SP}})^+ \big(r(w_{\tau(0)})-m_1^{\theta,\mathrm{SP}}\big)
\right].
\end{aligned}
$$

##### **Natural Gradient**
Let
$$
\begin{aligned}
\xi_0 &:= \operatorname{clr}(w_{\tau(0)}) - \operatorname{clr}(w_{\tau(1)}), \\
\nu_1^\theta &:= G(w_{\tau(1)})^+ \hat f(\tau(1),x_{\tau(1)},\hat y_{\theta,1})\Delta t, \\
\Gamma_1 &:= G(w_{\tau(1)})^+ C_w(\tau(1),x_{\tau(1)}) G(w_{\tau(1)})^+.
\end{aligned}
$$
Then
$$
\begin{aligned}
L_{\text{Reconstruction}}^{\mathrm{NG}}
& \approx \mathbb E_{q(x_{\tau(0)},x_{\tau(1)}\mid w_{\tau(0)})}
\left[
-\log \mathcal N_T(\xi_0;\nu_1^\theta,\Gamma_1\Delta t)
\right] \\
& = \mathbb E_{q(x_{\tau(0)},x_{\tau(1)}\mid w_{\tau(0)})}
\left[
\frac{1}{2}\log \operatorname{pdet}(2\pi \Gamma_1\Delta t)
+ \frac{1}{2}(\xi_0-\nu_1^\theta)^\top (\Gamma_1\Delta t)^+ (\xi_0-\nu_1^\theta)
\right].
\end{aligned}
$$
#### Projection $:L_{Projection}$
For the simplex projection, use the same deterministic map under both $q$ and $p_\theta$:
$$
q(w_{\tau(i)}\mid x_{\tau(i)})
=
p_\theta(w_{\tau(i)}\mid x_{\tau(i)})
=
\delta_{h_{\mathcal T}(x_{\tau(i)})}.
$$
Hence
$$
\begin{aligned}
L_{\text{Projection}}
&=
\sum_{i=1}^{T}
\mathbb E_{q(x_{\tau(i)}\mid w_{\tau(0)})}
\left[
D_{\mathrm{KL}}\!\bigl(q(w_{\tau(i)}\mid x_{\tau(i)})\,\|\,p_\theta(w_{\tau(i)}\mid x_{\tau(i)})\bigr)
\right] \\
&= 0.
\end{aligned}
$$
#### Init $:L_{Init}$
The init term remains
$$
\begin{aligned}
L_{\text{Init}}
&=
\mathbb E_{q(x_{\tau(0)}\mid w_{\tau(0)})}
\left[
\log q(x_{\tau(0)}\mid w_{\tau(0)})
- \log p_\theta(w_{\tau(0)}\mid x_{\tau(0)})
\right].
\end{aligned}
$$
If we choose a deterministic local inverse $\psi$ such that
$$
q(x_{\tau(0)}\mid w_{\tau(0)})=\delta_{\psi(w_{\tau(0)})},
\qquad
w_{\tau(0)} = h_{\mathcal T}(\psi(w_{\tau(0)})),
$$
and use the same deterministic projection
$$
p_\theta(w_{\tau(0)}\mid x_{\tau(0)})=\delta_{h_{\mathcal T}(x_{\tau(0)})},
$$
then
$$
L_{\text{Init}} = 0.
$$
### Local-Chart Approximated NELBO in Discrete Space with Small Step
For a probability model $p_{\theta}$ and a categorical process $w_{1:T}$, the NELBO of the bridge diffusion conditioned on $w_0$ is following.

%% # ELBO Ver. 1 %%
$$
\begin{aligned}
- \log p_{\theta}(w_0) 
& \leq \mathbb E_{q(w_{1:T},x_{0:T}\mid w_0)}
\left[
\log \frac{q(w_{1:T},x_{0:T}\mid w_0)}{p_\theta(w_{0:T},x_{0:T})}
\right] \\
& = \underbrace{
\mathbb E_{q(x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(x_T\mid x_0)\,\|\,p_\theta(x_T)\bigr)
}_{\text{Prior}}
\\[0.5em]
&\quad+
\underbrace{
\mathbb E_{q(x_T\mid w_0)}
D_{\mathrm{KL}}\!\bigl(q(w_T\mid x_T)\,\|\,p_\theta(w_T\mid x_T)\bigr)
}_{\text{Terminal Emission }(=0\text{ if both softmax)}}
\\[0.5em]
&\quad+
\sum_{t=2}^T
\underbrace{
\mathbb E_{q(x_t,x_0\mid w_0)}
D_{\mathrm{KL}}\!\bigl(
q(w_{t-1}\mid x_t,x_0)\,\|\,p_\theta(w_{t-1}\mid x_t)
\bigr)
}_{\text{Discrete Denoising}} \\
&\quad+
\sum_{t=2}^T
\underbrace{
\mathbb E_{q(w_{t-1},x_t,x_0\mid w_0)}
D_{\mathrm{KL}} \bigl(
q(x_{t-1} \mid w_{t-1},x_t,x_0) || p_\theta(x_{t-1}\mid w_{t-1},x_t)
\bigr)
}_{\text{Continuous Refinement}} \\
&\quad+
\underbrace{
\mathbb{E}_{q(x_1 | w_0)} \Bigl[ 
- \log p_\theta(w_0 | x_1)
\Bigr]}_{\text{Reconstruction}}
+ 
\underbrace{
\mathbb{E}_{q(x_1 | w_0)} D_{\mathrm{KL}} \bigl(q(x_0 | w_0) || p_\theta(x_0\mid w_0,x_1) \bigr)
}_{\text{Initial}}.
\end{aligned}
$$
#### Prior $:L_{Prior}$
$$
\begin{aligned}
L_{\text{Prior}} 
& = \mathbb E_{q(x_0 | w_0)} \left[ D_{\mathrm{KL}} \bigl(q(x_T | x_0) || p_{\theta}(x_T) \bigr) \right] = 0
\end{aligned}
$$
#### Terminal Emission $:L_{Terminal}$
$$
L_{Terminal} = \mathbb E_{q(x_T\mid w_0)}
D_{\mathrm{KL}} \bigl(q(w_T\mid x_T) || p_\theta(w_T\mid x_T)\bigr) = 0
$$
#### Discrete Denoising $:L_{Discrete}$
After projecting the Gaussian local-chart transition to the discrete state space, the exact categorical transition is no longer available in closed form. We therefore apply the diagonal probit approximation to the analytical Gaussian moments to estimate $q(w_{t-1} \vert x_t, x_0)$ and $p_\theta(w_{t-1} \vert x_t)$ without Monte Carlo sampling. Let
$$
z(x) := \frac{E x}{\mathcal T} \in \mathbb{R}^{K}.
$$
The discrete transitions marginalize over the latent continuous state:
$$
\begin{aligned}
q(w_{t-1} \vert x_t, x_0)
& = \int q(w_{t-1} \mid x_{t-1})\, q(x_{t-1} \mid x_t, x_0) \, d x_{t-1}, \\
p_\theta(w_{t-1} \vert x_t)
& = \int p_{\theta}(w_{t-1} \mid x_{t-1})\, p_{\theta}(x_{t-1} \mid x_t) \, d x_{t-1}.
\end{aligned}
$$
From the Euler-Gaussian approximation, the ambient-space transitions are
$$
\begin{aligned}
q(x_{t-1} \mid x_t, x_0) &\approx \mathcal{N}\!\big(x_{t-1};\; m_{t-1}^{q},\; \sigma_{t-1}^2\, I_d\big), \\
p_\theta(x_{t-1} \mid x_t) &\approx \mathcal{N}\!\big(x_{t-1};\; m_{t-1}^{\theta},\; \sigma_{t-1}^2\, I_d\big),
\end{aligned}
$$
where
$$
\begin{aligned}
m_{t-1}^{q} := x_t + f(\tau(t), x_t, x_0)\,\Delta t, \quad
m_{t-1}^{\theta} := x_t + f(\tau(t), x_t, \hat{y}_\theta)\,\Delta t, \quad
\sigma_{t-1}^2 := g(\tau(t), x_t)^2\,\Delta t, \\
\end{aligned}
$$
and
$$
\hat{y}_\theta := \frac{f_{\theta}(x_t, \tau(t))}{|| f_{\theta}(x_t, \tau(t)) ||}
$$The probit approximation computes the marginal categorical probability $\mathbb{E}_{x_{t-1}}[\operatorname{softmax}(E x_{t-1}/\mathcal{T})]$. Since the logit mean and diagonal variance are
$$
\mathbb{E}\!\left[\frac{E x_{t-1}}{\mathcal{T}}\right] = \frac{E\, m}{\mathcal{T}}, \qquad
\operatorname{Var}\!\left[\frac{[E x_{t-1}]_k}{\mathcal{T}}\right] = \frac{\sigma^2 \|e_k\|^2}{\mathcal{T}^2},
$$
where $e_k \in \mathbb{R}^d$ is the $k$-th row of $E$, the diagonal probit approximation gives
$$
\begin{aligned}
\hat{\pi}_{t-1, k}^{q}
&:= \operatorname{softmax}\!\left(
\frac{E\, m_{t-1}^{q}}{\mathcal{T}} \oslash \sqrt{ 1 + \frac{\pi}{8} \frac{\sigma_{t-1}^2 \|e_k\|^2}{\mathcal{T}^2} } \right), \\
\hat{\pi}_{t-1, k}^{\theta}
&:= \operatorname{softmax}\!\left(
\frac{E\, m_{t-1}^{\theta}}{\mathcal{T}} \oslash \sqrt{ 1 + \frac{\pi}{8} \frac{\sigma_{t-1}^2 \|e_k\|^2}{\mathcal{T}^2} } \right),
\end{aligned}
$$
where $\oslash$ is element-wise division and each $k$-th component is scaled by its own factor. Since $\sigma_{t-1}^2$ is shared, the scaling denominator is the same for both $\hat{\pi}^q$ and $\hat{\pi}^\theta$. Thus
$$
q(w_{t-1}\mid x_t,x_0) \approx \text{Cat}(w_{t-1};\hat{\pi}_{t-1}^{q}),
\qquad
p_\theta(w_{t-1}\mid x_t) \approx \text{Cat}(w_{t-1};\hat{\pi}_{t-1}^{\theta}).
$$
Hence the discrete denoising term is estimated by
$$
\begin{aligned}
L_{\text{Discrete}}
& \approx \sum_{t=2}^T
\mathbb E_{q(x_t,x_0\mid w_0)}
\left[
D_{\mathrm{KL}}\!\bigl(
\text{Cat}(\hat{\pi}_{t-1}^{q})
\;\|\;
\text{Cat}(\hat{\pi}_{t-1}^{\theta})
\bigr)
\right] \\
& =
\sum_{t=2}^T
\mathbb E_{q(x_t,x_0\mid w_0)}
\left[
\sum_{k=1}^{K}
\hat{\pi}_{t-1,k}^{q}
\log \frac{\hat{\pi}_{t-1,k}^{q}}{\hat{\pi}_{t-1,k}^{\theta}}
\right].
\end{aligned}
$$
#### Continuous Refinement: $L_{Refinement}$

The Discrete Denoising term above handles *which category* $w_{t-1}$ to pick. The Continuous Refinement term handles the complementary question: *given* the chosen category $w_{t-1}$, where should the continuous state $x_{t-1}$ land?

$$
L_{\text{Refinement}} = \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} D_{\mathrm{KL}}\!\bigl(q(x_{t-1}\mid w_{t-1}, x_t, x_0) \;\|\; p_\theta(x_{t-1}\mid w_{t-1}, x_t)\bigr).
$$
##### Intractable posterior via Bayesian rule
By Bayesian rule,
$$
\begin{aligned}
q(x_{t-1} | w_{t-1}, x_t, x_0) 
& = \frac{q(w_{t-1} | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0)}{q(w_{t-1} | x_t, x_0)} \\
& = \frac{q(w_{t-1} | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0)}{\int q(w_{t-1} | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0) d x_{t-1}}, \\
\end{aligned}
$$
where $q(x_{t-1}\mid x_t, x_0) = \mathcal{N}(x_{t-1};\, m_{t-1}^{q},\, \sigma_{t-1}^2 I_d)$ is the Euler-Gaussian bridge posterior with

$$

m_{t-1}^{q} := x_t + f(\tau(t), x_t, x_0)\,\Delta t, \qquad \Sigma_{t-1} := g(\tau(t), x_t)^2\,\Delta t,

$$

and $q(w_{t-1}\mid x_{t-1}) = \mathrm{Cat}(w_{t-1};\, \operatorname{softmax}(E x_{t-1}/\mathcal{T}))$ is the softmax emission.

Considering the case $q(x_{t-1} | w_{t-1} = e_k, x_t, x_0)$,
$$
\begin{aligned}
q(x_{t-1} | w_{t-1} = e_k, x_t, x_0) 
= \frac{\operatorname{softmax}_{k} \left(\frac{E x_{t-1}}{\mathcal{T}}\right)}{\operatorname{softmax}\!\left(
\frac{E\, m_{t-1}^{q}}{\mathcal{T}} \oslash \sqrt{ 1 + \frac{\pi}{8} \frac{\sigma_{t-1}^2 \|e_k\|^2}{\mathcal{T}^2} } \right)} 
	\cdot \mathcal{N}(x_{t-1}; m_{t-1}^{q},\, \Sigma_{t-1})
\end{aligned}
$$
The density $q(x_{t-1} \mid w_{t-1}=e_k, x_t, x_0)$ is not Gaussian, because the factor
$$
q(w_{t-1}=e_k \mid x_{t-1}) = \operatorname{softmax}_{k}\!\left(\frac{E x_{t-1}}{\mathcal{T}}\right)
$$
depends on the random variable $x_{t-1}$. Therefore, we use a Laplace approximation to find a Gaussian distribution whose mean and variance matches $q(x_{t-1} | w_{t-1} = e_k, x_t, x_0)$.

##### Method 1: KL Divergence Chain Rule
Recall that by Bayesian rule, the $q$ and $p_{\theta}$ are
$$
\begin{aligned}
q(x_{t-1} | w_{t-1}, x_t, x_0) 
& = \frac{q(w_{t-1} | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0)}{q(w_{t-1} | x_t, x_0)} \\
& = \frac{q(w_{t-1} | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0)}{\int q(w_{t-1} | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0) d x_{t-1}}, \\
& = \frac{\operatorname{Cat}(w_{t-1}; \operatorname{softmax}(\frac{E x_{t-1}}{\mathcal{T}})) \mathcal{N}(x_{t-1}; m_{t-1}^{q}, \sigma_{t-1}^2)}{\operatorname{Cat}(w_{t-1}; \hat{\pi}_{t-1}^{q})} \\\end{aligned}
$$
$$
\begin{aligned}
p_{\theta}(x_{t-1} | w_{t-1}, x_t) 
& = \frac{p_{\theta}(w_{t-1} | x_{t-1}) \cdot p_{\theta}(x_{t-1} | x_t)}{p_{\theta}(w_{t-1} | x_t)} \\
& = \frac{p_{\theta}(w_{t-1} | x_{t-1}) \cdot p_{\theta}(x_{t-1} | x_t)}{\int p_{\theta}(w_{t-1} | x_{t-1}) \cdot p_{\theta}(x_{t-1} | x_t) d x_{t-1}}, \\
& = \frac{\operatorname{Cat}(w_{t-1}; \operatorname{softmax}(\frac{E x_{t-1}}{\mathcal{T}})) \mathcal{N}(x_{t-1}; m_{t-1}^{\theta}, \sigma_{t-1}^{2})}{\operatorname{Cat}(w_{t-1}; \hat{\pi}_{t-1}^{\theta})}
\end{aligned}
$$
where
$$
\begin{aligned}
m_{t-1}^{q} := x_t + f(\tau(t), x_t, x_0)\,\Delta t, \quad
m_{t-1}^{\theta} := x_t + f(\tau(t), x_t, \hat{y}_\theta)\,\Delta t, \quad
\sigma_{t-1}^2 := g(\tau(t), x_t)^2\,\Delta t, \\
\end{aligned}
$$
and 
$$
\begin{aligned}
\hat{\pi}_{t-1, k}^{q}
&:= \operatorname{softmax}\!\left(
\frac{E\, m_{t-1}^{q}}{\mathcal{T}} \oslash \sqrt{ 1 + \frac{\pi}{8} \frac{\sigma_{t-1}^2 \|e_k\|^2}{\mathcal{T}^2} } \right), \\
\hat{\pi}_{t-1, k}^{\theta}
&:= \operatorname{softmax}\!\left(
\frac{E\, m_{t-1}^{\theta}}{\mathcal{T}} \oslash \sqrt{ 1 + \frac{\pi}{8} \frac{\sigma_{t-1}^2 \|e_k\|^2}{\mathcal{T}^2} } \right).
\end{aligned}
$$By the chain rule of KL Divergence,
$$
\begin{aligned}
D_{\mathrm{KL}} \bigl(
q || p_{\theta}
\bigr)
& = D_{KL} \bigl( q(w_{t-1} | x_{t-1}) || p_{\theta}(w_{t-1} | x_{t-1}) \bigr) 
+ D_{KL} \bigl( q(x_{t-1} | x_{t}, x_{0}) || p_{\theta}(x_{t-1} | x_{t}) \bigr)
- D_{KL} \bigl( q(w_{t-1} | x_{t}, x_{0}) || p_{\theta}(w_{t-1} | x_{t}) \bigr) \\
& = \frac{|| m_{t-1}^{q} - m_{t-1}^{\theta} ||}{2 \sigma_{t-1}^2} 
- \sum_{k=1}^{K}
\hat{\pi}_{t-1,k}^{q} \log \frac{\hat{\pi}_{t-1,k}^{q}}{\hat{\pi}_{t-1,k}^{\theta}}
\end{aligned}
$$
##### Method 2: Gaussian Stein

Recall that by Bayesian rule expansion provided in Method 1: KL Divergence Chain Rule, the $q$ and $p_{\theta}$ are
$$
\begin{aligned}
q(x_{t-1} | w_{t-1}=e_k, x_t, x_0) 
& = \frac{\operatorname{softmax}_{k}(\frac{E x_{t-1}}{\mathcal{T}}) \mathcal{N}(x_{t-1}; m_{t-1}^{q}, \sigma_{t-1}^2)}{\hat{\pi}_{t-1, k}^{q}} \\
& = \frac{1}{\hat{\pi}_{t-1, k}^{q}} \mathbb{E}_{x_{t-1} \sim \mathcal{N}(m_{t-1}^{q}, \sigma_{t-1}^2 I)} \left[ \operatorname{softmax}_{k}(\frac{E x_{t-1}}{\mathcal{T}})\right]
\end{aligned}
$$
$$
\begin{aligned}
p_{\theta}(x_{t-1} | w_{t-1} = e_k, x_t)
& = \frac{\operatorname{softmax}_{k}(\frac{E x_{t-1}}{\mathcal{T}}) \mathcal{N}(x_{t-1}; m_{t-1}^{\theta}, \sigma_{t-1}^2)}{\hat{\pi}_{t-1, k}^{\theta}} \\
& = \frac{1}{\hat{\pi}_{t-1, k}^{\theta}} \mathbb{E}_{x_{t-1} \sim \mathcal{N}(m_{t-1}^{\theta}, \sigma_{t-1}^2 I)} \left[ \operatorname{softmax}_{k}(\frac{E x_{t-1}}{\mathcal{T}})\right] \\
\end{aligned}
$$

The KL divergence between them is
$$
\begin{aligned}
D_{KL}(q(x_{t-1} | w_{t-1}, x_t, x_0) || p_{\theta}(x_{t-1} | w_{t-1}, x_t))
& = \int_{x_{t-1}} \int_{x_{t}} \int_{x_{0}} \sum_{w_{t-1}} q(x_{t-1} | w_{t-1}, x_t, x_0) \log \frac{q(x_{t-1} | w_{t-1}, x_t, x_0)}{p_{\theta}(x_{t-1} | w_{t-1}, x_t)}
\end{aligned}
$$
**Gaussian Stein Identity Overview**
Stein's method (Stein 1972, Chen et al. Ch. 2) rests on one key fact: if $W \sim \mathcal{N}(0, 1)$, then
$$
\mathbb{E}[f'(W)] = \mathbb{E}[W f(W)]
$$
for every absolutely continuous $f$ with $\mathbb{E}|f'(W)| < \infty$, and conversely this identity *characterizes* the standard normal. The proof is a single integration by parts: $\int w f(w) e^{-w^2/2} dw = -\int f(w)\, d(e^{-w^2/2}) = \int f'(w) e^{-w^2/2} dw$.

**General Gaussian Stein Identity.** Let $p(x) = \mathcal{N}(x;\, m,\, \sigma^2 I_d)$, $m \in \mathbb{R}^d$, $\sigma > 0$. For any differentiable $h: \mathbb{R}^d \to \mathbb{R}$ satisfying $|h(x)|, \|\nabla_x h(x)\| = o(e^{\|x-m\|^2/(2\sigma^2)})$ as $\|x\| \to \infty$ (i.e. $h$ and $\nabla h$ grow slower than the Gaussian density decays — polynomial growth or bounded $h$ is more than enough), the first-order Gaussian Stein identity states
$$
\boxed{\mathbb{E}_{x \sim p}\!\left[(x - m)\, h(x)\right] = \sigma^2\, \mathbb{E}_{x \sim p}\!\left[\nabla_x h(x)\right].}
$$
Both sides are $d$-dimensional vectors. We prove this component-wise: for each $i \in \{1, \dots, d\}$,
$$
\mathbb{E}_p[(x_i - m_i)\, h(x)] = \sigma^2\, \mathbb{E}_p[\partial_i h(x)]. \tag{$\ast_i$}
$$
Furthermore, the second-order Gaussian Stein identity states
$$
\boxed{\mathbb{E}_{x \sim p}\!\left[(x - m) (x - m)^{\top} h(x)\right] = \sigma^2 I_d + \sigma^2\, \mathbb{E}_{x \sim p} \left[(x - m) (\nabla_x h(x))^\top\right].}
$$

> Proof: Gaussian integration by parts

**Key observation (prior score).** The density $p(x) = (2\pi\sigma^2)^{-d/2} \exp(-\|x-m\|^2 / (2\sigma^2))$ satisfies
$$
\partial_i p(x) = -\frac{x_i - m_i}{\sigma^2}\, p(x)
\quad\Longleftrightarrow\quad
(x_i - m_i)\, p(x) = -\sigma^2\, \partial_i p(x). \tag{$\dagger$}
$$
Multiplying by $(x_i - m_i)$ under a Gaussian is the same, up to $-\sigma^2$, as differentiating the density.

**Integration by parts.** Substitute $(\dagger)$ into the LHS of $(\ast_i)$:
$$
\mathbb{E}_p[(x_i - m_i) h(x)] = \int_{\mathbb{R}^d} (x_i - m_i)\, h(x)\, p(x)\, 
dx = -\sigma^2 \int_{\mathbb{R}^d} h(x)\, \partial_i p(x)\, dx.
$$
Split $x = (x_i, x_{-i})$ via Fubini and integrate by parts in $x_i$:
$$
\int_{-\infty}^{\infty} h\, \partial_i p\, dx_i = \bigl[h(x)\, p(x)\bigr]_{x_i = -\infty}^{x_i = +\infty} - \int_{-\infty}^{\infty} (\partial_i h)\, p\, dx_i.
$$
The boundary term vanishes: as $|x_i| \to \infty$, $p(x) \lesssim e^{-x_i^2/(2\sigma^2)}$ while $|h(x)| = o(e^{\|x-m\|^2/(2\sigma^2)})$, so $h(x) p(x) \to 0$. Hence
 $$
\int_{\mathbb{R}^d} h(x)\, \partial_i p(x)\, dx = -\int_{\mathbb{R}^d} (\partial_i h)\, p\, dx = -\mathbb{E}_p[\partial_i h(x)].
$$
Substituting back: $\mathbb{E}_p[(x_i - m_i) h(x)] = -\sigma^2 (-\mathbb{E}_p[\partial_i h]) = \sigma^2\, \mathbb{E}_p[\partial_i h(x)]$. Stack over $i = 1, \dots, d$ to get the vector identity.

**Intuition.** The Gaussian score $\nabla_x \log p = -(x-m)/\sigma^2$ means that "$(x - m) \cdot$" acts as $-\sigma^2$ times a derivative on the density. Integration by parts moves that derivative onto $h$, with a sign flip, producing $+\sigma^2 \nabla h$. Equivalently, let $Z := (x - m)/\sigma \sim \mathcal{N}(0, I_d)$ and $g(z) := h(m + \sigma z)$. Then $\nabla_z g = \sigma \nabla_x h$ and the standard Stein identity $\mathbb{E}[Z g(Z)] = \mathbb{E}[\nabla_z g(Z)]$ gives $\mathbb{E}_p[(x-m) h(x)] = \sigma \mathbb{E}[\nabla_z g] = \sigma^2 \mathbb{E}_p[\nabla_x h(x)]$.

**Tilted (posterior) form.** Let $L: \mathbb{R}^d \to (0, \infty)$ be differentiable with $Z := \int p(x) L(x)\, dx < \infty$, and define $q(x) := \frac{1}{Z} p(x) L(x)$. Substituting $h(x) = L(x)/Z$ into the identity above yields

**First order** (posterior mean):
$$
\mathbb{E}_{x \sim p}[(x - m) h(x)] = \mathbb{E}_{x \sim q}[x - m] = \sigma^2\, \mathbb{E}_{x \sim q}[\nabla_x \log L(x)].
$$

> Proof: first-order tilted Stein identity

Set $h(x) = L(x)/Z$. Then $p(x)\, h(x) = q(x)$, so the LHS becomes $\mathbb{E}_q[x - m]$. For the RHS, $\nabla_x h(x) = \frac{\nabla L(x)}{Z} = \frac{L(x)}{Z} \nabla_x \log L(x)$, so $\mathbb{E}_p[\nabla h(x)] = \mathbb{E}_q[\nabla_x \log L(x)]$.

**Second order** (posterior second moment): substituting $h(x) = (x_i - m_i) L(x)/Z$ into the base identity gives
$$
\mathbb{E}_{x \sim q}\!\left[(x - m)(x - m)^\top\right] = \sigma^2 I_d + \sigma^2\, \mathbb{E}_{x \sim q}\!\left[(x - m)\,(\nabla_x \log L(x))^\top\right].
$$

> Proof: second-order tilted Stein identity

Set $h(x) = (x_i - m_i) L(x)/Z$. LHS: $\mathbb{E}_p[(x_j - m_j)(x_i - m_i) L(x)/Z] = \mathbb{E}_q[(x_j - m_j)(x_i - m_i)]$. RHS: $\sigma^2 \mathbb{E}_p[\partial_{x_j}((x_i - m_i) L(x)/Z)] = \sigma^2 \delta_{ji} + \sigma^2 \mathbb{E}_q[(x_i - m_i)\, \partial_{x_j} \log L(x)]$. Stacking over all $(j, i)$ yields the matrix identity.

The posterior covariance follows by subtracting the outer product of the first-order identity:
$$
\operatorname{Cov}_q(x) = \sigma^2 I_d + \sigma^2\, \operatorname{Cov}_q\!\left(x,\, \nabla_x \log L(x)\right) - \sigma^4\, \mathbb{E}_q[\nabla_x \log L]\, \mathbb{E}_q[\nabla_x \log L]^\top.
$$

**Posterior by Gaussian Stein Identity**

Let prior $\bar{p}(x_{t-1}) = \mathcal{N}(m_{t-1}^{q}, \sigma_{t-1}^2 I)$ and posterior $\bar{q}(x_{t-1}) = \mathcal{N}(\bar{m}_{t-1}^{q}, \bar{\Sigma}_{t-1}^{q})$, consider the posterior should match the Bayesian decomposition of $q(x_{t-1} | w_{t-1} = e_k, x_t, x_0)$

$$
\begin{aligned}
\bar{q}(x_{t-1}) = \frac{1}{\bar{Z}} \bar{p}(x_{t-1}) L(x_{t-1}) 
& = q(x_{t-1} | w_{t-1} = e_k, x_t, x_0) \\
& = \frac{q(w_{t-1} = e_k | x_{t-1}) \cdot q(x_{t-1} | x_t, x_0)}{q(w_{t-1} = w_k | x_t, x_0)} \\
& = \frac{1}{\hat{\pi}_{t-1, k}^{q}} \mathbb{E}_{x_{t-1} \sim \mathcal{N}(m_{t-1}^{q}, \sigma_{t-1}^2 I)} \left[ \operatorname{softmax}_{k}(\frac{E x_{t-1}}{\mathcal{T}})\right] \\
\end{aligned}
$$ 
Therefore, $L(x_{t-1}) = q(w_{t-1} = e_k | x_{t-1}) = \operatorname{softmax}_k(\frac{E x_{t-1}}{\mathcal{T}})$, and $\bar{Z} = q(w_{t-1} = e_k | x_t, x_0) = \hat{\pi}_{t-1, k}^{q}$. Since
$$
\nabla_{x_{t-1}} \log L(x_{t-1}) = \nabla_{x_{t-1}} \log \operatorname{softmax}_k\!\left(\frac{E x_{t-1}}{\mathcal{T}}\right) = \frac{1}{\mathcal{T}} E^\top (e_k - \pi(x_{t-1})),
$$
where $\pi(x) := \operatorname{softmax}(Ex/\mathcal{T})$, applying the first and second-order tilted Stein identities gives:

**Posterior mean** (first-order Stein):
$$
\begin{aligned}
\bar{m}_{t-1}^{q} 
& = \mathbb{E}_{q}\!\left[ x_{t-1} \right] 
= m_{t-1}^{q} + \sigma_{t-1}^{2}\, \mathbb{E}_{q}\!\left[ \nabla_{x_{t-1}} \log L(x_{t-1}) \right] \\
& = m_{t-1}^{q} + \frac{\sigma_{t-1}^{2}}{\mathcal{T}}\, E^\top \left( e_k - \mathbb{E}_{q}[\pi(x_{t-1})] \right).
\end{aligned}
$$
This is exact but implicit: $\mathbb{E}_q[\pi(x_{t-1})]$ depends on $q$, which depends on $L$. In practice, approximate $\mathbb{E}_q[\pi(x_{t-1})] \approx \hat{\pi}_{t-1}^{q}$ (the probit approximation from the Discrete Denoising section), giving
$$
\bar{m}_{t-1}^{q} \approx m_{t-1}^{q} + \frac{\sigma_{t-1}^{2}}{\mathcal{T}}\, E^\top (e_k - \hat{\pi}_{t-1}^{q}).
$$

**Posterior second moment** (second-order Stein): Centered at prior mean $m_{t-1}^{q}$
$$
\begin{aligned}
\mathbb{E}_{q}\!\left[(x_{t-1} - m_{t-1}^{q})(x_{t-1} - m_{t-1}^{q})^\top\right]
& = \sigma_{t-1}^2 I_d + \sigma_{t-1}^2\, \mathbb{E}_{q}\!\left[(x_{t-1} - m_{t-1}^{q})\,(\nabla_{x_{t-1}} \log L)^\top\right] \\
& = \sigma_{t-1}^2 I_d + \frac{\sigma_{t-1}^2}{\mathcal{T}}\, \mathbb{E}_{q}\!\left[(x_{t-1} - m_{t-1}^{q})\,(e_k - \pi(x_{t-1}))^\top\right] E.
\end{aligned}
$$

**Posterior covariance:** subtracting $(\bar{m}_{t-1}^{q} - m_{t-1}^{q})(\bar{m}_{t-1}^{q} - m_{t-1}^{q})^\top$ from the second moment to center at posterior mean $\bar{m}_{t-1}^{q}$,
$$
\begin{aligned}
\bar{\Sigma}_{t-1}^{q}
& = \sigma_{t-1}^2 I_d 
+ \frac{\sigma_{t-1}^2}{\mathcal{T}}\, \operatorname{Cov}_{q}\!\left(x_{t-1},\; (e_k - \pi(x_{t-1}))^\top\right) E
- \frac{\sigma_{t-1}^4}{\mathcal{T}^2}\, E^\top (e_k - \mathbb{E}_q[\pi(x_{t-1})])(e_k - \mathbb{E}_q[\pi(x_{t-1})])^\top E.
\end{aligned}
$$
The first term $\sigma_{t-1}^2 I_d$ is the prior covariance. The second term is the covariance coupling between $x_{t-1}$ and the softmax residual $(e_k - \pi)$ — it captures how observing category $k$ reshapes the spread. The third term subtracts the mean shift squared.

**Small-step simplification.** When $\Delta t$ is small, $\sigma_{t-1}^2 = g_t^2 \Delta t \ll 1$. The second- and third-order terms (both $\propto \sigma_{t-1}^4$ or higher after expanding the covariance) become negligible relative to $\sigma_{t-1}^2 I_d$, so
$$
\bar{\Sigma}_{t-1}^{q} \approx \sigma_{t-1}^2 I_d.
$$
The categorical observation barely changes the covariance at small step sizes — it primarily shifts the mean.

**Learned Posterior by Gaussian Stein Identity**

Similarly, let prior $\bar{p}_{\theta}(x_{t-1}) = \mathcal{N}(m_{t-1}^{\theta}, \sigma_{t-1}^2 I)$ and posterior $\bar{q}_{\theta}(x_{t-1}) = \mathcal{N}(\bar{m}_{t-1}^{\theta}, \bar{\Sigma}_{t-1}^{\theta})$. The learned posterior matches the Bayesian decomposition of $p_{\theta}(x_{t-1} | w_{t-1} = e_k, x_t)$:

$$
\begin{aligned}
\bar{q}_{\theta}(x_{t-1}) = \frac{1}{\bar{Z}_{\theta}} \bar{p}_{\theta}(x_{t-1}) L(x_{t-1}) 
& = p_{\theta}(x_{t-1} | w_{t-1} = e_k, x_t) \\
& = \frac{p_{\theta}(w_{t-1} = e_k | x_{t-1}) \cdot p_{\theta}(x_{t-1} | x_t)}{p_{\theta}(w_{t-1} = e_k | x_t)},
\end{aligned}
$$
with the same likelihood $L(x_{t-1}) = \operatorname{softmax}_k(E x_{t-1}/\mathcal{T})$ and normalizer $\bar{Z}_{\theta} = \hat{\pi}_{t-1,k}^{\theta}$.

The learned posterior mean (first-order Stein):
$$
\begin{aligned}
\bar{m}_{t-1}^{\theta} 
& = \mathbb{E}_{\bar{q}_{\theta}} \left[ x_{t-1} \right] 
= m_{t-1}^{\theta} + \sigma_{t-1}^{2}\, \mathbb{E}_{\bar{q}_{\theta}} \left[ \nabla_{x_{t-1}} \log L(x_{t-1}) \right] \\
& = m_{t-1}^{\theta} + \frac{\sigma_{t-1}^{2}}{\mathcal{T}}\, E^\top \left( e_k - \mathbb{E}_{\bar{q}_{\theta}}[\pi(x_{t-1})] \right).
\end{aligned}
$$
In practice, approximate $\mathbb{E}_{\bar{q}_{\theta}}[\pi(x_{t-1})] \approx \hat{\pi}_{t-1}^{\theta}$ (the probit approximation from the Discrete Denoising section), giving
$$
\bar{m}_{t-1}^{\theta} \approx m_{t-1}^{\theta} + \frac{\sigma_{t-1}^{2}}{\mathcal{T}}\, E^\top (e_k - \hat{\pi}_{t-1}^{\theta}).
$$

The learned posterior second moment (second-order Stein), centered at prior mean $m_{t-1}^{\theta}$:
$$
\begin{aligned}
\mathbb{E}_{\bar{q}_{\theta}} \left[(x_{t-1} - m_{t-1}^{\theta})(x_{t-1} - m_{t-1}^{\theta})^\top\right]
& = \sigma_{t-1}^2 I_d + \sigma_{t-1}^2\, \mathbb{E}_{\bar{q}_{\theta}} \left[(x_{t-1} - m_{t-1}^{\theta}) (\nabla_{x_{t-1}} \log L)^\top\right] \\
& = \sigma_{t-1}^2 I_d + \frac{\sigma_{t-1}^2}{\mathcal{T}}\, \mathbb{E}_{\bar{q}_{\theta}} \left[(x_{t-1} - m_{t-1}^{\theta}) (e_k - \pi(x_{t-1}))^\top\right] E.
\end{aligned}
$$

The learned posterior covariance, subtracting $(\bar{m}_{t-1}^{\theta} - m_{t-1}^{\theta})(\bar{m}_{t-1}^{\theta} - m_{t-1}^{\theta})^\top$ from the second moment:
$$
\begin{aligned}
\bar{\Sigma}_{t-1}^{\theta}
& = \sigma_{t-1}^2 I_d 
+ \frac{\sigma_{t-1}^2}{\mathcal{T}}\, \operatorname{Cov}_{\bar{q}_{\theta}} \left(x_{t-1},\; (e_k - \pi(x_{t-1}))^\top\right) E
- \frac{\sigma_{t-1}^4}{\mathcal{T}^2}\, E^\top (e_k - \mathbb{E}_{\bar{q}_{\theta}}[\pi])(e_k - \mathbb{E}_{\bar{q}_{\theta}}[\pi])^{\top} E.
\end{aligned}
$$
Under the small-step regime ($\sigma_{t-1}^2 = g_t^2 \Delta t \ll 1$), the correction terms are $O(\sigma_{t-1}^4)$, so
$$
\bar{\Sigma}_{t-1}^{\theta} \approx \sigma_{t-1}^2 I_d.
$$

**KL Divergence by Gaussian Stein Identity**

With the Stein-derived Gaussian approximations $\bar{q} \approx \mathcal{N}(\bar{m}_{t-1}^{q},\, \bar{\Sigma}_{t-1}^{q})$ and $\bar{q}_\theta \approx \mathcal{N}(\bar{m}_{t-1}^{\theta},\, \bar{\Sigma}_{t-1}^{\theta})$, the refinement KL becomes a standard Gaussian KL:
$$
\begin{aligned}
L_{\text{Refinement}}
& = \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} D_{\mathrm{KL}}\!\bigl(\bar{q} \;\|\; \bar{q}_\theta\bigr) \\
& = \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} \frac{1}{2}\bigg[
\operatorname{tr}\!\Big((\bar{\Sigma}_{t-1}^{\theta})^{-1} \bar{\Sigma}_{t-1}^{q}\Big) - d + \log \frac{\det \bar{\Sigma}_{t-1}^{\theta}}{\det \bar{\Sigma}_{t-1}^{q}} + \delta_t^\top (\bar{\Sigma}_{t-1}^{\theta})^{-1} \delta_t
\bigg],
\end{aligned}
$$
where $\delta_t := \bar{m}_{t-1}^{\theta} - \bar{m}_{t-1}^{q}$.

**Small-step simplification.** Under $\sigma_{t-1}^2 = g_t^2 \Delta t \ll 1$, both covariances reduce to $\bar{\Sigma}_{t-1}^{q} \approx \bar{\Sigma}_{t-1}^{\theta} \approx \sigma_{t-1}^2 I_d$. The trace and log-det terms cancel, leaving
$$
L_{\text{Refinement}} \approx \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} \frac{\|\delta_t\|^2}{2\sigma_{t-1}^2}.
$$
Substituting the probit-approximated means $\bar{m}_{t-1}^{q} \approx m_{t-1}^{q} + \frac{\sigma_{t-1}^2}{\mathcal{T}} E^\top(e_k - \hat{\pi}_{t-1}^{q})$ and $\bar{m}_{t-1}^{\theta} \approx m_{t-1}^{\theta} + \frac{\sigma_{t-1}^2}{\mathcal{T}} E^\top(e_k - \hat{\pi}_{t-1}^{\theta})$,
$$
\delta_t = \underbrace{(m_{t-1}^{\theta} - m_{t-1}^{q})}_{\text{drift mismatch}} + \underbrace{\frac{\sigma_{t-1}^2}{\mathcal{T}}\, E^\top(\hat{\pi}_{t-1}^{q} - \hat{\pi}_{t-1}^{\theta})}_{\text{discrete feedback}\; O(\Delta t)},
$$
so
$$
L_{\text{Refinement}} \approx \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} \frac{1}{2\sigma_{t-1}^2} \left\| (m_{t-1}^{\theta} - m_{t-1}^{q}) + \frac{\sigma_{t-1}^2}{\mathcal{T}}\, E^\top(\hat{\pi}_{t-1}^{q} - \hat{\pi}_{t-1}^{\theta}) \right\|^2.
$$

##### Method 3: Laplace Approximation Method
###### **Overview**
Consider an unnormalized density $p(x) \propto e^{f(x)}$ with mode $x^*$, so $\nabla f(x^*)=0$. A second-order Taylor expansion around $x^*$ gives
$$
f(x) \approx f(x^*) + \frac{1}{2}(x-x^*)^\top H_f(x^*)(x-x^*),
$$
where $H_f(x^*)$ is the Hessian of $f$ at the mode. Hence, we can derive an approximation for $p(x)$
$$
p(x) \approx \mathcal{N}\!\bigl(x^*,\, (-H_f(x^*))^{-1}\bigr),
$$
provided $H_f(x^*) \prec 0$.
###### **Finding Mode (Local Minima / Maxima)**

Consider
$$
q(x_{t-1}\mid w_{t-1}=e_k, x_t, x_0)
\propto
q(w_{t-1}=e_k\mid x_{t-1})\, q(x_{t-1}\mid x_t, x_0).
$$
Since the normalizing term $q(w_{t-1}=e_k\mid x_t,x_0)$ does not depend on $x_{t-1}$, which has no effect when we derive the Hessian and derivative to $x_{t-1}$,
$$
\begin{aligned}
\nabla_{x_{t-1}} \log q(x_{t-1}\mid w_{t-1}=e_k, x_t, x_0)
&=
\nabla_{x_{t-1}} \log q(w_{t-1}=e_k\mid x_{t-1})
+
\nabla_{x_{t-1}} \log q(x_{t-1}\mid x_t, x_0).
\end{aligned}
$$
Let
$$
\begin{aligned}
\eta(x_{t-1}) := \frac{E x_{t-1}}{\mathcal T},
\qquad
\pi(x_{t-1}) := \operatorname{softmax}(\eta(x_{t-1})).
\end{aligned}
$$
Therefore, the log derivative is
$$
\nabla_{x_{t-1}} \log q(x_{t-1}\mid w_{t-1}=e_k, x_t, x_0)
=
\frac{1}{\mathcal T} E^\top\bigl(e_k - \pi(x_{t-1})\bigr)
- \frac{1}{\sigma_{t-1}^2}(x_{t-1}-m_{t-1}^{q}).
$$

We fit a Gaussian at the mode of the log-posterior. The gradient of $\log(\text{softmax}_k(Ex/\mathcal{T}))$ is $\frac{1}{\mathcal{T}}E^\top(e_k - \pi)$ where $\pi = \operatorname{softmax}(Ex/\mathcal{T})$, so setting the total gradient to zero gives the **mode**:
$$
x^{*,q} = m_{t-1}^{q} + \frac{\sigma_{t-1}^2}{\mathcal{T}}\, E^\top(e_k - \pi^{*,q}), \qquad \pi^{*,q} := \operatorname{softmax}(E x^{*,q}/\mathcal{T}).
$$
This is a fixed-point equation, not a closed-form solution. In practice, one can compute it by fixed-point iteration
$$
x^{(n+1)}
=
m_{t-1}^{q}
E^\top\!\left(e_k-\operatorname{softmax}(E x^{(n)}/\mathcal T)\right),
\qquad
x^{(0)} = m_{t-1}^{q},
$$
  or use the one-step approximation
  $$
  x^{*,q} \approx
  m_{t-1}^{q}
  +
  \frac{\sigma_{t-1}^2}{\mathcal T}
  E^\top\!\left(e_k-\operatorname{softmax}(E m_{t-1}^{q}/\mathcal T)\right).
  $$
> Proof: $x^{*,q} = m_{t-1}^{q} + \frac{\sigma_{t-1}^2}{\mathcal{T}}\, E^\top(e_k - \pi^{*,q})$

Consider the log derivative to $x_{t-1}$ is
$$
\begin{aligned}
\nabla_{x_{t-1}} \log q(x_{t-1}\mid w_{t-1}=e_k, x_t, x_0)
&=
\nabla_{x_{t-1}} \log q(w_{t-1}=e_k\mid x_{t-1})
+
\nabla_{x_{t-1}} \log q(x_{t-1}\mid x_t, x_0).
\end{aligned}
$$
The first term can be expanded as
$$
\log q(w_{t-1}=e_k\mid x_{t-1})
=
\log \pi_k(x_{t-1})
=
e_k^\top \eta(x_{t-1}) - \log \sum_{i=1}^{K} e^{\eta_i(x_{t-1})},
$$
so the log derivative
$$
\nabla_{x_{t-1}} \log q(w_{t-1}=e_k\mid x_{t-1})
=
\frac{1}{\mathcal T} E^\top\bigl(e_k - \pi(x_{t-1})\bigr).
$$
For the second term
$$
q(x_{t-1}\mid x_t, x_0) = \mathcal N(x_{t-1}; m_{t-1}^{q}, \sigma_{t-1}^2 I_d),
$$
we have log derivative
$$
\nabla_{x_{t-1}} \log q(x_{t-1}\mid x_t, x_0)
=
-\frac{1}{\sigma_{t-1}^2}(x_{t-1}-m_{t-1}^{q}).
$$
Therefore, the full log derivative is
$$
\nabla_{x_{t-1}} \log q(x_{t-1}\mid w_{t-1}=e_k, x_t, x_0)
=
\frac{1}{\mathcal T} E^\top\bigl(e_k - \pi(x_{t-1})\bigr)
- \frac{1}{\sigma_{t-1}^2}(x_{t-1}-m_{t-1}^{q}).
$$
###### **Finding Precision (Negative Hessian)**

The **precision** (negative Hessian) at the mode is
$$
\Lambda^q = \frac{1}{\sigma_{t-1}^2}\, I_d + \frac{1}{\mathcal{T}^2}\, E^\top F(\pi^{*,q})\, E,
$$
where $F(\pi) := \operatorname{diag}(\pi) - \pi\pi^\top \succeq 0$ is the Fisher information of the categorical distribution. The first term is the Gaussian precision; the second is the curvature added by the softmax observation. Together they give
$$
q(x_{t-1}\mid e_k, x_t, x_0) \approx \mathcal{N}\!\big(x^{*,q},\; (\Lambda^q)^{-1}\big).
$$
The same applies to the reverse process with $m_{t-1}^{\theta} := x_t + f(\tau(t), x_t, \hat{y}_\theta)\,\Delta t$:
$$
p_\theta(x_{t-1}\mid e_k, x_t) \approx \mathcal{N}\!\big(x^{*,\theta},\; (\Lambda^\theta)^{-1}\big),
$$
$$
x^{*,\theta} = m_{t-1}^{\theta} + \frac{\sigma_{t-1}^2}{\mathcal{T}}\, E^\top(e_k - \pi^{*,\theta}), \qquad
\Lambda^\theta = \frac{1}{\sigma_{t-1}^2}\, I_d + \frac{1}{\mathcal{T}^2}\, E^\top F(\pi^{*,\theta})\, E.
$$

###### Result: Gaussian KL

Both sides are now Gaussian, so
$$
L_{\text{Refinement}}
\approx \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} \frac{1}{2}\bigg[
\operatorname{tr}\!\Big(\Lambda^\theta (\Lambda^q)^{-1}\Big) - d + \log \frac{\det \Lambda^q}{\det \Lambda^\theta} + \delta_t^\top \Lambda^\theta\, \delta_t
\bigg],
$$
where $\delta_t := x^{*,\theta} - x^{*,q}$.

###### **Small-step simplification**

When $\Delta t$ is small, $\sigma_{t-1}^2 = g_t^2 \Delta t$ is small, so the Gaussian precision $\frac{1}{\sigma_{t-1}^2} I_d$ dominates the softmax curvature in both $\Lambda^q$ and $\Lambda^\theta$. The precisions become approximately equal, the trace and log-det terms cancel, and we get
$$
L_{\text{Refinement}} \approx \sum_{t=2}^T \mathbb{E}_{q(w_{t-1}, x_t, x_0 \mid w_0)} \frac{\left\|x^{*,\theta} - x^{*,q}\right\|^2}{2\sigma_{t-1}^2}.
$$
To keep it simple, if we use one-step approximation for the fixed-point equation, it gives
$$
x^{*,\theta} - x^{*,q} \approx \underbrace{(m_{t-1}^{\theta} - m_{t-1}^{q})}_{\text{drift mismatch}} + \underbrace{\frac{\sigma_{t-1}^2}{\mathcal{T}}\, E^\top(\pi_0^{q} - \pi_0^{\theta})}_{\text{discrete feedback}},
$$
where $\pi_0^{q} := \operatorname{softmax}(E m_{t-1}^{q}/\mathcal{T})$ and $\pi_0^{\theta} := \operatorname{softmax}(E m_{t-1}^{\theta}/\mathcal{T})$. The first term penalizes the drift mismatch between forward and reverse; the second is a correction from the induced discrepancy in softmax probabilities.
#### Reconstruction $:L_{Reconstruction}$
The reconstruction term calculate the reverse transition at $t=1$:
$$
p_\theta(w_0 \mid x_1)
= \int p_\theta(w_0 \mid x_0)\, p_\theta(x_0 \mid x_1)\, d x_0.
$$
From the Euler-Gaussian approximation, $p_\theta(x_0 \mid x_1) = \mathcal{N}(x_0;\, m_0^{\theta},\, \sigma_0^2 I_d)$ with
$$
m_0^{\theta} := x_1 + f(\tau(1), x_1, \hat{y}_\theta)\,\Delta t, \qquad \sigma_0^2 := g(\tau(1), x_1)^2\,\Delta t.
$$
Therefore, if we expand the $p_\theta(w_0 \mid x_1)$ with diagonal probit approximation
$$
\begin{aligned}
p_\theta(w_0 \mid x_1)
& = \int p_\theta(w_0 \mid x_0) p_\theta(x_0 \mid x_1) d x_0 \\
& = \int \operatorname{Cat}(w_0; \operatorname{softmax}(\frac{E x_0}{\mathcal{T}})) \cdot \mathcal{N}(x_0;\, m_0^{\theta},\, \sigma_0^2 I_d) d x_0 \\
& \approx \operatorname{Cat} \left( w_0; \operatorname{softmax} \left(
\mu_0^{\theta} \oslash \sqrt{ 1 + \tfrac{\pi}{8}\, \operatorname{diag}(\Sigma_0) }
\right) \right)
\end{aligned}
$$
where
$$
\mu_0^{\theta} := \frac{E x_1}{\mathcal{T}} + \frac{E\, f(\tau(1), x_1, \hat{y}_\theta)}{\mathcal{T}}\,\Delta t, \qquad \Sigma_0 := \frac{g(\tau(1), x_1)^2}{\mathcal{T}^2}\, E E^\top\, \Delta t.
$$
Consider the $p_{\theta}(w_0 | x_1)$ is categorical distribution, the reconstruction term is then
$$
L_{\text{Reconstruction}}
\approx
\mathbb E_{q(x_1\mid w_0)}
\left[
- \log \sum_{i=1}^{K} [w_0]_i \operatorname{softmax}_i \left(
\mu_0^{\theta} \oslash \sqrt{ 1 + \tfrac{\pi}{8}\, \operatorname{diag}(\Sigma_0) }
\right) 
\right].
$$
If $w_0$ is a one-hot vector $w_0 = e_k$ at class $k$, the summation over $K$ classes can be reduced to
$$
L_{\text{Reconstruction}}
\approx
\mathbb E_{q(x_1\mid w_0)}
\left[
- \log [w_0]_k \operatorname{softmax}_k \left(
\mu_0^{\theta} \oslash \sqrt{ 1 + \tfrac{\pi}{8}\, \operatorname{diag}(\Sigma_0) }
\right) 
\right].
$$
#### Initial $:L_{Initial}$
$$
\begin{aligned}
L_{\text{Initial}} 
& = \mathbb{E}_{q(x_1 | w_0)} D_{\mathrm{KL}} \bigl(q(x_0 | w_0) || p_\theta(x_0\mid w_0,x_1) \bigr) \\
& = \mathbb{E}_{q(x_1 | w_0)} \left[ D_{\mathrm{KL}} \bigl(q(x_0 | w_0, x_1, x_0) || p_\theta(x_0\mid w_0,x_1) \bigr) - \log q(x_0 | w_0, x_1, x_0) + \log q(x_0 | w_0) \right] \\
\end{aligned}
$$
Since $- \log q(x_0 | w_0, x_1, x_0) + \log q(x_0 | w_0)$ is a constant, not dependent to the model $\theta$, therefore, the gradient $\nabla_{\theta} L_{\text{Initial}}$ is identical to
$$
\nabla_{\theta} L_{\text{Initial}} = \nabla_{\theta} \mathbb{E}_{q(x_1 | w_0)} D_{\mathrm{KL}} \bigl(q(x_0 | w_0, x_1, x_0) || p_\theta(x_0\mid w_0,x_1) \bigr)
$$
Thus, the $L_{\text{Initial}}$ can be computed by $L_{Refinement}$ with $t=1$
### Training Loss
When the model output is used as a boundary point in the bridge drift, define
$$
\hat{y}_{\theta,i}
:=
\frac{f_\theta(x_{\tau(i)}, \tau(i))}{\|f_\theta(x_{\tau(i)}, \tau(i))\|}.
$$
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
\frac{\hat{y}_{\theta,i}-x_{\tau(i)}}{\|\hat{y}_{\theta,i}-x_{\tau(i)}\|^2}
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
- \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|\hat{y}_{\theta,1} - x_{\tau(1)}\|^2}
\big(\hat{y}_{\theta,1} - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta t \Bigg\|^2
\right]
$$
### Proposal Generator
- Choose a proposal schedule generator $\rho$ from
	- $\text{Exp}(\lambda)$
	- $\text{Unif}(\alpha, \beta)$
	- A fixed interval $[\alpha, \beta]$ with fixed step size $\frac{\beta - \alpha}{T}$
- if $\rho$ is a distribution:
	- Generate $\{\tau(i)\}_{i=1}^{T} = \rho^{-1}(\{\frac{i}{T}\}_{i=1}^{T})$ from uniform grid $\{\frac{i}{T}\}_{i=1}^{T}$ and inverse CDF $\rho^{-1}$
- else if $\rho$ is a fixed interval:
	- Set $\{\tau(i)\}_{i=1}^{T} = \rho$
- Permute $\{\tau(i)\}_{i=1}^{T}$ in ascending order, set $\tau(0)=0$, and define $\Delta \tau_i = \tau(i) - \tau(i-1)$
- Return $\{\tau(i)\}_{i=1}^{T}$ and $\{\Delta \tau_i\}_{i=1}^{T}$
### Training Algorithm
We choose $T = 1000$ and $r = 1.0$
Given training dataset $Y$ in word embedding format, a word embedding $E \in \mathbb{R}^{K \times d}$, and a proposal schedule generator $\rho$:
For each data point $y \in Y, y \in \mathbb{R}^{d}$ drawn from the training dataset $Y$,
- Normalize $y = \frac{y}{|| y ||} r$
- Set discrete data point $y' = \arg \max (y E^{\top})$
- Generate $\{\tau(i)\}_{i=1}^{T}$ and $\{\Delta \tau_i\}_{i=1}^{T}$ from $\rho$ using algorithm Proposal Generator
- Simulate bridge states $\{x_{\tau(i)}\}_{i=T}^{1}$ from $q(\cdot \mid y)$, $T \to 1$. One Euler-Maruyama step is
$$
x_{\tau(i-1)} =
x_{\tau(i)}
+ f(\tau(i), x_{\tau(i)}, y) \Delta \tau_i
+ g(\tau(i), x_{\tau(i)}) \sqrt{\Delta \tau_i}\,\varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N(0, I).
$$
- Normalize $x_{\tau(i)} = \frac{x_{\tau(i)}}{|| x_{\tau(i)} ||} r$ if $|| x_{\tau(i)} || > r$
- When the model output is used as a boundary point in the bridge drift, normalize it as
$$
\hat{y}_{\theta,i}
:=
\frac{f_\theta(x_{\tau(i)}, \tau(i))}{\|f_\theta(x_{\tau(i)}, \tau(i))\|} r.
$$
- If using a surrogate objective, sample $i \in \{2, \ldots, T\}$ and compute one of the following:
	- Cross Entropy
$$
\mathcal{L}_{\mathrm{CE}}(Y, \theta)
=
\sum_{i = 2}^{T}
\left[
\mathrm{CE}(y \mathbf{E}^{\top}, \mathrm{softmax}(f_\theta(x_{\tau(i)}, \tau(i)) \mathbf{E}^{\top}))
\right]
$$
	- L2 Norm
$$
\mathcal{L}_{\mathrm{L2}}(Y, \theta)
=
\sum_{i = 2}^{T}
\left[
\left\| y - f_\theta(x_{\tau(i)}, \tau(i)) \right\|_2^2
\right]
$$
	- Diffusion
$$
\mathcal{L}_{\mathrm{diff}}(Y, \theta)
=
\sum_{i = 2}^{T}
\left[
\frac{\Delta \tau_i}{2} (d-1)^2 (1-\|x_{\tau(i)}\|^2)^2
\left\|
\frac{y-x_{\tau(i)}}{\|y-x_{\tau(i)}\|^2}
-
\frac{\hat{y}_{\theta,i}-x_{\tau(i)}}{\|\hat{y}_{\theta,i}-x_{\tau(i)}\|^2}
\right\|_2^2
\right]
$$
    - NELBO
$$
\mathcal{L}_{\mathrm{NELBO}}(Y, \theta) = \mathcal{L}_{\mathrm{diff}}(Y, \theta) + \mathcal{L}_{0}(Y, \theta)
$$
		where 
$$
\mathcal{L}_{0}(Y, \theta)
=
\left[
\frac{d}{2} \log 2 \pi 
+ d \log \left( \frac{1-\|x_{\tau(1)}\|^2}{2} \right)
+ \frac{d}{2} \log \Delta \tau_1
+ \frac{2}{(1-\|x_{\tau(1)}\|^2)^2 \Delta \tau_1}
\Bigg\| y - x_{\tau(1)}
- \Bigg(
\frac{d-1}{2}
\frac{(1-\|x_{\tau(1)}\|^2)^2}{\|\hat{y}_{\theta,1} - x_{\tau(1)}\|^2}
\big(\hat{y}_{\theta,1} - x_{\tau(1)}\big)
- \frac{d}{4} (1-\|x_{\tau(1)}\|^2) x_{\tau(1)}
\Bigg) \Delta \tau_1 \Bigg\|^2
\right]
$$
- Update the model weight with $\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}_{*}(Y, \theta)$, where $\mathcal{L}_{*}$ is either a surrogate objective or $\mathcal{L}_{\mathrm{NELBO}}(Y, \theta)$.

### Inference Algorithm
- Choose $T = 1000$ and $\Delta \tau(i) = 0.03$
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
+ \left(
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
---
Given a $d$-dimensional SDE as $d z_t = dw$ on Poincare Disk $\mathbb{D}^d$ and conditioned a point $x$ at the boundary of Poincare disk , the Brownian bridge can be written as
$$
dz_t = \underbrace{\frac{d-1}{2} \frac{(1-\|z_t\|^2)^2}{\|x-z_t\|^2} (x-z_t) dt}_{\text{Bridge Drift}} - \underbrace{\frac{d}{4} (1-\|z_t\|^2) z_t dt}_{\text{Itô Correction}} + \underbrace{\frac{1-\|z_t\|^2}{2} dW_t}_{\text{Hyperbolic Noise}}
$$
# Part III: Optimal Control
# Part IV: Schrodinger Bridge
## Related Theory Derivation
### Ito's Lemma Derivation
#### Scaler
##### Formal Mathematical Definition of Itô's Lemma

Let $x_t$ be an Itô drift-diffusion process that satisfies the stochastic differential equation (SDE):
$$dx_t = \mu(t, x_t) dt + \sigma(t, x_t) dw_t$$
where $w_t$ is a standard Wiener process (Brownian motion).

If $g(t, x)$ is a scalar function that is twice continuously differentiable in $x$ and once continuously differentiable in $t$ (i.e., $g \in C^{1,2}$), then Itô's Lemma states that $y_t = g(t, x_t)$ is also an Itô process, and its differential is given by:
$$dg(t, x_t) = \left( \frac{\partial g}{\partial t} + \mu(t, x_t) \frac{\partial g}{\partial x} + \frac{1}{2} \sigma(t, x_t)^2 \frac{\partial^2 g}{\partial x^2} \right) dt + \sigma(t, x_t) \frac{\partial g}{\partial x} dw_t$$
##### Derivation

#### Multivariate

### Moore Pseudoinverse
> Pseudoinverse gives minimum norm solution for a least squared problem

### Logistic Normal
#### The KL Divergence between Logistic Normal

### Given

Two Gaussians on $\mathbb{R}^K$:
$$ p = \mathcal{N}(\mu_p,,\Sigma_p), \qquad q = \mathcal{N}(\mu_q,,\Sigma_q), $$
and their temperature-softmax pushforwards onto $\operatorname{int}(\Delta^{K-1})$:
$$P := \mathcal{S}_{\mathcal{T}\#},p, \qquad Q := \mathcal{S}_{\mathcal{T}\#},q.$$
We seek $D_{\mathrm{KL}}(P | Q)$.

---
### Step 1 — ALR diffeomorphism

The map $S_\mathcal{T}(z) = \operatorname{softmax}(z/\mathcal{T})$ is **not injective** — it is invariant under $z \mapsto z + c\mathbf{1}$. To work on the simplex properly, introduce the **additive log-ratio (ALR)** diffeomorphism $\phi : \operatorname{int}(\Delta^{K-1}) \to \mathbb{R}^{K-1}$:
$$ \phi(w)_j = \log\frac{w_j}{w_K}, \qquad j = 1,\dots,K{-}1. $$
Its inverse is
$$ \phi^{-1}(\eta)_j = \frac{e^{\eta_j}}{1 + \sum_{k=1}^{K-1} e^{\eta_k}}, \qquad \phi^{-1}(\eta)_K = \frac{1}{1 + \sum_{k=1}^{K-1} e^{\eta_k}}. $$
Since $\phi$ is a diffeomorphism, the Jacobian of $\phi^{-1}$ with respect to $\eta$ is
$$ \left|\det \frac{\partial w}{\partial \eta}\right| = \prod_{j=1}^{K} w_j. $$
---
### Step 2 — Push $P$ and $Q$ to ALR coordinates
Given $z \sim \mathcal{N}(\mu_p, \Sigma_p)$, the ALR coordinate of $w = S_\mathcal{T}(z)$ is
$$ \eta_j = \log\frac{w_j}{w_K} = \frac{z_j - z_K}{\mathcal{T}}. $$
In matrix form, define the **contrast matrix** $F \in \mathbb{R}^{(K-1)\times K}$, with $(F)_{jk} = \delta_{jk} - \delta_{kK}$. Then
$$ \eta = \frac{1}{\mathcal{T}},F z. $$
Since $\eta$ is a linear function of the Gaussian $z$, we get
$$ \phi_\#,P = \mathcal{N}!\left(\frac{F\mu_p}{\mathcal{T}},;\frac{F\Sigma_p F^\top}{\mathcal{T}^2}\right), \qquad \phi_\#,Q = \mathcal{N}!\left(\frac{F\mu_q}{\mathcal{T}},;\frac{F\Sigma_q F^\top}{\mathcal{T}^2}\right). $$

---

### Step 3 — KL is invariant under diffeomorphisms

Since $\phi$ is a diffeomorphism (a bijective smooth map with smooth inverse), KL divergence is preserved:

$$ D_{\mathrm{KL}}(P | Q) = D_{\mathrm{KL}}(\phi_\# P | \phi_\# Q). $$

This holds because the density ratio transforms as

$$ \frac{dP}{dQ}(w) = \frac{p_P(w)}{p_Q(w)} = \frac{p_{\phi_\# P}(\eta)\ \cancel{|\det \partial\eta/\partial w|}}{p_{\phi_\# Q}(\eta)\ \cancel{|\det \partial\eta/\partial w|}} = \frac{p_{\phi_\# P}(\eta)}{p_{\phi_\# Q}(\eta)}, $$

and the integration measure transforms covariantly, so the integral is identical.

---

### Step 4 — Standard Gaussian KL

Define the ALR-space parameters:

$$ \tilde\mu_p = \frac{F\mu_p}{\mathcal{T}}, \quad \tilde\mu_q = \frac{F\mu_q}{\mathcal{T}}, \quad \tilde\Sigma_p = \frac{F\Sigma_p F^\top}{\mathcal{T}^2}, \quad \tilde\Sigma_q = \frac{F\Sigma_q F^\top}{\mathcal{T}^2}. $$

The standard formula gives:

$$ \boxed{D_{\mathrm{KL}}!\bigl(\mathcal{S}_{\mathcal{T}\#} p ;\big|; \mathcal{S}_{\mathcal{T}\#} q\bigr) = \frac{1}{2}\left[\operatorname{tr}!\left(\tilde\Sigma_q^{-1}\tilde\Sigma_p\right) - (K{-}1) + \log\frac{\det\tilde\Sigma_q}{\det\tilde\Sigma_p} + (\tilde\mu_q - \tilde\mu_p)^\top \tilde\Sigma_q^{-1}(\tilde\mu_q - \tilde\mu_p)\right]} $$

---

### Step 5 — Temperature cancellation

Substituting back, observe that $\tilde\Sigma_q^{-1}\tilde\Sigma_p = (F\Sigma_q F^\top)^{-1}(F\Sigma_p F^\top)$, and $\det\tilde\Sigma_q / \det\tilde\Sigma_p = \det(F\Sigma_q F^\top)/\det(F\Sigma_p F^\top)$, and

$$ (\tilde\mu_q - \tilde\mu_p)^\top \tilde\Sigma_q^{-1}(\tilde\mu_q - \tilde\mu_p) = \frac{1}{\mathcal{T}^2}(F\delta\mu)^\top \cdot \mathcal{T}^2(F\Sigma_q F^\top)^{-1}\cdot \frac{1}{\mathcal{T}}(F\delta\mu) \cdot \mathcal{T} $$

More cleanly: every $\mathcal{T}^2$ in the numerator covariance cancels with the $\mathcal{T}^2$ in the denominator covariance, and the $1/\mathcal{T}$ in the means cancels with the $\mathcal{T}^2$ from $\tilde\Sigma_q^{-1}$. Explicitly:

$$ D_{\mathrm{KL}} = \frac{1}{2}\Big[\operatorname{tr}!\big((F\Sigma_q F^\top)^{-1}F\Sigma_p F^\top\big) - (K{-}1) + \log\frac{\det(F\Sigma_q F^\top)}{\det(F\Sigma_p F^\top)} + (F\delta\mu)^\top(F\Sigma_q F^\top)^{-1}(F\delta\mu)\Big] $$

where $\delta\mu = \mu_q - \mu_p$.

> **The temperature $\mathcal{T}$ drops out entirely.** The KL between two logistic-normals induced by $\mathcal{S}_{\mathcal{T}\#}$ is independent of $\mathcal{T}$, equaling the KL between the underlying Gaussians projected through $F$ into ALR coordinates.

---

### Special case: $\Sigma_p = \Sigma_q = \Sigma$

The trace and log-det terms cancel, leaving

$$ D_{\mathrm{KL}} = \frac{1}{2},(F\delta\mu)^\top,(F\Sigma F^\top)^{-1},(F\delta\mu). $$

This is the squared Mahalanobis distance of the mean difference projected into the $(K{-}1)$-dimensional ALR subspace — independent of temperature.

### Fokker Plank Equation Derivation
The Fokker Plank Equation (FPE) states given a general SDE $dx_{t} = F(x, t) dt + \phi(x, t) d W_{t}$ with $d$ dimensional state $x, x_{t} \in \mathbb{R}^d$, drift $F: \mathbb{R}^{d} \times \mathbb{R} \to \mathbb{R}^{d}$, and diffusion matrix $G: \mathbb{R}^{d} \times \mathbb{R} \to \mathbb{R}^{d \times d}$. The diffusion term is an isotropic Wienner process $d W_t \in \mathbb{R}^{w}$ governed by $g(x, t) = \phi(x, t) \phi(x, t)^{\top}$ with $\phi: \mathbb{R}^{d} \times \mathbb{R} \to \mathbb{R}^{d \times  w}$, the Fokker Plank equation describes the marginal probability $p_t(x)$ as
$$
\begin{aligned}
\frac{\partial p_t(x)}{\partial t} 
& = - \nabla_x \cdot \left[ F(x, t) p_t(x) \right] + \frac{1}{2} \nabla_x^2 \left[ G(x, t) p_t(x) \right] \\
& = - \sum_{i=1}^{d} \frac{\partial}{\partial x_i} \left[ F_i(x, t) p_t(x) \right] + \frac{1}{2} \sum_{i=1}^{d} \sum_{j=1}^{d} \frac{\partial}{\partial x_i} \frac{\partial}{\partial x_j} \left[ G_{ij}(x, t) p_t(x) \right] \\
\end{aligned}
$$
where $\nabla_x \cdot$ is divergence and $\nabla^2_{x}$ is Laplacian operator. 

The intuition of deriving Fokker Plank equation is leveraging the Ito's lemma to calculate the evolution of the probability density.

### Kolmogorov Backward Equation Derivation
The Kolmogorov Backward equation (KBE) states given a general reversed-time SDE $dx_{t} = F(x, t) dt + \phi(x, t) d \bar{W}_{t}$ with $d$ dimensional state $x, x_{t} \in \mathbb{R}^d$, drift $F: \mathbb{R}^{d} \times \mathbb{R} \to \mathbb{R}^{d}$, and diffusion matrix $G: \mathbb{R}^{d} \times \mathbb{R} \to \mathbb{R}^{d \times d}$. The diffusion term is a reversed isotropic Wienner process $d \bar{W}_t \in \mathbb{R}^{w}$ governed by $g(x, t) = \phi(x, t) \phi(x, t)^{\top}$ with $\phi: \mathbb{R}^{d} \times \mathbb{R} \to \mathbb{R}^{d \times  w}$, the Kolmogorov Backward equation describes the reversed marginal probability $p_t(x)$ as
$$
\begin{aligned}
\frac{\partial p_t(x)}{\partial t} 
& = - \nabla_x \cdot \left[ F(x, t) p_t(x) \right] - \frac{1}{2} \nabla_x^2 \left[ G(x, t) p_t(x) \right] \\
& = - \sum_{i=1}^{d} \frac{\partial}{\partial x_i} \left[ F_i(x, t) p_t(x) \right] - \frac{1}{2} \sum_{i=1}^{d} \sum_{j=1}^{d} \frac{\partial}{\partial x_i} \frac{\partial}{\partial x_j} \left[ G_{ij}(x, t) p_t(x) \right] \\
\end{aligned}
$$
where $\nabla_x \cdot$ is divergence and $\nabla^2_{x}$ is Laplacian operator. 
### Feynman–Kac Formula
