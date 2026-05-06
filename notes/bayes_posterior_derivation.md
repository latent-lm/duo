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
$$

where:

- $q_t(x_t)=P_{\mathbb H}(d_{\mathbb H}(x_t,0);t)$ is the free-BM marginal — depends on $\|x_t\|$ and $t$, **not on $y$**;
- $q_\infty(y)$ is the boundary exit measure — by rotational symmetry it is **uniform on the sphere**, so it does not depend on $y$;
- $q_{\infty\mid t}(y\mid x_t)$ is the Poisson kernel,

$$
q_{\infty\mid t}(y\mid x_t)=e^{(d-1) \langle y,x_t\rangle_{\mathbb H}}=\left(\dfrac{1-\|x_t\|^2}{\|y-x_t\|^2}\right)^{d-1}.
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
