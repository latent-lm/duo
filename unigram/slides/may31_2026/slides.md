---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
size: 4:3
class: lead
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
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# Hyperbolic DLM

#### May 31, 2026

---

### Does a bigger test set tighten the ELBO to the entropy?

---

## The Question

> As the test set grows $4\times10^4 \to 4\times10^8$, does
> <b>`test_wnelbo`</b> converge to the data entropy?
> And does that convergence hold **across $\lambda$**?

For an **optimal model** the IS-NELBO is an *unbiased* estimator of the entropy — so in principle "more data ⇒ tighter." 

---

## Experiment Set Up

- Driver: `unigram/unigram_test2_tmp3.py` · script: `unigram_test_lorentz_tmp3.sh` · **seed = 42**
- Model: **optimal** (`mode=opt`) — emits the ground-truth unigram, no training
- Vocab 10 · Data dist $[0.91,\ 0.01\times 9]$ · entropy $H \approx$ **0.5003**
- Geometry: **Poincaré–Polar** · bridge horizon `hyper_T=1000`, `hyper_dt=0.01`
- Proposal: **plain exponential** $p(t)=\lambda e^{-\lambda t}$, rate $\lambda \in \{0.01,0.1,0.2,0.3,0.5,0.8,1.0,2.0\}$
- **test-set size:** $\{4\mathrm{e}4,\ 4\mathrm{e}5,\ 4\mathrm{e}6,\ 4\mathrm{e}7,\ 4\mathrm{e}8\}$

---

## Two Metrics, Two Roles

| Metric | Meaning | Role |
|---|---|---|
| `test_ce` | empirical cross-entropy of the optimal logits | **the target** $=H$ |
| `test_wnelbo` | importance-weighted Poincaré bridge ELBO | **the estimator** of $H$ |


<!-- ---

## `test_ce` is the Fixed Target — It Doesn't Move

`test_ce` $=$ <b>0.5003</b> for **every** $\lambda$ and **every** test size (drifts only $0.50029 \to 0.50031$ from 4e4 to 4e8 — pure sampling noise on the empirical entropy).

| metric | 4e4 | 4e5 | 4e6 | 4e7 | 4e8 |
|:--|:--:|:--:|:--:|:--:|:--:|
| `test_ce` (any $\lambda$) | 0.50029 | 0.50029 | 0.50029 | 0.50030 | 0.50031 |

It is independent of the timestep proposal — so it serves as the **ground-truth line** every `test_wnelbo` should approach. -->

---

## ELBO under various Testing Dataset Size

![w:1000 center](wnelbo_vs_ts.jpg)

**More test data tightens to the entropy <r>only</r> when $\lambda$ is matched.**

---

## But the variance might also affect

![w:1000 center](wnelbo_std_vs_ts.jpg)

Only $\lambda=0.1,0.01$ mainten flat std, but others **grow** with $N$.

---

## `test_wnelbo` Across Test Size and $\lambda$

| $\lambda$ | 4e4 | 4e5 | 4e6 | 4e7 | 4e8 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.01 | 0.5020 | 0.4449 | 0.4638 | 0.4611 | <r>0.4612</r> |
| **0.1**  | 0.4681 | 0.4874 | 0.4901 | 0.4905 | 0.4905 |
| **0.2**  | 0.4777 | 0.4946 | 0.4986 | **0.4999** | **0.4998** 
| **0.3**  | 0.4670 | 0.4921 | 0.4974 | 0.5055 | 0.5066 |
| 0.5  | 0.4919 | 0.4909 | 0.5049 | 0.5237 | 0.5195 |
| 0.8  | 0.4861 | 0.6099 | 0.5253 | 0.5361 | 0.5293 |
| 1.0  | 0.5018 | 0.5993 | 0.5271 | 0.5267 | 0.5264 |
| 2.0  | 0.3874 | 0.4608 | 0.4433 | 0.4581 | <r>0.4564</r> |

---

## The ELBO Estimation is Heavily Skew

Fraction of the **whole** weighted-NELBO sum coming from the very top samples (4M test set):

| $\lambda$ | top $10^{-5}$ | top $10^{-4}$ | top $10^{-3}$ | max weight |
|:--:|:--:|:--:|:--:|:--:|
| 0.1 | 0.5% | 2.3% | 10% | 713 |
| 0.2 | 1.3% | 4.2% | 13% | 2,836 |
| <r>1.0</r> | <r>**18%**</r> | <r>31%</r> | 44% | <r>28,261</r> |
| <r>2.0</r> | <r>**18%**</r> | <r>40%</r> | 57% | <r>14,294</r> |

At $\lambda{=}1.0$, **400 samples out of 4,000,000 (0.01%) carry ~31%** of the estimate

---

## The ELBO Estimation is Heavily Skew

![](wnelbo_lorenz.jpg)

---

## Three Regimes — Test-Size View

| Regime | $\lambda$ | More data does… | At 4e8 |
|---|:--:|---|:--:|
| **Too diffuse** | 0.01 | nothing — samples land past horizon $t{=}10$ | <r>0.461</r> |
| **Matched** | **0.1 – 0.3** | <b>tightens to $H$</b>, std flat/mild | **0.490 – 0.507** |
| **Heavy-tailed** | 0.5 – 1.0 | shrinks SEM, not the gap | 0.52 – 0.53 |
| **Tail-starved** | 2.0 | misses the tail entirely | <r>0.456</r> |

---

## Conclusion

- ELBO point estimate is heavily skew, few samples dominate the IS weighted average

---

# Learnable Word Embedding & Cross Entropy Loss

---

## Cross Entropy Loss

$$
\begin{aligned}
\mathcal{L}_{CE}(\theta)
 & =
\mathbb{E}_{z_{t} \sim q_{t \mid \infty}(\cdot \mid x), t \sim \text{Unif}([0, \infty]), y \sim q_{data}}
\left[
	- \sum_{i=1}^{L} \log p_{\infty | t}^{\theta}(x^{i} | z_{t})
\right] \\
& = \mathbb{E}_{z_{t} \sim q_{t \mid \infty}(\cdot \mid x), t \sim \text{Unif}([0, \infty]), y \sim q_{data}}
\left[ 
	- \sum_{i=1}^{L} \log \mu^{\theta, i}(z_t) 
\right]
\end{aligned}
$$

where the normalized embedding at $i$-th position is $x^{i} = \frac{e^{y^i}}{|| e^{y^i} ||_2}$ and the posterior $q_{t | \infty}(z | x)$, $y^i$ is the token index at position $i$.

---

## Two Model Parametrization

- Horocycle

$$
\mu^{\theta}(z_t) 
:= \operatorname{softmax}(f_{\theta}(z_t) + (d-1) \sum_{v \in V} e_v \langle v, z_t \rangle_{\mathbb{H}})
$$

where $\langle v, z_t \rangle_{\mathbb{H}}$ is the Busemann inner product

- Direct

$$
\mu^{\theta}(z_t) 
:= \operatorname{softmax}(f_{\theta}(z_t))
$$

---

## Learnable Embedding Normalization (Same as Hyperspherical FM)

![alt text](image.png)

---

## Experiments

2 Model Parametrization (Horocycle / Direct) 

$\times$ Learnable / Fixed Word Embedding

---

## Experiment Set Up

- Driver `unigram/unigram_test2_tmp4.py` · script `unigram_test_lorentz_tmp4_loss_emb.sh` · `mode=tnb` (trains) · **seed 42** · 4M test
- Vocab 10 · data $[0.91,\ 0.01\times 9]$ · entropy $H \approx$ **0.5003**
- **Parametrization:** Horocycle $=$ `horo_cross_entropy` $\big(\mu=\mathrm{softmax}(f(z)+\text{horosphere})\big)$ vs Direct $=$ `cross_entropy` $\big(\mu=\mathrm{softmax}(f(z))\big)$
- **Word embedding:** Learnable $\big(\phi_v=\mathrm{atan2}(e_v)$, `trainable_word_embedding=True`$\big)$ vs Fixed $\big($equally spaced $(v{+}0.5)\,2\pi/V\big)$
- **Proposal swept, ELBO eval $=$ same proposal as loss:** `unif` $\times$ `hyper_T`$\in\{1000,2000,3000,5000\}$ and `stratified_exp` $\times\ \lambda\in\{1.0,0.1,0.2,0.3,0.8,0.5\}$

---

## Results — the 2×2 at a glance

| | **Learnable emb** | **Fixed emb** |
|---|---|---|
| **Horocycle** (horo-CE) | <r>**NaN — diverges (all 10)**</r> | finite · `wnelbo` **0.33–0.95** |
| **Direct** (CE) | finite · `ce` 0.22–1.98 · `wnelbo` 0.68–77 | finite · $\approx$ Learnable |

- **Learnable embedding breaks Horocycle** (back-prop through the singular $\log(1-\cos(\theta-\phi_v))$ via $\phi_v$) but is **harmless for Direct** — Direct Learnable $\approx$ Fixed.
- **Horocycle $\times$ Fixed** is the only well-behaved bridge ELBO (`test_wnelbo` $\approx 0.33$–$0.47$, low variance on `unif`).
- **Direct (CE)** is a strong *denoiser* (low conditional `test_ce` on `unif`) but a poor *ELBO* estimator (large, heavy-tailed `test_wnelbo`).

---

## Horocycle (`horo_cross_entropy`)

**$\times$ Learnable:** `test_ce` $=$ `test_wnelbo` $=$ <r>**NaN**</r> for **every** proposal — the loss differentiates $\log(1-\cos(\theta-\phi_v))$ through the learnable $\phi_v$.

**$\times$ Fixed** — `test_ce` $\pm$ std · `test_wnelbo` $\pm$ std:

| proposal | test_ce | test_wnelbo |
|---|--:|--:|
| unif hT=1000 | 0.441 ± 1.09 | **0.330** ± 1.67 |
| unif hT=2000 | 0.475 ± 1.02 | 0.383 ± 2.58 |
| unif hT=3000 | 0.536 ± 0.96 | 0.392 ± 3.22 |
| unif hT=5000 | 0.679 ± 0.85 | 0.394 ± 4.21 |
| strat λ=0.1 | 0.478 ± 1.06 | 0.402 ± 2.65 |
| strat λ=0.2 | 0.446 ± 1.19 | 0.409 ± 3.82 |
| strat λ=0.3 | 0.429 ± 1.29 | 0.417 ± 5.52 |
| strat λ=0.5 | 0.478 ± 1.31 | 0.472 ± 11.3 |
| strat λ=0.8 | 0.677 ± 1.12 | 0.628 ± 15.9 |
| strat λ=1.0 | 1.085 ± 0.88 | 0.953 ± 15.7 |

---

## Direct (`cross_entropy`) — Learnable $\approx$ Fixed

| proposal | ce (Learn) | ce (Fixed) | wnelbo (Learn) | wnelbo (Fixed) |
|---|--:|--:|--:|--:|
| unif hT=1000 | 0.340 | 0.316 | 0.684 | 0.721 |
| unif hT=2000 | 0.283 | 0.262 | 1.308 | 1.380 |
| unif hT=3000 | 0.251 | 0.238 | 1.879 | 2.008 |
| unif hT=5000 | 0.222 | 0.217 | 2.981 | 3.256 |
| strat λ=0.1 | 1.958 | 1.984 | 77.2 | 79.2 |
| strat λ=0.2 | 1.967 | 1.981 | 31.0 | 32.7 |
| strat λ=0.3 | 1.976 | 1.979 | 20.0 | 21.7 |
| strat λ=0.5 | 1.982 | 1.979 | 11.2 | 12.7 |
| strat λ=0.8 | 1.984 | 1.981 | 6.18 | 7.61 |
| strat λ=1.0 | 1.985 | 1.982 | 4.58 | 5.93 |

- `ce_std`: $\approx$ 0.8–1.2 (`unif`), $\approx$ 0.18–0.26 (`stratified`, small $\Rightarrow$ <r>confidently wrong</r>).
- `wnelbo_std`: **huge on stratified** ($\approx$ 75–1130) $\Rightarrow$ heavy-tailed, unreliable ELBO.

---

## Conclusion

- **Trainable embedding destabilizes only the Horocycle loss** $\Rightarrow$ NaN (it differentiates $\log(1-\cos(\theta-\phi_v))$ through $\phi_v$). **Direct (CE) is indifferent** to learnable vs fixed — no singular kernel in its loss, so Learnable $\approx$ Fixed.
- **Fixed-embedding Horocycle** is the only setting with a sensible, low-variance bridge ELBO (`test_wnelbo` $\approx 0.33$–$0.47$).
- **Direct (CE)** wins on conditional `test_ce` (denoising, beats $H$ on `unif`) but is a poor, heavy-tailed ELBO estimator; its `stratified` `test_ce` plateaus at $\approx 1.98 \approx 4H$ ("confidently wrong").

---

# Hyperbolic Riemannian FLM

---

## Motivation

Hyperspherical FLM claims that they found normalizing the word embedding might cause worse performance.

![alt text](image-1.png)

---

## Idea

Hyperbolic space is a more natural way to model the word embedding with varying length.


Some old papers have shown that hyperbolic space can represent word embedding with very low dimension
- [POINCARE GLOVE: HYPERBOLIC WORD EMBEDDINGS](https://arxiv.org/pdf/1810.06546)


---

## Method

- [FLOW MATCHING ON GENERAL GEOMETRIES](https://arxiv.org/pdf/2302.03660) has shown that the OT path on a manifold is geodesic. 
- $q_{t|1}(z | x)$ can be computed by free hyperbolic heat kernel + geodesic moving
- Use Cross Entropy Loss instead of velocity regression

---

## Cross Entropy Loss

$$
\begin{aligned}
\mathcal{L}_{CE}(\theta)
 & =
\mathbb{E}_{z_{t} \sim q_{t \mid 1}(\cdot \mid x), t \sim \text{Unif}([0, 1]), y \sim q_{data}}
\left[
	- \sum_{i=1}^{L} \log p_{1 | t}^{\theta}(x^{i} | z_{t})
\right] \\
& = \mathbb{E}_{z_{t} \sim q_{t \mid 1}(\cdot \mid x), t \sim \text{Unif}([0, 1]), y \sim q_{data}}
\left[ 
	- \sum_{i=1}^{L} \log \mu^{\theta, i}(z_t) 
\right]
\end{aligned}
$$

where the word embedding at $i$-th position is $x^{i} = e^{y^i}$ and $y^i$ is the token index at position $i$. The model output $\mu^{\theta}$ can be represented as 

$$
\mu^{\theta}(z_t) 
:= \operatorname{softmax}(f_{\theta}(z_t))
$$

---

## Hyperbolic Geodesic

Lorentz model $\mathbb{H}^d=\{z:\langle z,z\rangle_L=-1\}$, with $\langle a,b\rangle_L=-a_0 b_0+\sum_{i\ge 1} a_i b_i$. Constant-speed geodesic from $x$ ($t{=}0$) to $y$ ($t{=}1$):

$$
\gamma(t)=\frac{\sinh\big((1-t)\,d\big)}{\sinh d}\,x+\frac{\sinh\big(t\,d\big)}{\sinh d}\,y,
\qquad d=\operatorname{arccosh}\!\big(-\langle x,y\rangle_L\big)
$$

---

## Hyperbolic Geodesic

- **It is SLERP with $\sin\!\to\!\sinh$, $\arccos\!\to\!\operatorname{arccosh}$** — same code, curvature flag $\kappa$: $\kappa{=}{+}1$ sphere ($\sin$), $\kappa{=}{-}1$ hyperboloid ($\sinh$). Stays on-manifold, $\langle\gamma(t),\gamma(t)\rangle_L=-1$.
- **Numerically stable distance** — compute $d$ from the *difference vector*, not $\langle x,y\rangle_L$ (which cancels to $\approx 1$ at large radius / high $d$):

$$
\cosh d-1=\tfrac12\,\langle x-y,\;x-y\rangle_L
\;\;\Rightarrow\;\;
d=\operatorname{arccosh}\!\Big(1+\tfrac12\langle x-y,x-y\rangle_L\Big)
$$

---

## FLOW MATCHING ON GENERAL GEOMETRIES

![alt text](image-3.png)

---

## FLOW MATCHING ON GENERAL GEOMETRIES

![w:600px center](image-2.png)

---

## High Dimensional Hyperbolic Free Heat Kernel

- High-Dimension Free Hyperbolic kernel sampling
  - [THE HEAT KERNEL ON ASYMPTOTICALLY HYPERBOLIC MANIFOLDS](https://arxiv.org/pdf/1612.06044)

---

## Some Potential Issue

sampling noisy data points by manifold linear interpolated method might be inaccurate if the source $x_0$ and target $x_1$ are far away.

$$
x_t = \exp_{x_1} (\kappa(t) \log_{x_1} (x_0))
$$

---

# Progress

- I've used Claude to implement the geodesic, free hyperbolic heat kernel sampler, and Poisson kernel. But I might need your help to double check.
- Still implementing
- Haven't derived the ELBO, but I think the performance on easy Sudoku is the most important test

---



### Trainable boundary points: CE survives, Poincaré ELBO diverges

---

## Set Up

- Driver `unigram/unigram_test2_tmp4.py` · `mode=tnb` (**trains** the denoiser) · seed 42 · 4M test
- **Learnable** embeddings: each word's boundary angle $=\operatorname{atan2}$ of its lm-head row
- Two train losses: <r>`cross_entropy`</r> and <b>`poincare_polar`</b> ELBO with uniform and exp proposal swept
- Target $=$ data entropy **0.5003** · CE `test_wnelbo` eval $=$ fixed PP stratified-exp(0.1)

---

## CE-Fixed Proposal Set Up

- CE `test_wnelbo` is evaluated by fixed proposal stratified-exp(0.1)

---

## CE-Fixed Proposal — Stable, but Stuck 

| proposal | `test_ce` | `ce_std` | `test_wnelbo` | `wnelbo_std` |
|:--|:--:|:--:|:--:|:--:|
| unif   | <b>0.356</b> | 1.17 | <b>0.570</b> | 3.74 |
| exp 0.1 | 1.959 | 0.18 | 2.215 | 5.56 |
| exp 0.2 | 1.968 | 0.21 | 2.227 | 5.58 |
| exp 0.3 | 1.977 | 0.23 | 2.219 | 5.58 |
| exp 0.5 | 1.984 | 0.24 | 2.195 | 5.56 |
| exp 0.8 | 1.986 | 0.24 | 2.167 | 5.54 |
| exp 1.0 | 1.987 | 0.24 | 2.154 | 5.53 |

---

## PP-Fixed Proposal Set Up

- PP (Poincare Polar) `test_wnelbo` is evaluated by identical proposal as loss

---

## PP-trained — Diverges to NaN

| proposal | `test_ce` | `ce_std` | `test_wnelbo` | `wnelbo_std` |
|:--|:--:|:--:|:--:|:--:|
| exp | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> | <r>NaN</r> |

`train_loss`: **2.07 → NaN at step 2.** Every Poincaré-ELBO run blows up almost immediately

---

## Insight — The Embedding Was the Wrong Knob

| train loss | fixed emb | learnable emb |
|:--|:--:|:--:|
| <b>Poincaré ELBO</b> | **best**, `wnelbo` ≈ 0.40 | <r>NaN — step 2</r> |
| <r>Cross-entropy</r> | ~2.0 | ~2.0 |

- **Trainable boundary points destabilize the ELBO**: its $1/\lVert y-z_t\rVert^2$ and $(1-\lVert z\rVert^2)^2$ terms blow up once $y$ can drift toward $z_t$.
- **CE is robust but uninformative** — learnable vs fixed barely moves `test_ce`; still ~4× the entropy.
- Net: learnability **broke the one loss that worked** (fixed-emb PP) and didn't help CE → needs boundary regularization / grad-clip / lower lr before ELBO training is viable.

