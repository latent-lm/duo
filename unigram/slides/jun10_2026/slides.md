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

#### June 10, 2026

---

# Sudoku Exp: Setting Up

**Same recipe for all systems — reproduces *Hyperspherical Flows* (arXiv:2605.11125, Tbl 1)**

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M) · 81-cell grid (seq-len 180)

---

# Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0

---

# Sudoku Exp: Setting Up

| Model | Space | Sampler @ eval (180 steps) |
|---|---|---|
| AR | discrete, causal | greedy autoregressive |
| MDLM · Duo | discrete, masked | ancestral |
| CANDI | discrete | CANDI |
| FLM (one-hot) | simplex | Euler ODE |

---

# Sudoku Exp: Setting Up

| Model | Space | Sampler @ eval (180 steps) |
|---|---|---|
| S-FLM (naive · trunc · +adaptive) | sphere | exact-velocity, greedy, top_k_v = 1 (top-1) |
| **HFLM (ours)** | hyperbolic | exact-velocity, greedy, top_k_v = 1 (top-1) |

<!-- - S-FLM noise: log-linear · +<b>α⋆=0.093</b> trunc · +adaptive (refit 50) · HFLM prior: cov 0.25, ρ_max 12 -->

---

# Sudoku Exp Results

**Exact-match accuracy (%)**

All values are reported by the paper

| Model | Easy | Med | Hard |
|---|---|---|---|
| AR (greedy) | 14.6 | 5.1 | 1.0 |
| MDLM | 92.0 | 77.1 | 30.2 |
| Duo | 96.3 | 84.7 | 58.4 |
| CANDI | 79.3 | 45.9 | 16.7 |
| FLM (one-hot) | 94.2 | 82.7 | 44.5 |

---

# Sudoku Exp Results

**Exact-match accuracy (%)**

S-FLMs: Reproduced, Use top_k_v = -1, velocity = average across the whole vocab

| Model | Easy | Med | Hard |
|---|---|---|---|
| S-FLM (naive) | 77.6 | 32.6 | 13.9 |
| S-FLM + α⋆(0.093) trunc | 95.1 | 77.9 | 48.3 |
| S-FLM + α⋆(0.093) + adaptive | 94.2 | 79.2 | 46.0 |
| **HFLM (naive, ours)** | **93.75** | **67.40** | **24.10** |

---

# Sudoku Exp Results

**Exact-match accuracy (%)**

S-FLMs: Reproduced, Use top_k_v = 1, velocity = only choose top-1 token

| Model | Easy | Med | Hard |
|---|---|---|---|
| S-FLM (naive) | 76.6 | 33.2 | 15.2 |
| S-FLM + α⋆(0.093) trunc | 94.8 | 78.1 | 47.6 |
| S-FLM + α⋆(0.093) + adaptive | 93.8 | 78.9 | 44.9 |
| **HFLM (naive, ours)** | **93.75** | **67.40** | **24.10** |

---

# Sudoku Exp Results: HFLM

**Geometry alone lifts the *naive* model: HFLM (naive) ≫ S-FLM (naive, reproduced)**

| naive model (ours, local) | easy | med | hard |
|---|---|---|---|
| S-FLM (naive), top_k_v = -1 | 77.6 | 32.6 | 13.9 |
| S-FLM (naive), top_k_v = 1 | 76.6 | 33.2 | 15.2 |
| **HFLM (naive)** | **93.8** | **67.4** | **24.1** |

- Hyperbolic geometry *alone* perform better than naive S-FLM
- HFLM still trails the *tricked* S-FLM

---

# Sudoku Exp Results: HFLM samples

**Failures are near-misses — HFLM learned the rules; errors are local digit swaps**

✓ easy — generated grid, fully correct (all 81 cells):

```
9 2 1 6 4 7 3 8 5
7 6 4 8 3 5 2 9 1
5 8 3 9 1 2 7 6 4
4 5 2 3 6 9 1 7 8
1 7 9 4 5 8 6 2 3
8 3 6 7 2 1 5 4 9
2 1 7 5 8 4 9 3 6
6 9 8 1 7 3 4 5 2
3 4 5 2 9 6 8 1 7
```

---

# TinyStories Exp: Setting Up

**Same recipe for all 3 — only the geometry/noise differs (geometry vs. tricks)**

- Data: TinyStories (gpt2 tok, ~472M train tokens) · ~33 epochs
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12** (~169M)

---

# TinyStories Exp: Setting Up

**Same recipe for all 3 — only the geometry/noise differs (geometry vs. tricks)**

- Training
  - Training Steps: **30k** · Batch Size: **512** · Max Seq Len: **1024**
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0 · EMA: 0.9999

---

# TinyStories Exp: Setting Up

**The 3 models** (everything above held fixed)

| Model | Geometry | Noise schedule |
|---|---|---|
| S-FLM (naive) | sphere | log-linear |
| S-FLM (trunc + adaptive) | sphere | log-linear, <b>α⋆=0.121</b> + adaptive |
| **HFLM (ours)** | hyperbolic | log-linear (prior cov 0.25, ρ_max 12) |

---

# TinyStories Exp: Setting Up

- Evaluation
  - Sampling Steps (NFE): **1024**
  - Sampler: geometry-matched (sphere / hyperbolic geodesic)
  - noise_removal: greedy (HFLM) / ancestral (S-FLM)
  - top_k_velocity: top-1 (HFLM, 1) / top-1 (S-FLM, 1)
  - Metrics: 
    - <b>GenPPL</b> (gpt2-large on samples) = the comparison
    - <r>"val PPL" = held-out denoising CE, a diagnostic — *not* a true perplexity</r>

---

# TinyStories Exp Results

**HFLM converges early; generation quality peaks mid-training, not at the end**

| HFLM ckpt | val CE (`exp`) | GenPPL ↓ | entropy |
|---|---|---|---|
| 10k | 6.00 | 45.0 | 4.88 |
| 20k | 6.14 | **41.6** | 4.89 |
| 30k | 6.02 | 49.2 | 4.90 |

- Held-out denoising loss is **flat ~6.0** across 10k→30k — the training-objective proxy is saturated and **does not track generation quality**.
- GenPPL is **non-monotonic** (best at 20k): more steps ≠ better samples.
- Entropy steady ~4.9 → samples stay **diverse, not degenerate** → the GenPPL is real.
- ⇒ Must judge with a generation-time metric (GenPPL), and checkpoint-select on it.

---

# TinyStories Exp Results

**Samples: fluent TinyStories micro-narratives — coherent locally, drifts globally** (HFLM, 20k)

> *Once upon a time there was a little girl named Sarah. She was only three years old … she was at the beach with her mom. … It was a big bottle of sweet water and began coll[ecting] …*

> *Once upon a time, there was a little boy named Timmy. Timmy was excited because he liked to play with his wagon outside … Timmy was curious and liked to play with it.*

- ✅ Names, story openings, simple arcs, child-level vocabulary — **clearly TinyStories**.
- ⚠️ Local slips — repetition (*"and and"*, *"soon soon"*), topic drift, occasional ungrammatical spans — the **non-AR (parallel-denoising) artifact**: strong token-local fluency, weaker long-range discourse.

---

# Background

### nGPT: Normalized Transformer with Representation Learning on the Hypersphere

- NVIDIA, ICLR 2025
- **Idea**: constrain every vector — embeddings, weight rows, hidden states — to the **unit hypersphere**
- Every matrix-vector product becomes a **cosine similarity** in $[-1, 1]$
- Each layer moves $h$ along the sphere toward the next-token prediction
- **Result**: 4–20x fewer training steps for the same loss

---

## What nGPT Normalizes

Everything with an embedding dimension, **after each optimizer step**:

- **Token embeddings**: both $E_{input}$ **and** $E_{output}$
- **Attention matrices**: $W_q, W_k, W_v, W_o$
- **MLP matrices**: $W_u, W_\nu, W_{oMLP}$

---

## What nGPT Normalizes

- **Hidden state $h$**: re-normalized after every attention / MLP update
  - replaces the residual add: $h \leftarrow \text{Norm}(h + \alpha(h_A - h))$
- **$q$ and $k$ vectors**: extra normalization before the dot product (RoPE distorts them)

No more LayerNorm / RMSNorm, weight decay, or LR warmup

---

## What nGPT Does NOT Normalize

Normalization erases magnitude → restore it with **learnable scaling factors**:

- **Logits**: $z \leftarrow z\, s_z$ — raw logits $z = E_{output} h$ are cosine similarities stuck in $[-1,1]$
  - $s_z \in \mathbb{R}^V$ is **per-token**: a learnable **length** for each output embedding, restoring softmax confidence (temperature)
- **QK scaling** $s_{qk}$: softmax scaling becomes $\sqrt{d_k}$ instead of $1/\sqrt{d_k}$

---

## What nGPT Does NOT Normalize

- **MLP intermediate states**: $u \leftarrow u\, s_u$, $\quad \nu \leftarrow \nu\, s_\nu \sqrt{d_{model}}$ (needed for SiLU non-linearity)

---

## Riemannian Gradient Descent View

$$h \leftarrow \text{Norm}(\,h + \alpha_A(h_A - h)\,)$$

Each layer = one optimization step **on the hypersphere**:

- **Gradient**: $g = h - h_A$ — the block's suggested displacement
- **Variable metric**: $\alpha$ ("eigen learning rates") $\approx$ diagonal of a learnable inverse-Hessian $B$: $\;h \leftarrow h - \alpha B g$
- **Retraction**: $\text{Norm}(\cdot)$ maps the updated point back onto the manifold

Exact Riemannian GD projects $g$ onto the tangent space:
$g_{proj} = h(h^T h_A) - h_A$ — but the $h^T h_A$ term is empirically negligible

→ the Transformer itself acts as a **variable-metric optimizer** on the sphere

---

## Experiments: 10x Fewer Iterations (Fig. 1)

1B models, 4k context, OpenWebText — nGPT at **20k** iters matches GPT at **200k**

![w:520 center](image-1.png)

---

## Experiments: Speedup Grows with Context (Fig. 2)

Same final loss with ~**4x** (1k ctx), **10x** (4k), **20x** (8k) fewer tokens

![w:560 center](image.png)