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

Cover S-FLM, and HFLM

- Model (DiT):
  - Width: , Depth: ,
  - Num of Heads:
- Training
  - Training Steps:
  - Batch Size:
  - Max Sequence Length
  - Optimizer: 
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0

---

# Sudoku Exp: Setting Up

- Evaluation
  - Sampling Steps: 180
  - Sampler Type:
  - Top-K / Velocity: 

---

> TODO: check the hyperparameter of AR, MDLM, Duo, CANDI, and FLM

| Model | Easy | Med | Hard |
|---|---|---|---|
| AR (greedy) | 14.6 | 5.1 | 1.0 |
| MDLM | 92.0 | 77.1 | 30.2 |
| Duo | 96.3 | 84.7 | 58.4 |
| CANDI | 79.3 | 45.9 | 16.7 |
| FLM (one-hot) | 94.2 | 82.7 | 44.5 |
| S-FLM (naive) | 81.5 | 50.6 | 14.0 |
| S-FLM + α⋆(0.1) trunc | 94.0 | 77.6 | 43.2 |
| S-FLM + α⋆(0.1) + adaptive | 94.8 | 85.2 | 45.0 |
| **HFLM (naive, ours)** | **93.75** | **67.40** | **24.10** |

---

# TinyStories Exp: Setting Up

**Same recipe for all 3 — only the geometry/noise differs (geometry vs. tricks)**

- Data: TinyStories (gpt2 tok, ~472M train tokens) · ~33 epochs
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12** (~169M)
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
  - top_k_velocity: top-1 (HFLM, 1) / All (S-FLM, -1)
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

# Idea: Hyperbolic GPT

---

