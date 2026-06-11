#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import transformers
import datasets
import tokenizers
from tokenizers import Tokenizer
import omegaconf
import yaml
import itertools
from tqdm import tqdm

from matplotlib import pyplot


# In[3]:


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------

class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)     # (s, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (s, dim)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor,
                         k: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor):
    # q, k : (b, s, h, d)   cos, sin : (s, d)
    cos = cos[None, :, None, :]   # (1, s, 1, d)
    sin = sin[None, :, None, :]   # (1, s, 1, d)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        x = F.layer_norm(x.float(), [self.dim]).to(x.dtype)
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


# ---------------------------------------------------------------------------
# Label embedder
# ---------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class DDiTMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        mlpdim = mlp_ratio * dim
        self.up = nn.Linear(dim, mlpdim, bias=False)
        self.gate = nn.Linear(dim, mlpdim, bias=False)
        self.down = nn.Linear(mlpdim, dim, bias=False)

    def forward(self, x):
        return self.down(self.up(x) * F.silu(self.gate(x)))

class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = DDiTMLP(dim, mlp_ratio)
        self.dropout = dropout

    def forward(self, x, rotary_cos_sin):
        # --- attention ---
        x = x + F.dropout(
            self.attn_out(self._attn(self.norm1(x), rotary_cos_sin)),
            p=self.dropout, training=self.training)

        # --- MLP ---
        x = x + F.dropout(
            self.mlp(self.norm2(x)),
            p=self.dropout, training=self.training)

        return x

    def _attn(self, x, rotary_cos_sin):
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                        three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=2)   # each (b, s, h, d)

        cos, sin = rotary_cos_sin
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return rearrange(attn, 'b h s d -> b s (h d)')


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.linear.weight.data.zero_()
        # self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(self.norm_final(x))


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size*2)
        self.rotary_emb = Rotary(
            config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size,
                      config.model.n_heads,
                      dropout=config.model.dropout)
            for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size)
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x):
        # x = self.vocab_embed(indices)
        rotary_cos_sin = tuple(r.to(x.dtype) for r in self.rotary_emb(x.shape[1], x.device))
        for block in self.blocks:
            x = block(x, rotary_cos_sin)
        return self.output_layer(x)


# In[4]:


def sample_chi(ns,dtype=torch.float64):
    nshape = ns.shape
    ns = ns.reshape(-1)
    M = ns.sum().item()
    x = torch.randn(M,dtype=dtype).square()
    chi2 = torch.segment_reduce(x,'sum',lengths=ns)
    return chi2.sqrt().reshape(nshape)

@torch.no_grad()
def bbridge(ts):
    ns = torch.poisson(ts/8).to(torch.int64)
    ss = ts.sqrt() * sample_chi(2*ns+3, ts.dtype)
    vs = torch.rand_like(ts)
    ps = torch.acosh(vs.square() + (1-vs.square())*torch.cosh(ss))
    us = torch.rand_like(ts)
    thetas = 2 * torch.atan((-ps).exp() * torch.tan(torch.pi * (us - 0.5)))
    return (ps,thetas)

# logits        (N,V)     float64 (converts)
# targets       (N,)      int64
# rhos          (N,)      float64
# thetas        (N,)      float64
def bridge_loss(forward, targets, rhos, thetas):
    (N,) = targets.shape
    (N,V) = logits.shape
    assert(rhos.shape == (N,))
    assert(thetas.shape == (N,))
    assert(targets.dtype == torch.int64)
    assert(rhos.dtype == torch.float64)
    assert(thetas.dtype == torch.float64)
    # construct phis
    phis = (torch.arange(V,dtype=torch.float64) + 0.5) * (2 * torch.pi / V)
    # first, we get the horosphere distances
    alphas = thetas[:,None] - phis[None,:]  # angular offsets between z and v
    cos_alphas = alphas.cos()
    sin_alphas = alphas.sin()
    horosphere_dists = torch.log(torch.tensor(2.0,dtype=torch.float64)) - torch.logaddexp((1 - cos_alphas).log() + rhos[:,None], (1 + cos_alphas).log() - rhos[:,None])
    # remake mu and subtract the target
    mu = (horosphere_dists + logits.to(torch.float64)).softmax(-1)
    mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float64)
    # next, we transform the angles alpha after motion by rho
    betas = torch.atan2(sin_alphas, rhos.cosh()[:,None] * cos_alphas - rhos.sinh()[:,None])
    cos_errors = (betas.cos() * mu).sum(-1)
    sin_errors = (betas.sin() * mu).sum(-1)
    return (cos_errors.square() + sin_errors.square())/2


# In[5]:


def ensure_log2(V):
    assert(V > 0)
    assert(V & (V-1) == 0)
    rv = int(math.log2(V))
    assert((1<<rv) == V)
    return rv

# scaled from 0 to log2(V) - 2
def rescale_rho(rhos, V):
    return torch.log1p(-(V-4)*torch.expm1(-rhos)/(4+V*torch.exp(-rhos))) / math.log(2.0)

def tf_embed(rhos, thetas, embeds):
    (V2,d) = embeds.shape
    assert(V2 % 2 == 0)
    V = V2 // 2
    (N,) = rhos.shape
    assert((N,) == thetas.shape)
    assert(rhos.dtype == torch.float64)
    assert(thetas.dtype == torch.float64)
    k = ensure_log2(V)
    # sanity checks
    assert(V > 8)
    assert(rhos.isfinite().all())
    assert(thetas.isfinite().all())
    assert((rhos >= 0).all())
    # rescale rhos: scaled from 0 to k-2
    rhos = rescale_rho(rhos, V)
    rhos = rhos.clamp(min=0, max=torch.nextafter(torch.tensor(k-2,dtype=torch.float64),torch.tensor(0.0,dtype=torch.float64)))
    rhos_floor = rhos.floor()
    rhos_fpart = rhos - rhos_floor
    rhos_floor = rhos_floor.to(torch.int64)
    floor_bits = rhos_floor + 2
    ceil_bits = rhos_floor + 3
    # shift and scale thetas
    thetas = (thetas / (2 * torch.pi)) % 1.0
    thetas_int = (thetas + 1.0).view(torch.int64) # viewing these as integers in 52-bit mantissa
    # bitwise magic!
    one = torch.tensor(1.0,dtype=torch.float64).view(torch.int64)
    theta_floor_idx =     ((thetas_int + (1 << (51 - floor_bits))) >> (52 - floor_bits)) & ((1 << floor_bits) - 1)
    theta_floor_fpart = ((((thetas_int + (1 << (51 - floor_bits))) << floor_bits) & ((1 << 52) - 1)) | one).view(torch.float64) - 1
    theta_ceil_idx  =     ((thetas_int + (1 << (51 -  ceil_bits))) >> (52 -  ceil_bits)) & ((1 <<  ceil_bits) - 1)
    theta_ceil_fpart  = ((((thetas_int + (1 << (51 -  ceil_bits))) <<  ceil_bits) & ((1 << 52) - 1)) | one).view(torch.float64) - 1
    theta_floor_idx_next = (theta_floor_idx + 1) & ((1 << floor_bits) - 1)
    theta_ceil_idx_next  = (theta_ceil_idx + 1)  & ((1 <<  ceil_bits) - 1)
    # adjust for rhos_floor
    theta_floor_idx      = theta_floor_idx      + (1 << floor_bits)
    theta_floor_idx_next = theta_floor_idx_next + (1 << floor_bits)
    theta_ceil_idx       = theta_ceil_idx       + (1 << ceil_bits)
    theta_ceil_idx_next  = theta_ceil_idx_next  + (1 << ceil_bits)
    # if rhos_floor = 0, need to set floor_idx = 0, since this is just the origin point with only one embedding
    theta_floor_idx      = theta_floor_idx      * (rhos_floor != 0)
    theta_floor_idx_next = theta_floor_idx_next * (rhos_floor != 0)
    theta_floor_fpart    = theta_floor_fpart    * (rhos_floor != 0)
    # calculate weights
    w_floor_idx      = (1 - rhos_fpart) * (1 - theta_floor_fpart)
    w_floor_idx_next = (1 - rhos_fpart) * theta_floor_fpart
    w_ceil_idx       = rhos_fpart       * (1 - theta_ceil_fpart)
    w_ceil_idx_next  = rhos_fpart       * theta_ceil_fpart
    # calculate normalization scale
    w_scale = (w_floor_idx.square() + w_floor_idx_next.square() + w_ceil_idx.square() + w_ceil_idx_next.square()).rsqrt()
    # form output
    out = (
          embeds[theta_floor_idx,:] * w_floor_idx[:,None].to(embeds.dtype)
        + embeds[theta_floor_idx_next,:] * w_floor_idx_next[:,None].to(embeds.dtype)
        + embeds[theta_ceil_idx,:] * w_ceil_idx[:,None].to(embeds.dtype)
        + embeds[theta_ceil_idx_next,:] * w_ceil_idx_next[:,None].to(embeds.dtype)
    ) * w_scale[:,None].to(embeds.dtype)
    return out


# In[6]:


# from huggingface_hub import notebook_login


# In[7]:


# notebook_login()


# In[8]:


# tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


# In[9]:


# trainset = torch.load("drive/MyDrive/Colab Notebooks/trainset_circle.pt", weights_only=True)  # data unavailable
import utils  # all experiment utilities (proposals, dataset, accumulation, plotting) live here


# In[10]:


# train_batches = trainset['mapped_batches'].view(-1,64,32)   # ORIGINAL data
# (nbatches,batchsize,seqlen) = train_batches.shape           # set up the dataset same as the slides:
V, ps, initial_bias_vec, (nbatches,batchsize,seqlen) = utils.unigram_dataset()  # V=10, ps=[0.91,0.01x9]
print(f'nbatches: {nbatches}')
print(f'batchsize: {batchsize}')
print(f'seqlen: {seqlen}')
print(f'V: {V}')


# In[11]:


# trainset.keys()


# In[12]:


# initial_bias = trainset['initial_bias'].to(torch.float64).cuda()  # ORIGINAL (GPU); no GPU here -> CPU float64
initial_bias = initial_bias_vec.to(torch.float64)   # = log p(y): the optimal logits / equally-divided prior
initial_probs = initial_bias.exp()
prior_entropy = torch.special.entr(initial_probs).sum()
print(initial_probs.sum().item())
print(prior_entropy.item())   # prior_entropy == data entropy H ~= 0.5003 nats (the ELBO target)


# In[13]:


# token_perm = torch.randperm(V)
# token_iperm = token_perm.sort().indices
# train_batches = token_perm[train_batches.to(torch.int32)]
# initial_bias = initial_bias[token_iperm]
# initial_probs = initial_probs[token_iperm]


# In[14]:


config = yaml.safe_load('''name: small
type: ddit
hidden_size: 768
cond_dim: 128
length: 1024
n_blocks: 12
n_heads: 12
scale_by_sigma: True
dropout: 0.1
tie_word_embeddings: False''')
config = {'model':config}



# name: tiny
# type: ddit
# hidden_size: 512
# cond_dim: 128
# length: 1024
# n_blocks: 8
# n_heads: 8
# scale_by_sigma: True
# dropout: 0.1
# tie_word_embeddings: False


# In[15]:


# checkpoint = torch.load('drive/MyDrive/Colab Notebooks/ddit_circle.pt')  # checkpoint unavailable; eval untrained model


# In[16]:


model = DIT(config=config,vocab_size=V)
model.eval()   # ESSENTIAL: evaluate, not train. DDitFinalLayer zero-inits its weight, so an
# model = model.to('cuda').to(torch.bfloat16)   # untrained eval DIT outputs 0 -> logits =
# model.load_state_dict(checkpoint['model'])    # model(z)-model(z0)+initial_bias = log p(y) =
# Bayes-optimal model, used directly below (tf_embed also needs V=2^k; slide V=10).


# In[17]:


# heuristic setting for lambda
# lamb = 0.06   # superseded: the proposal (incl. its rate) is now swept, see utils.PROPOSALS


# In[18]:


# opt = torch.optim.Adam(model.parameters(), 3e-4, (0.9,0.999), 1e-8) # 3e-4      # COMMENTED: no training
# scheduler = transformers.get_constant_schedule_with_warmup(opt, 1000)            # COMMENTED: no training


# In[19]:


# ESSENTIAL MODIFICATION: notebook training loop -> eval-only ELBO sweep over
# stratified_exp + unif (no backprop; variance-reduction lambda-estimator commented out).
results = {}
for (pname, pkw) in utils.PROPOSALS:           # sweep stratified_exp + unif (slide config)
 for SEED in utils.SEEDS:
  torch.manual_seed(SEED)                       # bbridge() draws from the global RNG
  g = torch.Generator().manual_seed(SEED)
  N = nbatches*batchsize*seqlen
  train_batches = torch.multinomial(initial_probs, N, replacement=True, generator=g).view(nbatches,batchsize,seqlen)
  ts_all, w_all = utils.sample_proposal(N, generator=g, **pkw)   # MODIFIED: was per-batch stratified-only
  acc = utils.Accum(H=prior_entropy.item())
  saved_losses = []
  avg_loss = 0.0
  avg_loss_denom = 0.0
  # lamb_est_num = 0.0; lamb_est_den = 0.0       # COMMENTED: variance-reduction (adaptive-lambda) tricks
  progress_bar = tqdm(range(nbatches), desc=f"{pname} seed={SEED}")
  for ibatch in progress_bar:
      # opt.zero_grad()                          # COMMENTED: no training
      targets = train_batches[ibatch,:].to(torch.int64).reshape(-1)
      ts = ts_all[ibatch*batchsize*seqlen:(ibatch+1)*batchsize*seqlen]      # MODIFIED: slice precomputed proposal
      weights = w_all[ibatch*batchsize*seqlen:(ibatch+1)*batchsize*seqlen]
      (rhos, thetas) = bbridge(ts)
      thetas = thetas + (targets.to(torch.float64) + 0.5) * (2 * torch.pi / V)
      assert(rhos.isfinite().all())
      assert(thetas.isfinite().all())
      # z = tf_embed(...); logits = model(z) - model(z0) + initial_bias     # COMMENTED: untrained DIT=0 & V!=2^k
      logits = initial_bias[None,:].expand(targets.shape[0], V)             # logits = log p(y) = optimal model
      losses = bridge_loss(logits, targets, rhos, thetas)
      losses = losses * weights
      l = losses.mean() # the actual minibatch loss (per-batch weighted NELBO)
      avg_loss = (avg_loss * 0.99) + l.item()
      avg_loss_denom = (avg_loss_denom * 0.99) + 1.0
      saved_losses.append(avg_loss/avg_loss_denom)
      acc.update(losses, logits, targets, weights)
      progress_bar.set_description(f"{pname} seed={SEED} [avg_nelbo={avg_loss/avg_loss_denom:.4f}]")
      # l.backward(); opt.step(); scheduler.step()   # COMMENTED: no training
  results[(pname, SEED)] = acc.summary()
  print(f"[{pname} seed={SEED}] test_wnelbo={results[(pname,SEED)]['wnelbo_mean']:.4f}"
        f" +/- {results[(pname,SEED)]['wnelbo_std']:.2f}  test_ce={results[(pname,SEED)]['ce_mean']:.4f}")

utils.dump_results(results)                                       # per-seed JSON (parallel-friendly)
if len(utils.SEEDS) > 1:                                          # full run -> figure + table + slide comparison
    utils.aggregate_and_plot(utils.load_all_results(), prior_entropy.item())


# In[ ]:


pyplot.figure(); pyplot.plot(saved_losses); pyplot.xlabel('batch'); pyplot.ylabel('running NELBO')
pyplot.title(f'eval NELBO curve (last proposal/seed)'); pyplot.savefig('loss_curve.png')   # log training/testing loss


# In[ ]:


# sd = model.state_dict()   # nothing to save: model is untrained (eval-only)
# torch.save({"model": sd, "saved_losses": saved_losses},'drive/MyDrive/Colab Notebooks/ddit_circle_randperm.pt')


# In[ ]:


# opt.param_groups[0].keys()


# In[ ]:


# sd = model.state_dict()
# torch.save({"model": sd, "saved_losses": saved_losses},'drive/MyDrive/Colab Notebooks/ddit_plane_d2_latest_34481.pt')


# In[ ]:


len(saved_losses)


# In[ ]:




