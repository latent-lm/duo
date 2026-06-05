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


trainset = torch.load("drive/MyDrive/Colab Notebooks/trainset_circle.pt", weights_only=True)


# In[10]:


train_batches = trainset['mapped_batches'].view(-1,64,32)
(nbatches,batchsize,seqlen) = train_batches.shape
V = trainset['initial_bias'].shape[0]
print(f'nbatches: {nbatches}')
print(f'batchsize: {batchsize}')
print(f'seqlen: {seqlen}')
print(f'V: {V}')


# In[11]:


trainset.keys()


# In[12]:


initial_bias = trainset['initial_bias'].to(torch.float64).cuda()
initial_probs = initial_bias.exp()
prior_entropy = torch.special.entr(initial_probs).sum()
print(initial_probs.sum().item())
print(prior_entropy.item())


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


checkpoint = torch.load('drive/MyDrive/Colab Notebooks/ddit_circle.pt')


# In[16]:


model = DIT(config=config,vocab_size=V)
model.train()
model = model.to('cuda').to(torch.bfloat16)
# model.load_state_dict(checkpoint['model'])


# In[17]:


# heuristic setting for lambda
lamb = 0.06


# In[18]:


opt = torch.optim.Adam(model.parameters(), 3e-4, (0.9,0.999), 1e-8) # 3e-4
scheduler = transformers.get_constant_schedule_with_warmup(opt, 1000)


# In[19]:


with torch.device('cuda'):
  saved_losses = []
  avg_loss = 0.0
  avg_loss_denom = 0.0
  lamb_est_num = 0.0
  lamb_est_den = 0.0
  progress_bar = tqdm(range(nbatches))
  for ibatch in progress_bar:
      opt.zero_grad()
      targets = train_batches[ibatch,:].cuda().to(torch.int64)
      ts = -((torch.arange(batchsize) + torch.rand(batchsize,dtype=torch.float64)) / batchsize).log() / lamb
      weights = (lamb*ts).exp()/lamb
      assert(ts.isfinite().all())
      ts = ts[:,None].expand(targets.shape).reshape(-1)
      weights = weights[:,None].expand(targets.shape).reshape(-1)
      targets = targets.reshape(-1)
      (rhos, thetas) = bbridge(ts)
      thetas = thetas + (targets.to(torch.float64) + 0.5) * (2 * torch.pi / V)
      assert(rhos.isfinite().all())
      assert(thetas.isfinite().all())
      z = tf_embed(rhos, thetas, model.vocab_embed.embedding).to(torch.bfloat16)
      z0 = tf_embed(torch.zeros(1,dtype=torch.float64), torch.zeros(1,dtype=torch.float64), model.vocab_embed.embedding).to(torch.bfloat16)
      logits = model(z.reshape(batchsize,seqlen,config['model']['hidden_size']))
      logits = logits - model(z0[None,:,:].expand(1,seqlen,config['model']['hidden_size']))
      logits = logits + initial_bias[None,None,:]
      logits = logits.reshape(-1,V)
      losses = bridge_loss(logits, targets, rhos, thetas)
      losses = losses * weights
      l = losses.mean() # the actual minibatch loss
      lamb_est_num = (lamb_est_num * 0.99) + losses.square().mean().detach().item()
      lamb_est_den = (lamb_est_den * 0.99) + (ts*losses.square()).mean().detach().item()
      avg_loss = (avg_loss * 0.99) + l.item()
      avg_loss_denom = (avg_loss_denom * 0.99) + 1.0
      saved_losses.append(avg_loss/avg_loss_denom)
      progress_bar.set_description(f"{l.item()} [avg={avg_loss/avg_loss_denom}] [lamb={lamb_est_num/lamb_est_den}]")
      l.backward()
      opt.step()
      scheduler.step()
      if ibatch % 100 == 99:
          pyplot.figure()
          pyplot.plot(saved_losses)
          pyplot.plot(checkpoint['saved_losses'][:len(saved_losses)])
          pyplot.xlabel('step')
          pyplot.ylabel('NELBO')
          pyplot.title('Hyperbolic Diffusion Model on Wikitext')
          pyplot.savefig('drive/MyDrive/Colab Notebooks/fig_in_training_circle_small.png')
          pyplot.show()


# In[ ]:


pyplot.plot(saved_losses)


# In[ ]:


sd = model.state_dict()
# torch.save({"model": sd, "saved_losses": saved_losses},'drive/MyDrive/Colab Notebooks/ddit_circle_randperm.pt')


# In[ ]:


# opt.param_groups[0].keys()


# In[ ]:


# sd = model.state_dict()
# torch.save({"model": sd, "saved_losses": saved_losses},'drive/MyDrive/Colab Notebooks/ddit_plane_d2_latest_34481.pt')


# In[ ]:


len(saved_losses)


# In[ ]:




