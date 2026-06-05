import os
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

class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
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

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
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

        embeds = torch.randn(self.vocab_size,config.model.hyper_dim)
        embeds = embeds / embeds.square().sum(-1,keepdim=True).sqrt()
        self.hyper_embeds = torch.nn.Parameter(embeds)

    @torch.no_grad()
    def normalize_embeds(self):
        self.hyper_embeds.copy_(self.hyper_embeds / self.hyper_embeds.square().sum(-1,keepdim=True).sqrt())

    def forward(self, x):
        # x = self.vocab_embed(indices)
        rotary_cos_sin = tuple(r.to(x.dtype) for r in self.rotary_emb(x.shape[1], x.device))
        for block in self.blocks:
            x = block(x, rotary_cos_sin)
        return self.output_layer(x)


# x is a matrix of embedded examples on the boundary of the sphere
# t is the time we want to sample the bridge process at
# can backprop through x but not t
# this is a very dumb way of sampling by simulating the SDE
def noise_train_example(x, t, nsteps):
    assert(len(x.shape)==2)
    d = x.shape[-1]
    N = x.shape[0]
    delta = (t / nsteps).detach()
    if len(delta.shape) == 0:
        delta = torch.ones(N) * delta
    delta = delta[:,None]
    x64 = x.detach().to(torch.float64)
    z = torch.zeros_like(x64)
    for _ in range(nsteps):
        sigma = 1 - z.square().sum(1,keepdim=True)
        dW = (delta.sqrt() * sigma / 2) * torch.randn(z.shape,dtype=torch.float64)
        ito = -(delta * d * sigma / 4) * z
        z_to_x = x64 - z
        drift = ((delta * (d-1) * sigma**2 / 2) / (z_to_x).square().sum(1,keepdim=True)) * z_to_x
        z += dW + ito + drift
        znorm = z.square().sum(1,keepdim=True).sqrt() + 1e-32
        znewnorm = znorm.clamp(max=0.999)
        z = z * (znewnorm / znorm)
    y = z.to(torch.float32)
    r = y.square().sum(1,keepdim=True).sqrt()
    y = y / r
    xdet = x.detach()
    epsilon = (y * xdet).sum(1,keepdim=True)
    return r * (epsilon * x + y * (xdet * x).sum(1,keepdim=True) - xdet * (y * x).sum(1,keepdim=True))

# tokenids  (B,N)   int64
# embeds    (V,d)   float32
# t         (B,)    float32
# nsteps            int
def noise_train_sequence(tokenids, embeds, t, nsteps):
    assert(len(embeds.shape)==2)
    assert(len(tokenids.shape)==2)
    assert(len(t.shape)==1)
    (B,N) = tokenids.shape
    assert(t.shape[0] == B)
    return noise_train_example(embeds[tokenids.reshape(-1),:], t[:,None].expand(B,N).reshape(-1), nsteps).reshape(B,N,-1)

# logits    (B,N,V)   float32
# targets   (B,N)     int64
# z         (B,N,d)   float32
# embeds    (V,d)     float32
# weights   (B,)
# ts        (B,)
def loss(logits, targets, z, embeds, weights, ts):
    # embeds must be normalized by some earlier step
    assert(len(logits.shape)==3)
    assert(len(targets.shape)==2)
    assert(len(z.shape)==3)
    assert(len(embeds.shape)==2)
    assert(len(weights.shape)==1)
    (B,N,V) = logits.shape
    (V,d) = embeds.shape
    assert((B,N) == targets.shape)
    assert((B,N,d) == z.shape)
    assert((B,) == weights.shape)
    # horosphere distances from z to targets (B,N,V)
    exp_horosphere_dists = ((1 - z.square().sum(-1,keepdim=True))
                               / (z.square().sum(-1,keepdim=True) + 1 - 2 * z @ embeds.t()))
    horosphere_dists = exp_horosphere_dists.log()
    # posterior distribution after adding logits (B,N,V)
    mu = ((d-1) * horosphere_dists + logits).softmax(-1)
    mu = mu - torch.nn.functional.one_hot(targets,V).to(torch.float32)
    # evaluate the guess - target
    error = (mu * exp_horosphere_dists) @ embeds - (mu * exp_horosphere_dists).sum(-1,keepdim=True) * z
    tlosses = ((d-1)*error).square().sum(-1) * weights[:,None]
    lamb_est = (tlosses.square().sum() / (ts[:,None]*tlosses.square()).sum()).detach().item()
    # pyplot.plot(tlosses.detach())
    return (tlosses.sum() / (2*N*B), lamb_est)

# from huggingface_hub import notebook_login

# notebook_login()

# tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tol = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")

# trainset = torch.load("drive/MyDrive/Colab Notebooks/trainset_llama_mapped.pt", weights_only=True)
trainset = torch.load("trainset_gpt2_wikitext2_ml256_bs32.pt", weights_only=True)

(nbatches,batchsize,seqlen) = trainset['token_batches'].shape
V = trainset['initial_bias'].shape[0]
d = trainset['A_emb'].shape[1]

config = yaml.safe_load('''name: tiny
type: ddit
hidden_size: 768
cond_dim: 128
length: 1024
n_blocks: 8
n_heads: 8
scale_by_sigma: True
dropout: 0.1
tie_word_embeddings: False''')
config['hyper_dim'] = d
config = {'model':config}

model = DIT(config=config,vocab_size=V)
model.train()
model = model.to('cuda').to(torch.bfloat16)
model.hyper_embeds = torch.nn.Parameter(model.hyper_embeds.to(torch.float32))

with torch.no_grad():
  model.hyper_embeds.copy_(trainset['A_emb'])
  model.vocab_embed.embedding.copy_(trainset['E_emb'])

# heuristic setting for lambda
lamb_factor = 0.25
lamb = lamb_factor * d**2/8
lamb_decay_exp = 0.05

initial_bias = trainset['initial_bias'].cuda()
initial_probs = initial_bias.exp().to(torch.bfloat16)
prior_entropy = torch.special.entr(initial_probs).sum()
print(prior_entropy.item())

# opt = torch.optim.Adam(model.parameters(), 3e-4, (0.9,0.999), 1e-8)
opt = torch.optim.Adam(model.parameters(), 3e-5, (0.9,0.999), 1e-8)
# scheduler = transformers.get_constant_schedule_with_warmup(opt, 1000) #2500)

saved_losses = []
avg_loss = 0.0
avg_loss_denom = 0.0
progress_bar = tqdm(range(len(saved_losses),nbatches))
os.makedirs("hyperdiff_results/images", exist_ok=True)
for ibatch in progress_bar:
    opt.zero_grad()
    targets = trainset['token_batches'][ibatch,:].cuda().to(torch.int64)
    noise_euler_steps = 128
    ts = (-((torch.arange(batchsize) + torch.rand(batchsize)) / batchsize).clamp(min=1e-5,max=0.999).log() / lamb).cuda()
    weights = (lamb*ts).exp()/lamb
    embeds_normalized = model.hyper_embeds / model.hyper_embeds.square().sum(-1,keepdim=True).sqrt()
    with torch.device('cuda'):
        z = noise_train_sequence(targets, embeds_normalized, ts, noise_euler_steps)
    assert(z.isfinite().all())
    # get horosphere dist
    exp_horosphere_dists = ((1 - z.square().sum(-1,keepdim=True))
                               / (z.square().sum(-1,keepdim=True) + 1 - 2 * z @ model.hyper_embeds.t()))
    horosphere_dists = exp_horosphere_dists.log()
    # prior distribution before adding logits (B,N,V)
    mu = ((d-1) * horosphere_dists + initial_bias).softmax(-1).to(torch.bfloat16)
    zemb = mu @ model.vocab_embed.embedding
    zemb_norms = mu @ (model.vocab_embed.embedding.float().square().sum(-1,keepdim=True).sqrt().to(torch.bfloat16)) # (B,N,1)
    zemb = zemb * (zemb_norms / zemb.float().square().sum(-1,keepdim=True).sqrt().to(torch.bfloat16))
    assert(zemb.isfinite().all())
    zemb0 = initial_probs.view(1,1,-1) @ model.vocab_embed.embedding
    zemb0_norms = initial_probs.view(1,1,-1) @ (model.vocab_embed.embedding.float().square().sum(-1,keepdim=True).sqrt().to(torch.bfloat16)) # (B,N,1)
    zemb0 = zemb0 * (zemb0_norms / zemb0.float().square().sum(-1,keepdim=True).sqrt().to(torch.bfloat16))
    assert(zemb0.isfinite().all())
    logits = model(zemb) - model(zemb0) + initial_bias
    assert(logits.isfinite().all())
    (l, lamb_est) = loss(logits, targets, z, embeds_normalized, weights, ts)
    (lvr, _) = loss(initial_bias[None,None,:].expand(logits.shape), targets, z, embeds_normalized, weights, ts)
    l = l - lvr + prior_entropy
    lamb = lamb * (lamb_factor*lamb_est/lamb)**lamb_decay_exp
    avg_loss = (avg_loss * 0.99) + 0.01 * l.item()
    avg_loss_denom = (avg_loss_denom * 0.99) + 0.01
    saved_losses.append(avg_loss/avg_loss_denom)
    progress_bar.set_description(f"{l.item()} [avg={avg_loss/avg_loss_denom}] [lambda={lamb}]")
    l.backward()
    opt.step()
    # scheduler.step()
    model.normalize_embeds()
    if ibatch % 200 == 0:
        pyplot.figure()
        pyplot.plot(saved_losses)
        pyplot.xlabel('step')
        pyplot.ylabel('NELBO')
        pyplot.title('Hyperbolic Diffusion Model on Wikitext')
        pyplot.savefig('hyperdiff_results/images/fig_in_training.png')
        pyplot.show()


