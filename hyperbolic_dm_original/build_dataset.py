"""Build training datasets for Hyperbolic Diffusion Language Models.

Produces a trainset dict with:
  - token_batches: (n_batches, batch_size, seq_len) int64 tensor
  - initial_bias:  (V,) float32 tensor of log token frequencies
  - A_emb:         (V, d) float32 tensor of L2-normalized embeddings
  - E_emb:         (V, d) float32 tensor of raw embeddings
  - vocab_map:     (V,) int64 tensor mapping compact→original IDs
                   (only present when map_vocab=True)

When map_vocab=True (default), V is the number of unique tokens
that actually appear in the dataset, not the full model vocab.
E.g., Llama-3.1 has 128k vocab but wikitext2 uses only ~15k
unique tokens. Mapping reduces embeddings from (128k, 4096) to
(15k, 4096) and the model output layer from 128k to 15k classes.

Compatible with HuggingFace datasets and transformers APIs.

Usage:
  # Build from a HuggingFace dataset + model
  builder = HyperbolicTrainsetBuilder(
      model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
      dataset_name='wikitext',
      dataset_config='wikitext-2-raw-v1',
  )
  trainset = builder.build()
  builder.save(trainset, 'trainset_llama_wikitext2.pt')

  # Load and use as a torch Dataset
  ds = HyperbolicTokenDataset('trainset_llama_wikitext2.pt')
  loader = torch.utils.data.DataLoader(ds, batch_size=64)

CLI:
  python build_dataset.py \\
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \\
    --seq-len 256 --batch-size 64 \\
    --output trainset_llama_wikitext2.pt
"""

import argparse
import itertools
from typing import Optional

import datasets
import torch
import torch.utils.data
import transformers
from tqdm import tqdm


class HyperbolicTrainsetBuilder:
  """Build a trainset dict from a HuggingFace dataset and model.

  The model is used only to extract the input embedding matrix.
  The dataset is tokenized, chunked, and token frequencies are
  computed for the initial_bias prior.
  """

  def __init__(
    self,
    model_name_or_path: str,
    dataset_name: str = 'wikitext',
    dataset_config: Optional[str] = 'wikitext-2-raw-v1',
    split: str = 'train',
    seq_len: int = 256,
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    text_column: Optional[str] = None,
    map_vocab: bool = False,
  ):
    """
    Args:
      model_name_or_path: HuggingFace model ID or local path.
        Used to load the tokenizer and extract embeddings.
      dataset_name: HuggingFace dataset name (e.g. 'wikitext').
      dataset_config: Dataset config (e.g. 'wikitext-2-raw-v1').
        Pass None for datasets without configs.
      split: Dataset split to use.
      seq_len: Sequence length for each example.
      batch_size: Batch size (examples are grouped into batches
        in the saved tensor for the reference notebook format).
      cache_dir: Cache directory for HuggingFace downloads.
      text_column: Name of the text column. Auto-detected if
        None (tries 'text', 'sentence', 'article').
      map_vocab: If True, reduce vocabulary to only tokens that
        appear in the dataset. Remaps token IDs to a compact
        range [0, V_mapped) and slices embeddings accordingly.
        Essential for large-vocab models (e.g. Llama 128k)
        on small datasets (e.g. wikitext2 ~15k unique tokens).
    """
    self.model_name = model_name_or_path
    self.dataset_name = dataset_name
    self.dataset_config = dataset_config
    self.split = split
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.cache_dir = cache_dir
    self.text_column = text_column
    self.map_vocab = map_vocab

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_name_or_path, cache_dir=cache_dir)
    # Match DUO dataloader: adds [PAD] as a new special token
    if self.tokenizer.pad_token is None:
      self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  def _detect_text_column(self, ds):
    """Auto-detect the text column in the dataset."""
    if self.text_column is not None:
      return self.text_column
    for col in ('text', 'sentence', 'article', 'content'):
      if col in ds.column_names:
        return col
    raise ValueError(
      f'Cannot detect text column. Columns: {ds.column_names}. '
      f'Pass text_column= explicitly.')

  def _load_and_tokenize(self):
    """Load dataset, tokenize all text into a flat token stream.

    Returns:
      1D int64 tensor of all token IDs concatenated.
    """
    kwargs = {'cache_dir': self.cache_dir}
    if self.dataset_config is not None:
      ds = datasets.load_dataset(
        self.dataset_name, self.dataset_config,
        split=self.split, **kwargs)
    else:
      ds = datasets.load_dataset(
        self.dataset_name, split=self.split, **kwargs)

    text_col = self._detect_text_column(ds)
    print(f'Using text column: {text_col!r}')

    all_tokens = []
    for example in tqdm(ds, desc='Tokenizing'):
      text = example[text_col]
      if text and text.strip():
        tokens = self.tokenizer.encode(
          text, add_special_tokens=False)
        all_tokens.append(tokens)

    flat = list(itertools.chain.from_iterable(all_tokens))
    print(f'Total tokens: {len(flat):,}')
    return torch.tensor(flat, dtype=torch.long)

  def _chunk_into_batches(self, all_tokens):
    """Reshape flat token stream into batched tensor.

    Args:
      all_tokens: 1D tensor of token IDs.

    Returns:
      (n_batches, batch_size, seq_len) int64 tensor.
      Remainder tokens that don't fill a complete batch
      are dropped.
    """
    total = len(all_tokens)
    n_examples = total // self.seq_len
    all_tokens = all_tokens[:n_examples * self.seq_len]
    all_tokens = all_tokens.reshape(n_examples, self.seq_len)
    n_batches = n_examples // self.batch_size
    token_batches = all_tokens[:n_batches * self.batch_size]
    token_batches = token_batches.reshape(
      n_batches, self.batch_size, self.seq_len)
    return token_batches

  def _build_vocab_map(self, all_tokens):
    """Build a compact vocab mapping from unique tokens.

    Args:
      all_tokens: 1D tensor of all token IDs (original IDs).

    Returns:
      (vocab_map, remap_table):
        vocab_map: (V_mapped,) int64 tensor, compact→original IDs.
        remap_table: (full_vocab,) int64 tensor, original→compact.
          Unmapped tokens get -1.
    """
    unique_ids = all_tokens.unique().sort().values  # sorted original IDs
    V_mapped = len(unique_ids)
    full_vocab = len(self.tokenizer)
    remap_table = torch.full((full_vocab,), -1, dtype=torch.long)
    remap_table[unique_ids] = torch.arange(V_mapped, dtype=torch.long)
    print(f'Vocab mapping: {full_vocab:,} → {V_mapped:,} '
          f'({V_mapped/full_vocab:.1%} of full vocab)')
    return unique_ids, remap_table

  def _compute_initial_bias(self, all_tokens, vocab_size):
    """Compute log token frequencies as the prior bias.

    Applies Laplace smoothing (add-one) so that unseen tokens
    get a small nonzero probability rather than log(0).

    Args:
      all_tokens: 1D tensor of all token IDs.
      vocab_size: Size of the vocabulary.

    Returns:
      (vocab_size,) float32 tensor of log-probabilities.
    """
    counts = torch.bincount(
      all_tokens, minlength=vocab_size).float()
    counts = counts + 1.0  # Laplace smoothing
    log_probs = (counts / counts.sum()).log()
    return log_probs

  def _extract_embeddings(self):
    """Extract the input embedding matrix from the model.

    Uses the standard HuggingFace `get_input_embeddings()` API
    which works across model families (GPT-2, Llama, BERT, etc.).

    Returns:
      (A_emb, E_emb): L2-normalized and raw embedding matrices,
        both (vocab_size, embed_dim) float32.
    """
    print(f'Loading model: {self.model_name} '
          f'(embedding extraction only)...')
    model = transformers.AutoModel.from_pretrained(
      self.model_name,
      cache_dir=self.cache_dir,
      torch_dtype=torch.float32)
    embed_layer = model.get_input_embeddings()
    E_emb = embed_layer.weight.detach().clone().float()
    # Pad for any extra special tokens added to the tokenizer
    V_model = E_emb.shape[0]
    V_tok = len(self.tokenizer)
    if V_tok > V_model:
      # Use small random vectors (not zeros) so that
      # A_emb normalization doesn't produce NaN (0/0).
      pad = torch.randn(V_tok - V_model, E_emb.shape[1]) * 0.01
      E_emb = torch.cat([E_emb, pad], dim=0)
    A_emb = E_emb / torch.clamp(E_emb.norm(dim=-1, keepdim=True), min=1e-6)
    e_norm = E_emb.norm(dim=-1, keepdim=True)
    print(f"any(e_norm == 0): {torch.any(e_norm == 0)}")
    print(f"any(A_emb == NaN): {torch.any(torch.isnan(A_emb))}")
    del model
    torch.cuda.empty_cache()
    print(f'Embedding dim: {E_emb.shape[1]}')
    return A_emb, E_emb

  def build(self):
    """Build the full trainset dictionary.

    Returns:
      dict with keys:
        token_batches: (n_batches, batch_size, seq_len) int64
        initial_bias:  (V,) float32
        A_emb:         (V, embed_dim) float32
        E_emb:         (V, embed_dim) float32
        vocab_map:     (V,) int64 (only when map_vocab=True)
    """
    all_tokens = self._load_and_tokenize()
    A_emb, E_emb = self._extract_embeddings()

    if self.map_vocab:
      vocab_map, remap_table = self._build_vocab_map(all_tokens)
      # Remap token IDs to compact range
      all_tokens = remap_table[all_tokens]
      # Slice embeddings to mapped vocab only
      A_emb = A_emb[vocab_map]
      E_emb = E_emb[vocab_map]
      vocab_size = len(vocab_map)
    else:
      vocab_map = None
      vocab_size = len(self.tokenizer)

    token_batches = self._chunk_into_batches(all_tokens)
    print(f'token_batches: {token_batches.shape} '
          f'(n_batches, batch_size, seq_len)')

    initial_bias = self._compute_initial_bias(
      all_tokens, vocab_size)
    print(f'initial_bias: {initial_bias.shape}, '
          f'min={initial_bias.min():.2f}, '
          f'max={initial_bias.max():.2f}')
    print(f'A_emb: {A_emb.shape}, E_emb: {E_emb.shape}')

    trainset = {
      'token_batches': token_batches,
      'initial_bias': initial_bias,
      'A_emb': A_emb,
      'E_emb': E_emb,
    }
    if vocab_map is not None:
      trainset['vocab_map'] = vocab_map
    return trainset

  @staticmethod
  def save(trainset, path):
    """Save trainset dict to a .pt file."""
    torch.save(trainset, path)
    print(f'Saved trainset to {path}')

  @staticmethod
  def load(path):
    """Load trainset dict from a .pt file."""
    return torch.load(path, weights_only=True)


class HyperbolicTokenDataset(torch.utils.data.Dataset):
  """Torch Dataset wrapping a pre-built trainset .pt file.

  Flattens (n_batches, batch_size, seq_len) into individual
  (seq_len,) examples, compatible with standard DataLoaders
  and the DUO codebase's training pipeline.

  Also exposes initial_bias, A_emb, E_emb as attributes
  for model initialization.

  Usage:
    ds = HyperbolicTokenDataset('trainset.pt')
    print(ds.vocab_size, ds.embed_dim)
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    for batch in loader:
        input_ids = batch['input_ids']       # (B, seq_len)
        attn_mask = batch['attention_mask']   # (B, seq_len)
  """

  def __init__(self, trainset_or_path):
    """
    Args:
      trainset_or_path: Either a trainset dict or a path to
        a .pt file produced by HyperbolicTrainsetBuilder.
    """
    if isinstance(trainset_or_path, (str, bytes)):
      trainset = torch.load(
        trainset_or_path, weights_only=True)
    else:
      trainset = trainset_or_path

    tb = trainset['token_batches']
    # Flatten (n_batches, batch_size, seq_len) → (N, seq_len)
    self.tokens = tb.reshape(-1, tb.shape[-1])
    self.initial_bias = trainset['initial_bias']
    self.A_emb = trainset['A_emb']
    self.E_emb = trainset['E_emb']
    self.vocab_map = trainset.get('vocab_map', None)

  @property
  def vocab_size(self):
    return self.initial_bias.shape[0]

  @property
  def embed_dim(self):
    return self.A_emb.shape[1]

  def __len__(self):
    return self.tokens.shape[0]

  def __getitem__(self, idx):
    input_ids = self.tokens[idx]
    return {
      'input_ids': input_ids,
      'attention_mask': torch.ones_like(input_ids),
    }


def main():
  parser = argparse.ArgumentParser(
    description='Build hyperbolic diffusion trainset from '
                'HuggingFace dataset + model')
  parser.add_argument(
    '--model', type=str, required=True,
    help='HuggingFace model name or path '
         '(e.g. meta-llama/Meta-Llama-3.1-8B-Instruct, gpt2)')
  parser.add_argument(
    '--dataset', type=str, default='wikitext',
    help='HuggingFace dataset name')
  parser.add_argument(
    '--dataset-config', type=str, default='wikitext-2-raw-v1',
    help='Dataset config name (pass "none" to skip)')
  parser.add_argument(
    '--split', type=str, default='train')
  parser.add_argument(
    '--seq-len', type=int, default=256)
  parser.add_argument(
    '--batch-size', type=int, default=64)
  parser.add_argument(
    '--cache-dir', type=str, default=None)
  parser.add_argument(
    '--text-column', type=str, default=None)
  parser.add_argument(
    '--map-vocab', action='store_true', default=False,
    help='Map vocab to only tokens in dataset')
  parser.add_argument(
    '--no-map-vocab', dest='map_vocab', action='store_false',
    help='Keep full model vocabulary')
  parser.add_argument(
    '--output', '-o', type=str, required=True,
    help='Output .pt file path')
  args = parser.parse_args()

  config = args.dataset_config
  if config and config.lower() == 'none':
    config = None

  builder = HyperbolicTrainsetBuilder(
    model_name_or_path=args.model,
    dataset_name=args.dataset,
    dataset_config=config,
    split=args.split,
    seq_len=args.seq_len,
    batch_size=args.batch_size,
    cache_dir=args.cache_dir,
    text_column=args.text_column,
    map_vocab=args.map_vocab,
  )
  trainset = builder.build()
  builder.save(trainset, args.output)


if __name__ == '__main__':
  main()
