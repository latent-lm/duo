"""Generate sanity-check dataset for HyperbolicDUO entropy convergence test.

Dataset: V=2 tokens (A=0, B=1), P(A)=0.8, P(B)=0.2, context_length=1.
Ground truth entropy: -0.8*ln(0.8) - 0.2*ln(0.2) ~ 0.5004 nats.

Outputs:
  1. HuggingFace dataset saved to disk (train + validation splits)
  2. Trainset .pt file with embeddings for HyperbolicDUO
     Keys: A_emb (V, d), E_emb (V, d), initial_bias (V,)

Usage:
  python sanity_dataset.py --d 16
"""
import argparse
import math
import os

import datasets
import numpy as np
import torch


def generate_sanity_data(size, context_length, prob_a=0.8,
                         seed=42):
    """Generate token sequences with exact prob_a fraction of A tokens."""
    n_a = int(round(size * prob_a))
    data = np.array([0] * n_a + [1] * (size - n_a),
                     dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(data)
    return data.reshape(size, context_length)


def make_hf_dataset(data):
    """Wrap numpy array as a HuggingFace Dataset."""
    input_ids = torch.from_numpy(data)
    attention_mask = torch.ones_like(input_ids)
    ds = datasets.Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    })
    ds.set_format(type='torch')
    return ds


def make_trainset_pt(vocab_size, d, prob_a=0.8):
    """Create the trainset .pt file expected by HyperbolicDUO.

    Contains:
        A_emb: (V, d) embeddings on the unit sphere.
        E_emb: (V, d) backbone input embeddings (same as A_emb).
        initial_bias: (V,) log-prior bias.
    """
    torch.manual_seed(0)
    embeds = torch.randn(vocab_size, d)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)

    probs = torch.tensor([prob_a, 1 - prob_a])
    initial_bias = probs.log()

    return {
        'A_emb': embeds,
        'E_emb': embeds.clone(),
        'initial_bias': initial_bias,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
                        default='/share/thickstun/sychou/.cache'
                                '/huggingface/datasets')
    parser.add_argument('--trainset_dir', type=str,
                        default='/share/thickstun/sychou/workspace'
                                '/research/duo/hyper_dataset'
                                '/sanity')
    parser.add_argument('--d', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--valid_size', type=int, default=100)
    parser.add_argument('--context_length', type=int, default=1)
    parser.add_argument('--prob_a', type=float, default=0.8)
    args = parser.parse_args()

    vocab_size = 2
    entropy = -(args.prob_a * math.log(args.prob_a)
                + (1 - args.prob_a) * math.log(1 - args.prob_a))
    print(f"Vocab size: {vocab_size}")
    print(f"P(A)={args.prob_a}, P(B)={1-args.prob_a}")
    print(f"Ground truth entropy: {entropy:.4f} nats")
    print()

    # ---- HuggingFace dataset ----
    train_data = generate_sanity_data(
        args.train_size, args.context_length,
        prob_a=args.prob_a, seed=42)
    valid_data = generate_sanity_data(
        args.valid_size, args.context_length,
        prob_a=args.prob_a, seed=41)

    train_ds = make_hf_dataset(train_data)
    valid_ds = make_hf_dataset(valid_data)

    ds_name = (f'sanity_cl{args.context_length}'
               f'_vs{vocab_size}_sz{args.train_size}')
    train_path = os.path.join(
        args.save_dir,
        f'{ds_name}_train_bs{args.context_length}_wrapped.dat')
    valid_path = os.path.join(
        args.save_dir,
        f'{ds_name}_validation_bs{args.context_length}_wrapped.dat')

    train_ds.save_to_disk(train_path)
    valid_ds.save_to_disk(valid_path)
    print(f"Train dataset saved to: {train_path}")
    print(f"Valid dataset saved to: {valid_path}")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Valid shape: {valid_data.shape}")
    print()

    # ---- Trainset .pt for HyperbolicDUO ----
    os.makedirs(args.trainset_dir, exist_ok=True)
    trainset = make_trainset_pt(vocab_size, args.d,
                                prob_a=args.prob_a)
    pt_path = os.path.join(
        args.trainset_dir,
        f'trainset_sanity_cl{args.context_length}'
        f'_vs{vocab_size}_d{args.d}.pt')
    torch.save(trainset, pt_path)
    print(f"Trainset .pt saved to: {pt_path}")
    print(f"  A_emb shape: {trainset['A_emb'].shape}")
    print(f"  E_emb shape: {trainset['E_emb'].shape}")
    print(f"  initial_bias: {trainset['initial_bias']}")
    print()
    print(f"Entropy target: {entropy:.4f} nats")


if __name__ == '__main__':
    main()
