#!/bin/bash
# Entropy convergence sanity test for HyperbolicDUO
#
# Dataset: V=2 tokens, P(A)=0.8, P(B)=0.2, context_length=1
# Ground truth entropy: -0.8*ln(0.8) - 0.2*ln(0.2) = 0.5004 nats
#
# Step 1: Generate the dataset and embeddings
# Step 2: Train HyperbolicDUO and verify NLL -> 0.5004

set -e

D=32  # embedding dimension (must match model hidden_size)

echo "=== Step 1: Generate sanity dataset ==="
python sanity_dataset.py --d $D

echo ""
echo "=== Step 2: Train HyperbolicDUO ==="
echo "Target entropy: 0.5004 nats"
echo ""

python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=sanity_cl1_vs2_sz100 \
  wandb.name=sanity-hyper-duo-entropy \
  model=sanity \
  algo=hyperbolic_duo \
  algo.trainset_path=/share/thickstun/sychou/workspace/research/duo/hyper_dataset/sanity/trainset_sanity_cl1_vs2_d${D}.pt \
  algo.hyper_loss_type=ce \
  model.length=1
