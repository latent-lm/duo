#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=hs${hidden_size}_d${depth}_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}' \
#   +folder=artifacts \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   +proposal_type=unif \
#   +proposal_exp_rate=1.0 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1000 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50


python unigram/unigram_test2.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  'hydra.sweep.subdir=hs${hidden_size}_d${depth}_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}' \
  +folder=artifacts \
  +max_steps=20000 \
  +lr=1e-5 \
  +proposal_type=exp \
  +proposal_exp_rate=0.1,0.01,0.2 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50
