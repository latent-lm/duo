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


# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=hs${hidden_size}_d${depth}_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}' \
#   +folder=artifacts \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   +proposal_type=exp \
#   +proposal_exp_rate=0.1,0.01,0.2 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1000 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50

# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}' \
#   +folder=artifacts \
#   +vocab_size=2 \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   +ps=0.8 \
#   +proposal_type=unif \
#   +proposal_exp_rate=1.0 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1000 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50


# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}' \
#   +folder=artifacts \
#   +vocab_size=10 \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
#   +proposal_type=exp \
#   +proposal_exp_rate=1.0,2.0,0.1,0.01 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1e6 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50

# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}' \
#   +folder=artifacts \
#   +vocab_size=10 \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
#   +proposal_type=exp \
#   +proposal_exp_rate=1.0,2.0,0.1,0.01 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1e12 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50

# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_opt' \
#   +folder=artifacts \
#   +vocab_size=10 \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
#   +proposal_type=exp \
#   +proposal_exp_rate=1.0 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1e12 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50

# =====================================================

# python unigram/unigram_test2.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_polar_new' \
#   +folder=artifacts \
#   +vocab_size=2 \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   '+ps=[0.8,0.2]' \
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
  'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_polar_new' \
  +folder=artifacts \
  +vocab_size=2 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.8,0.2]' \
  +proposal_type=exp \
  +proposal_exp_rate=0.01,0.1,0.5,1.0,2.0,3.0 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1e12 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50

python unigram/unigram_test2.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_polar_new' \
  +folder=artifacts \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
  +proposal_type=unif \
  +proposal_exp_rate=1.0 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50

python unigram/unigram_test2.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_polar_new' \
  +folder=artifacts \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
  +proposal_type=exp \
  +proposal_exp_rate=0.01,0.1,0.5,1.0,2.0,3.0 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1e12 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50

# python unigram/unigram_test2_cart.py -m \
#   hydra.job.chdir=true \
#   hydra.sweep.dir=outputs/unigram_test2 \
#   'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_cart_new' \
#   +folder=artifacts \
#   +vocab_size=2 \
#   +max_steps=20000 \
#   +lr=1e-5 \
#   '+ps=[0.8,0.2]' \
#   +proposal_type=unif \
#   +proposal_exp_rate=1.0 \
#   +hidden_size=128 \
#   +depth=3 \
#   +hyper_T=1000 \
#   +hyper_dt=0.01 \
#   +loss_plot_ma_window=50

python unigram/unigram_test2_cart.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_cart_new' \
  +folder=artifacts \
  +vocab_size=2 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.8,0.2]' \
  +proposal_type=exp \
  +proposal_exp_rate=0.01,0.1,0.5,1.0,2.0,3.0 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1e12 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50

python unigram/unigram_test2_cart.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_cart_new' \
  +folder=artifacts \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
  +proposal_type=unif \
  +proposal_exp_rate=1.0 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50

python unigram/unigram_test2_cart.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  'hydra.sweep.subdir=unigram_test2_losses_sp${max_steps}_lr${lr}_pt${proposal_type}_per${proposal_exp_rate}_ht${hyper_T}_vs${vocab_size}_cart_new' \
  +folder=artifacts \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]' \
  +proposal_type=exp \
  +proposal_exp_rate=0.01,0.1,0.5,1.0,2.0,3.0 \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1e12 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50
