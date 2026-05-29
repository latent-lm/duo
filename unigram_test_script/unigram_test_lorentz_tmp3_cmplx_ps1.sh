#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Lorentz Cartesian Optimal
# Unif
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=lorentz_cartesian_opt_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=opt \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=unif \
  +proposal_exp_rate=1.0 \
  +loss_geometry=lorentz_cartesian \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42 \
# Exp
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=lorentz_cartesian_opt_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=opt \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=stratified_exp \
  +proposal_exp_rate=1.0,0.1,0.01,0.2,0.3,0.4,0.8,0.5,2.0 \
  +loss_geometry=lorentz_cartesian \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42
# ==================
# Poincare Polar Optimal
# Unif
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=opt \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=unif \
  +proposal_exp_rate=1.0 \
  +loss_geometry=poincare_polar \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42 \
# Exp
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=opt \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=stratified_exp \
  +proposal_exp_rate=1.0,0.1,0.01,0.2,0.3,0.4,0.5,0.8,2.0 \
  +loss_geometry=poincare_polar \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42

python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=opt \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=stratified_exp \
  +proposal_exp_rate=2.0 \
  +loss_geometry=poincare_polar \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=43
# ===========================

# Lorentz Cartesian Trainable
# Unif
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=lorentz_cartesian_tnb_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=tnb \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=unif \
  +proposal_exp_rate=1.0 \
  +loss_geometry=lorentz_cartesian \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42
# Exp
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=lorentz_cartesian_tnb_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=tnb \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=stratified_exp \
  +proposal_exp_rate=1.0,0.1,0.01,0.2,0.3,0.4,0.5,2.0 \
  +loss_geometry=lorentz_cartesian \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42
# ===========================

# Poincare Polar Trainable
# Unif
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=poincare_polar_tnb_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=tnb \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=unif \
  +proposal_exp_rate=1.0 \
  +loss_geometry=poincare_polar \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42
# Exp
python unigram/unigram_test2_tmp3.py -m \
  hydra.job.chdir=true \
  hydra.sweep.dir=outputs/unigram_test2 \
  hydra.sweep.subdir=poincare_polar_tnb_tmp3_rlog_newd2_rot_ts4000000_cmplx_ps1 \
  +mode=tnb \
  +vocab_size=10 \
  +max_steps=20000 \
  +lr=1e-5 \
  '+ps=[0.11,0.10,0.10,0.11,0.11,0.10,0.10,0.10,0.09,0.08]' \
  +proposal_type=stratified_exp \
  +proposal_exp_rate=1.0,0.1,0.01,0.2,0.3,0.4,0.5,2.0 \
  +loss_geometry=poincare_polar \
  "+rotate_emb=True" \
  +hidden_size=128 \
  +depth=3 \
  +hyper_T=1000 \
  +hyper_dt=0.01 \
  +loss_plot_ma_window=50 \
  +seed=42
