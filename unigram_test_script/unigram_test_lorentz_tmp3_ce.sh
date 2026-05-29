#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# if (($# > 0)); then
#   seeds=("$@")
# else
#   read -r -a seeds <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62}"
# fi

seed=42
PS='[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]'

run_tmp4() {
  local seed="$1"
  local loss_geometry="$2"
  local sweep_name="$3"
  local proposal_type="$4"
  local proposal_exp_rate="$5"
  local hyper_T="$6"

  python unigram/unigram_test2_tmp3.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +max_steps=20000 \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    "+loss_geometry=${loss_geometry}" \
    +nelbo_proposal_type=stratified_exp \
    +nelbo_proposal_exp_rate=0.1 \
    +nelbo_geometry=poincare_polar \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}"
}

run_tmp4 \
  "$seed" \
  "cross_entropy" \
  "cross_entropy_tnb_tmp3_rlog_newd2_rot_ts4000000" \
  "unif" \
  "1.0"

# run_tmp4 \
#   "$seed" \
#   "cross_entropy" \
#   "cross_entropy_tnb_tmp3_rlog_newd2_rot_ts4000000" \
#   "stratified_exp" \
#   "1.0,0.1,0.01,0.2,0.3,0.8,0.5,2.0"

run_tmp4 \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_tnb_tmp3_rlog_newd2_rot_ts4000000" \
  "unif" \
  "1.0"

# run_tmp4 \
#   "$seed" \
#   "poincare_polar" \
#   "poincare_polar_tnb_tmp3_rlog_newd2_rot_ts4000000" \
#   "stratified_exp" \
#   "0.1,0.01,0.2,0.3,0.8,0.5,1.0,2.0"

run_tmp4 \
  "$seed" \
  "horo_cross_entropy" \
  "horo_cross_entropy_tnb_tmp3_rlog_newd2_rot_ts4000000" \
  "unif" \
  "1.0"

# run_tmp4 \
#   "$seed" \
#   "horo_cross_entropy" \
#   "horo_cross_entropy_tnb_tmp3_rlog_newd2_rot_ts4000000" \
#   "stratified_exp" \
#   "0.1,0.01,0.2,0.3,0.8,0.5,1.0,2.0"
