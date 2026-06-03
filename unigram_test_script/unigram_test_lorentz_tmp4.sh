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

# tnb mode = LEARNABLE word embedding. Each vocab word's boundary angle is
# atan2 of its lm-head weight row (HyperBridge._vocab_angles / rotate_with_target
# in unigram/unigram_test2_tmp4.py). main() runs trainer.fit only for mode=tnb,
# so this trains the model and then tests it.
run_tmp4_tnb() {
  local seed="$1"
  local loss_geometry="$2"
  local sweep_name="$3"
  local test_size="$4"
  local proposal_type="$5"
  local proposal_exp_rate="$6"
  local hyper_T="$7"

  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+test_size=${test_size}" \
    +max_steps=20000 \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    "+loss_geometry=${loss_geometry}" \
    "+nelbo_proposal_type=${proposal_type}" \
    "+nelbo_proposal_exp_rate=${proposal_exp_rate}" \
    "+nelbo_geometry=${loss_geometry}" \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}"
}

# Primary run: learnable-embedding poincare_polar with an exponential proposal.
run_tmp4_tnb \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_tnb_tmp4_learnemb_ts4000000_TEST" \
  "4000000" \
  "unif" \
  "1.0" \
  "1000,2000,3000,5000"

run_tmp4_tnb \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_tnb_tmp4_learnemb_ts4000000_TEST" \
  "4000000" \
  "stratified_exp" \
  "1.0,0.1,0.2,0.3,0.8,0.5" \
  "1000"

run_tmp4_tnb_ce() {
  local seed="$1"
  local loss_geometry="$2"
  local sweep_name="$3"
  local test_size="$4"
  local proposal_type="$5"
  local proposal_exp_rate="$6"
  local hyper_T="$7"

  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+test_size=${test_size}" \
    +max_steps=20000 \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    "+loss_geometry=${loss_geometry}" \
    "+nelbo_proposal_type=stratified_exp" \
    "+nelbo_proposal_exp_rate=0.1" \
    "+nelbo_geometry=poincare_polar" \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}"
}

run_tmp4_tnb_ce \
  "$seed" \
  "cross_entropy" \
  "cross_entropy_tnb_tmp4_learnemb_ts4000000_TEST" \
  "4000000" \
  "unif" \
  "1.0" \
  "1000,2000,3000,5000"

run_tmp4_tnb_ce \
  "$seed" \
  "cross_entropy" \
  "cross_entropy_tnb_tmp4_learnemb_ts4000000_TEST" \
  "4000000" \
  "stratified_exp" \
  "1.0,0.1,0.2,0.3,0.8,0.5" \
  "1000"

# Optional diagonal sweep over the proposal exp rate (loss_rate == nelbo_rate per
# job). Uncomment to scale up; unlike the opt-mode sweeps, every tnb job TRAINS for
# max_steps, so this is much heavier.
# exp_rates=(1.0 0.1 0.01 0.2 0.3 0.8 0.5 2.0)
# for er in "${exp_rates[@]}"; do
#   run_tmp4_tnb \
#     "$seed" \
#     "poincare_polar" \
#     "poincare_polar_tnb_tmp4_learnemb_diag" \
#     "4000000" \
#     "exp" \
#     "${er}" \
#     "1000"
# done
