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

run_tmp3_opt() {
  local seed="$1"
  local loss_geometry="$2"
  local sweep_name="$3"
  local test_size="$4"
  local proposal_type="$5"
  local proposal_exp_rate="$6"
  local hyper_T="$7"

  python unigram/unigram_test2_tmp3.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=opt \
    +vocab_size=10 \
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

run_tmp3_opt \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_hT" \
  "4000000" \
  "unif" \
  "1.0" \
  "10"

run_tmp3_opt \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_hT" \
  "4000000" \
  "unif" \
  "1.0" \
  "100"

run_tmp3_opt \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_hT" \
  "4000000" \
  "unif" \
  "1.0" \
  "1e5"

run_tmp3_opt \
  "$seed" \
  "poincare_polar" \
  "poincare_polar_opt_tmp3_rlog_newd2_rot_ts4000000_hT" \
  "4000000" \
  "unif" \
  "1.0" \
  "1e7"

# Run the DIAGONAL of the (loss_rate x nelbo_rate) grid: one Hydra job per rate, so
# loss_proposal_exp_rate == nelbo_proposal_exp_rate in every run. Passing the comma
# list to `-m` instead cross-products the two axes (8x8=64 jobs); because the leaf
# folder is named by the loss rate only, every surviving leaf would run nelbo_rate=1.0.
exp_rates=(1.0 0.1 0.01 0.2 0.3 0.8 0.5 2.0)

# for ts in 4e4 4e5 4e6 4e7 4e8; do
for ts in 4e8 4e7 4e6 4e5 4e4; do
  for er in "${exp_rates[@]}"; do
    run_tmp3_opt \
      "$seed" \
      "poincare_polar" \
      "poincare_polar_opt_tmp3_rlog_newd2_rot_ts${ts}_diag" \
      "${ts}" \
      "exp" \
      "${er}" \
      "1000"
  done
done

# run_tmp3_tnb() {
#   local seed="$1"
#   local loss_geometry="$2"
#   local sweep_name="$3"
#   local proposal_exp_rate="$4"

#   python unigram/unigram_test2_tmp3.py -m \
#     hydra.job.chdir=true \
#     hydra.sweep.dir=outputs/unigram_test2 \
#     "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
#     +mode=tnb \
#     +vocab_size=10 \
#     +max_steps=20000 \
#     +lr=1e-5 \
#     "+ps=${PS}" \
#     +loss_proposal_type=stratified_exp \
#     "+loss_proposal_exp_rate=${proposal_exp_rate}" \
#     "+loss_geometry=${loss_geometry}" \
#     +nelbo_proposal_type=stratified_exp \
#     +nelbo_proposal_exp_rate=0.1 \
#     +nelbo_geometry=poincare_polar \
#     "+rotate_emb=True" \
#     +hidden_size=128 \
#     +depth=3 \
#     +hyper_T=1000 \
#     +hyper_dt=0.01 \
#     +loss_plot_ma_window=50 \
#     "+seed=${seed}" \
#     "+postfix=rs${seed}"
# }

# run_tmp3_tnb \
#   "$seed" \
#   "poincare_polar" \
#   "poincare_polar_tnb_tmp3_rlog_newd2_rot_ts4000000" \
#   "1.0,0.1,0.01,0.2,0.3,0.8,0.5,2.0"

# run_tmp3_tnb \
#   "$seed" \
#   "lorentz_cartesian" \
#   "lorentz_cartesian_tnb_tmp3_rlog_newd2_rot_ts4000000" \
#   "1.0,0.1,0.01,0.2,0.3,0.8,0.5,2.0"
