#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if (($# > 0)); then
  seeds=("$@")
else
  read -r -a seeds <<< "${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62}"
fi

PS='[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]'

run_tmp2_opt() {
  local seed="$1"
  local loss_geometry="$2"
  local sweep_name="$3"
  local proposal_exp_rate="$4"

  python unigram/unigram_test2_tmp2.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=opt \
    +vocab_size=10 \
    +max_steps=20000 \
    +lr=1e-5 \
    "+ps=${PS}" \
    +proposal_type=stratified_exp \
    "+proposal_exp_rate=${proposal_exp_rate}" \
    "+loss_geometry=${loss_geometry}" \
    "+rotate_emb=False" \
    +hidden_size=128 \
    +depth=3 \
    +hyper_T=1000 \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}"
}

for seed in "${seeds[@]}"; do
  echo "Running tmp2 optimal experiments with seed=${seed}"

  run_tmp2_opt \
    "$seed" \
    "lorentz_cartesian" \
    "lorentz_cartesian_opt_tmp_rlog_newd2_ts4000000" \
    "1.0,0.1,0.01,0.2,0.3,0.8,0.5,2.0"

  run_tmp2_opt \
    "$seed" \
    "poincare_polar" \
    "poincare_polar_opt_tmp_rlog_newd2_ts4000000" \
    "1.0,0.1,0.01,0.2,0.3,0.8,0.5,2.0"
done
