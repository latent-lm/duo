#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Optimal-model Poincaré-polar bridge-ELBO sweep with tmp4, to compare against
#   hyperbolic_dm_workspace/Copy of HyperDiffTransformerPlane.py
#   (see hyperbolic_dm_workspace/RESULTS_elbo_reproduction.md).
#
# mode=opt -> OptimalModel emits logits = log p(y) (Bayes-optimal), no training.
# With word_embedding=None (opt has no lm-head) the bridge/loss use the fixed
# equally-spaced phi_v -> bit-identical to the notebook's bridge math (verified).
#
# Same hyperparameters as the reproduction:
#   Vocab 10 · ps=[0.91, 0.01x9] · H=0.50029 · N (test) = 4e6 · hyper_dt=0.01
#   Proposals: unif with hyper_T in {1000,2000,3000,5000}  -> [0.01,10/20/30/50]
#              stratified_exp with lambda in {0.1,0.2,0.3,0.5,0.8,1.0} (hyper_T=1000)
#   Seeds: {42, 0, 1, 2, 3}   -> report test_wnelbo mean +/- seed-std per proposal.
# => 10 proposals x 5 seeds = 50 opt-model eval runs (no training).
#
# NELBO eval = same proposal as loss (matched). Re-running: clear stale
# finished.json under outputs/unigram_test2/opt_repro_* first.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PS='[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]'
TEST_SIZE=4000000
SEEDS="42 0 1 2 3"
UNIF_HYPER_T="1000,2000,3000,5000"        # Hydra multirun list (hyper_T is one shared key)
STRAT_RATES="0.1 0.2 0.3 0.5 0.8 1.0"     # space-separated (POSIX for-loop, no arrays)

# Optimal-model bridge-ELBO run. proposal type/rate/hyper_T are swept by the
# caller; the NELBO uses the SAME proposal as the loss.
run_opt() {
  local seed="$1" sweep_name="$2" proposal_type="$3" proposal_exp_rate="$4" hyper_T="$5"
  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=opt \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+test_size=${TEST_SIZE}" \
    +max_steps=20000 \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    +loss_geometry=poincare_polar \
    "+nelbo_proposal_type=${proposal_type}" \
    "+nelbo_proposal_exp_rate=${proposal_exp_rate}" \
    +nelbo_geometry=poincare_polar \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}_${sweep_name}"
}

for seed in ${SEEDS}; do
  echo "================ seed=${seed} ================"
  # unif: sweep hyper_T as a Hydra multirun list (one shared key, no cross-product)
  run_opt "$seed" "opt_repro_unif" "unif" "1.0" "${UNIF_HYPER_T}"
  # stratified_exp: diagonal loop, one Hydra job per lambda (loss_rate == nelbo_rate)
  for er in ${STRAT_RATES}; do
    run_opt "$seed" "opt_repro_strat" "stratified_exp" "${er}" "1000"
  done
done

echo "================ AGGREGATE (mean +/- seed-std vs entropy) ================"
python unigram_test_script/agg_opt_elbo.py
