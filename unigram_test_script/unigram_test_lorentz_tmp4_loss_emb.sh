#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 2 x 2 experiment (slides unigram/slides/may31_2026/slides.md "Two Model
# Parametrization" + "Learnable / Fixed Word Embedding"):
#
#   Model parametrization                       x   Word embedding
#   --------------------------------------          -----------------------------
#   Horocycle = horo_cross_entropy                  Learnable = trainable_word_embedding=True
#     mu = softmax(f(z) + horosphere term)          Fixed     = trainable_word_embedding=False
#   Direct    = cross_entropy
#     mu = softmax(f(z))
#
# Each of the 4 cells is swept over the proposal:
#   - unif           with hyper_T in {1000, 2000, 3000, 5000}
#   - stratified_exp with lambda  in {1.0, 0.1, 0.2, 0.3, 0.8, 0.5}  (hyper_T=1000)
# => 4 cells x (4 + 6) = 40 training runs.
#
# NELBO eval convention: for EVERY cell the test_wnelbo (PP / poincare_polar) is
# evaluated with the SAME proposal as the training loss (loss proposal == ELBO
# proposal). Because the stratified rate then varies on BOTH the loss and the
# nelbo axis, that sweep is a DIAGONAL bash loop (one Hydra job per lambda) --
# passing the rate list to `-m` would cross-product the two axes (6x6 = 36 jobs).
# The unif sweep stays a Hydra multirun list because hyper_T is one shared key.
#
# No `set -e`: a diverging cell (Horocycle x Learnable is expected to NaN) must
# not abort the rest of the sweep.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

seed=42
PS='[0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]'
TEST_SIZE=4000000
MAX_STEPS=20000
UNIF_HYPER_T="1000,2000,3000,5000"      # Hydra multirun list (hyper_T is one shared key)
STRAT_RATES="1.0 0.1 0.2 0.3 0.8 0.5"   # space-separated for POSIX for-loops (no arrays)

# Horocycle parametrization: loss_geometry=horo_cross_entropy
#   (CE on the horosphere-augmented logits: mu = softmax(f(z) + horosphere)).
# NELBO = poincare_polar evaluated with stratified_exp(0.1).
run_horocycle() {
  local trainable="$1" sweep_name="$2" proposal_type="$3" proposal_exp_rate="$4" hyper_T="$5"
  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+trainable_word_embedding=${trainable}" \
    "+test_size=${TEST_SIZE}" \
    "+max_steps=${MAX_STEPS}" \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    +loss_geometry=horo_cross_entropy \
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
    "+postfix=rs${seed}_${sweep_name}"
}

# Direct parametrization: loss_geometry=cross_entropy (mu = softmax(f(z))).
# NELBO = poincare_polar evaluated with the stratified_exp(0.1).
run_direct() {
  local trainable="$1" sweep_name="$2" proposal_type="$3" proposal_exp_rate="$4" hyper_T="$5"
  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+trainable_word_embedding=${trainable}" \
    "+test_size=${TEST_SIZE}" \
    "+max_steps=${MAX_STEPS}" \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    +loss_geometry=cross_entropy \
    +nelbo_proposal_type=stratified_exp \
    +nelbo_proposal_exp_rate=0.1 \
    +nelbo_geometry=poincare_polar_horocycle \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}_${sweep_name}"
}

# ---- 2x2 driver: {Horocycle, Direct} x {Learnable, Fixed} ------------------
# `learn/fix` is baked into postfix because folder() does NOT include
# trainable_word_embedding; without it the two embedding cells would resolve to
# the same run folder and TaskMgr.check_finished() would skip the second.
for twe in True False; do
  if [ "$twe" = "True" ]; then emb="learnemb"; else emb="fixedemb"; fi

  echo "================ Horocycle (horo_cross_entropy) x ${emb} ================"
  run_horocycle "$twe" "horocycle_${emb}_unif_new" "unif" "1.0" "${UNIF_HYPER_T}"
  run_horocycle "$twe" "horocycle_${emb}_strat_new" "stratified_exp" "1.0,0.1,0.2,0.3,0.8,0.5" "1000"

  echo "================ Direct (cross_entropy) x ${emb} ================"
  # NELBO is fixed (PP strat 0.1), so the loss-side lists are safe as Hydra lists.
  run_direct "$twe" "direct_${emb}_unif_new" "unif" "1.0" "${UNIF_HYPER_T}"
  run_direct "$twe" "direct_${emb}_strat_new" "stratified_exp" "1.0,0.1,0.2,0.3,0.8,0.5" "1000"
done

# Horocycle parametrization: loss_geometry=horo_cross_entropy
#   (CE on the horosphere-augmented logits: mu = softmax(f(z) + horosphere)).
# NELBO = poincare_polar evaluated with the SAME proposal as the loss.
run_horocycle_align() {
  local trainable="$1" sweep_name="$2" proposal_type="$3" proposal_exp_rate="$4" hyper_T="$5"
  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+trainable_word_embedding=${trainable}" \
    "+test_size=${TEST_SIZE}" \
    "+max_steps=${MAX_STEPS}" \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    +loss_geometry=horo_cross_entropy \
    +nelbo_proposal_type=${proposal_type} \
    +nelbo_proposal_exp_rate=${proposal_exp_rate} \
    +nelbo_geometry=poincare_polar \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}_${sweep_name}"
}

# Direct parametrization: loss_geometry=cross_entropy (mu = softmax(f(z))).
# NELBO = poincare_polar evaluated with the SAME proposal as the loss.
run_direct_align() {
  local trainable="$1" sweep_name="$2" proposal_type="$3" proposal_exp_rate="$4" hyper_T="$5"
  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+trainable_word_embedding=${trainable}" \
    "+test_size=${TEST_SIZE}" \
    "+max_steps=${MAX_STEPS}" \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    +loss_geometry=cross_entropy \
    +nelbo_proposal_type=${proposal_type} \
    +nelbo_proposal_exp_rate=${proposal_exp_rate} \
    +nelbo_geometry=poincare_polar_horocycle \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}_${sweep_name}"
}

run_poincare_polar_elbo_align() {
  local trainable="$1" sweep_name="$2" proposal_type="$3" proposal_exp_rate="$4" hyper_T="$5"
  python unigram/unigram_test2_tmp4.py -m \
    hydra.job.chdir=true \
    hydra.sweep.dir=outputs/unigram_test2 \
    "hydra.sweep.subdir=${sweep_name}_rs${seed}" \
    +mode=tnb \
    +vocab_size=10 \
    +hyper_dim=2 \
    "+trainable_word_embedding=${trainable}" \
    "+test_size=${TEST_SIZE}" \
    "+max_steps=${MAX_STEPS}" \
    +lr=1e-5 \
    "+ps=${PS}" \
    "+loss_proposal_type=${proposal_type}" \
    "+loss_proposal_exp_rate=${proposal_exp_rate}" \
    +loss_geometry=poincare_polar \
    +nelbo_proposal_type=${proposal_type} \
    +nelbo_proposal_exp_rate=${proposal_exp_rate} \
    +nelbo_geometry=poincare_polar \
    "+rotate_emb=True" \
    +hidden_size=128 \
    +depth=3 \
    "+hyper_T=${hyper_T}" \
    +hyper_dt=0.01 \
    +loss_plot_ma_window=50 \
    "+seed=${seed}" \
    "+postfix=rs${seed}_${sweep_name}"
}

# ---- 2x2 driver: {Horocycle, Direct} x {Learnable, Fixed} ------------------
# `learn/fix` is baked into postfix because folder() does NOT include
# trainable_word_embedding; without it the two embedding cells would resolve to
# the same run folder and TaskMgr.check_finished() would skip the second.
for twe in True False; do
  if [ "$twe" = "True" ]; then emb="learnemb"; else emb="fixedemb"; fi

  echo "================ Horocycle (horo_cross_entropy) x ${emb} ================"
  run_horocycle_align "$twe" "horocycle_${emb}_unif_align_new" "unif" "1.0" "${UNIF_HYPER_T}"
  for pr in ${STRAT_RATES}; do
    run_horocycle_align "$twe" "horocycle_${emb}_strat_align_new" "stratified_exp" ${pr} "1000"
  done

  echo "================ Direct (cross_entropy) x ${emb} ================"
  # NELBO is fixed (PP strat 0.1), so the loss-side lists are safe as Hydra lists.
  run_direct_align "$twe" "direct_${emb}_unif_align_new" "unif" "1.0" "${UNIF_HYPER_T}"
  for pr in ${STRAT_RATES}; do
    run_direct_align "$twe" "direct_${emb}_strat_align_new" "stratified_exp" ${pr} "1000"
  done

  echo "================ Poincare Polar (ELBO) x ${emb} ================"
  run_poincare_polar_elbo_align "$twe" "poincare_polar_${emb}_unif_align_new" "unif" "1.0" "${UNIF_HYPER_T}"
  for pr in ${STRAT_RATES}; do
    run_poincare_polar_elbo_align "$twe" "poincare_polar_${emb}_strat_align_new" "stratified_exp" ${pr} "1000"
  done
done
