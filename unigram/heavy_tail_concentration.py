"""Heavy-tail concentration diagnostic for the IS-NELBO estimator.

Reproduces the slide table "A Handful of Samples Decide the Answer"
(unigram/slides/may31_2026/slides.md).

Data source: each diagonal run writes per-sample arrays to
  outputs/unigram_test2/poincare_polar_opt_tmp3_rlog_newd2_rot_ts{TS}_diag_rs42/
      unigram_test2_losses*_per{LAM}_*/test_loss_vs_timestep.json
saved by `plot_test_loss_vs_timestep` in unigram/unigram_test2_tmp3.py.

The key `weighted_nelbo` is the per-sample importance-weighted NELBO, i.e.
  w_i = bridge_loss(t_i) * proposal_weight(t_i)
and test_wnelbo = mean_i(w_i). So "fraction of the sum from the top-k samples"
measures how much of the estimate is decided by a handful of rare, huge-weight
draws -- the signature of a heavy-tailed (near-infinite-variance) estimator.

Run:  conda activate duo && python unigram/heavy_tail_concentration.py
"""
import json
import glob
import numpy as np

BASE = "outputs/unigram_test2"
SWEEP = "poincare_polar_opt_tmp3_rlog_newd2_rot_ts{ts}_diag_rs42"
LEAF = "unigram_test2_losses*_per{lam}_*/test_loss_vs_timestep.json"


def load_weighted_nelbo(ts: str, lam: str) -> np.ndarray:
    """Per-sample weighted NELBO array for one (test_size, lambda) cell."""
    pattern = f"{BASE}/{SWEEP.format(ts=ts)}/{LEAF.format(lam=lam)}"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(pattern)
    return np.asarray(json.load(open(matches[0]))["weighted_nelbo"], dtype=np.float64)


def concentration(w: np.ndarray, fracs=(1e-5, 1e-4, 1e-3)) -> dict:
    """Fraction of the total sum contributed by the top `frac` of samples."""
    n = w.size
    total = w.sum()
    top_desc = np.sort(w)[::-1]                    # largest first
    out = {"n": n, "mean": w.mean(), "max": w.max()}
    for frac in fracs:
        k = max(1, int(n * frac))
        out[frac] = (k, top_desc[:k].sum() / total)
    return out


def main():
    ts = "4e6"   # the slide table uses the 4M test set
    lams = ["0.1", "0.2", "1.0", "2.0"]
    print(f"=== top-sample concentration of the weighted-NELBO sum (test size {ts}) ===")
    hdr = f"{'lambda':>7} {'mean':>8} {'max':>10} {'top1e-5':>10} {'top1e-4':>10} {'top1e-3':>10}"
    print(hdr)
    for lam in lams:
        w = load_weighted_nelbo(ts, lam)
        r = concentration(w)
        print(f"{lam:>7} {r['mean']:>8.4f} {r['max']:>10.1f} "
              f"{r[1e-5][1]*100:>9.1f}% {r[1e-4][1]*100:>9.1f}% {r[1e-3][1]*100:>9.1f}%")

    # The "max grows with the test set" claim (lambda = 1.0): 4e5 -> 4e6
    print("\n=== max single weighted-NELBO sample vs test size (lambda=1.0) ===")
    for ts in ["4e5", "4e6"]:
        w = load_weighted_nelbo(ts, "1.0")
        print(f"  ts={ts}: max = {w.max():.1f}")


if __name__ == "__main__":
    main()
