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


def plot_cumulative_curve(
    ts: str = "4e6",
    lams=("0.1", "0.2", "0.5", "1.0", "2.0"),
    out_path: str = "unigram/slides/may31_2026/wnelbo_lorenz.jpg",
    n_plot: int = 3000,
) -> str:
    """Cumulative-contribution (Lorenz) curve of the per-sample weighted NELBO.

    Two panels share the same data (one (test_size, lambda) array per curve):

    (a) Lorenz curve  -- exactly the requested axes:
        X: the estimation points (samples) ordered small -> large, as a
           fraction of all N samples.
        Y: cumulative portion of the total sum contributed up to that point,
           cumsum(sort_ascending(w)) / sum(w).
        The dashed diagonal is the no-concentration reference (all samples
        equal); the more a curve bows to the bottom-right, the heavier the tail.

    (b) Tail view (log x) -- the same numbers, read from the top, so the lambda
        curves separate: X = top fraction q of samples (largest first, log),
        Y = portion of the sum carried by that top q. The dotted verticals at
        q = 1e-5, 1e-4, 1e-3 are the columns of the slide table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _idx_dense_near(n, end):
        """Subsample indices, log-dense near `end` ('low' rank 0, or 'high' n-1)."""
        log = np.logspace(0, np.log10(n - 1), n_plot).astype(np.int64)
        lo = log if end == "low" else (n - 1) - log
        both = np.concatenate([np.linspace(0, n - 1, n_plot).astype(np.int64), lo])
        return np.unique(np.clip(both, 0, n - 1))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("viridis")
    axL.plot([0, 1], [0, 1], "k--", lw=1, label="even (no concentration)")

    for i, lam in enumerate(lams):
        w = np.sort(load_weighted_nelbo(ts, lam))          # ascending: small -> large
        n = w.size
        total = w.sum()
        color = cmap(i / max(1, len(lams) - 1))

        # (a) Lorenz: cumulative share vs fraction of samples small -> large
        cum = np.cumsum(w) / total
        x = np.arange(1, n + 1) / n
        a = _idx_dense_near(n, "high")                     # the turn is near x = 1
        axL.plot(x[a], cum[a], lw=2, color=color, label=f"λ={lam}")

        # (b) tail: share carried by the top-q fraction (read from the largest)
        cum_top = np.cumsum(w[::-1]) / total
        q = np.arange(1, n + 1) / n
        b = _idx_dense_near(n, "low")                      # detail at small q (the top)
        axR.plot(q[b], cum_top[b], lw=2, color=color, label=f"λ={lam}")

    axL.set_xlabel("fraction of samples, ordered small → large")
    axL.set_ylabel("cumulative portion of the IS-NELBO sum")
    axL.set_title("(a) Lorenz curve")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.grid(alpha=0.3); axL.legend(fontsize=8, loc="upper left")

    for f in (1e-5, 1e-4, 1e-3):
        axR.axvline(f, color="gray", ls=":", lw=0.8)
    axR.set_xscale("log")
    axR.set_xlabel("top fraction of samples (largest first, log)")
    axR.set_ylabel("portion of the sum from the top fraction")
    axR.set_title("(b) tail contribution")
    axR.set_ylim(0, 1)
    axR.grid(alpha=0.3, which="both"); axR.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Concentration of the weighted-NELBO sum  (test size {ts})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


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

    # Cumulative (Lorenz) curve of the per-sample contributions.
    path = plot_cumulative_curve(ts="4e6", lams=("0.1", "0.2", "0.5", "1.0", "2.0"))
    print(f"\nSaved cumulative curve to: {path}")


if __name__ == "__main__":
    main()
