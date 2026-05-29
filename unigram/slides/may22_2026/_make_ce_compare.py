"""Build CE-vs-PP training comparison figures for the may22 deck.

Numbers are read directly from the test_metrics.json values reported in
outputs/unigram_test2/{cross_entropy,poincare_polar}_tnb_tmp3_rlog_newd2_rot_ts4000000_rs42.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# proposal label -> (CE-trained, PP-trained)
labels = ["unif", "0.1", "0.2", "0.3", "0.5", "0.8", "1.0", "2.0"]

ce_test_ce = [0.3164, 1.9838, 1.9805, 1.9794, 1.9787, 1.9805, 1.9823, 1.9897]
pp_test_ce = [0.9658, 1.8352, 0.8907, 0.5453, 0.6175, 1.2055, 1.6483, 2.0032]

ce_wnelbo = [0.4814, 2.0307, 2.0173, 2.0094, 1.9995, 1.9923, 1.9912, 1.9949]
pp_wnelbo = [0.4156, 0.3960, 0.4136, 0.4303, 0.5116, 0.9828, 1.3546, 1.7160]

H = 0.5003       # data entropy (nats)
LOGV = 2.3026    # uniform ceiling, log 10

x = list(range(len(labels)))
CE_C = "#d1495b"   # cross_entropy-trained (red)
PP_C = "#2e6f95"   # poincare_polar-trained (blue)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

for ax, (ce_y, pp_y, title) in zip(
    axes,
    [(ce_test_ce, pp_test_ce, "test_ce  (plain cross-entropy)"),
     (ce_wnelbo, pp_wnelbo, "test_wnelbo  (common Poincaré-ELBO yardstick)")],
):
    ax.axhline(LOGV, ls=":", c="0.55", lw=1.3, zorder=0)
    ax.axhline(H, ls="--", c="#3a923a", lw=1.3, zorder=0)
    ax.plot(x, ce_y, "o-", color=CE_C, lw=2.4, ms=7, label="trained w/ cross_entropy")
    ax.plot(x, pp_y, "s-", color=PP_C, lw=2.4, ms=7, label="trained w/ poincare_polar")
    ax.text(len(labels) - 1, LOGV + 0.03, "uniform ceiling  log V = 2.30",
            ha="right", va="bottom", fontsize=8.5, color="0.45")
    ax.text(0, H - 0.07, "data entropy 0.50", ha="left", va="top",
            fontsize=8.5, color="#3a923a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"proposal  (stratified-exp rate $\lambda$,  or uniform)")
    ax.set_ylabel("nats")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 2.45)
    ax.grid(alpha=0.25)

axes[0].legend(loc="center right", fontsize=9, framealpha=0.95)
fig.suptitle("Cross-entropy loss vs Poincaré-polar loss  (trained MLP, seed 42, V=10)",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig("ce_vs_pp_compare.jpg", dpi=150, bbox_inches="tight")
print("wrote ce_vs_pp_compare.jpg")
