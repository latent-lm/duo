#!/usr/bin/env python
# Utility functions for the ELBO reproduction in `Copy of HyperDiffTransformerPlane.py`.
# All non-notebook logic lives here so the notebook script stays a minimal diff.
#
# Config mirrors slides.md "Poincare-Polar ELBO - Fixed embedding" /
# unigram_test_lorentz_tmp4_loss_emb.sh: vocab V=10, ps=[0.91, 0.01x9] (H~=0.5003),
# proposals = unif[0.01,{10,20,30,50}] + stratified_exp lambda{0.1,0.2,0.3,0.5,0.8,1.0},
# test_size=4e6, fixed equally-divided embedding (built into bridge_loss's phis).
import os
import glob as _glob
import json as _json

import torch

PS = [0.91] + [0.01] * 9
TEST_SIZE = int(float(os.environ.get("ELBO_TEST_SIZE", "4000000")))   # slide uses 4e6
CHUNK = int(float(os.environ.get("ELBO_CHUNK", "1000000")))
SEEDS = [int(s) for s in os.environ.get("ELBO_SEEDS", "42,0,1,2,3").split(",") if s.strip()]

# slide's "Poincare-Polar ELBO - Fixed embedding" experiment proposals
PROPOSALS = (
    [(f"unif[0.01,{t}]", dict(kind="unif", unif_min=0.01, unif_max=float(t))) for t in (10, 20, 30, 50)]
    + [(f"strat_exp(lam={l})", dict(kind="stratified_exp", exp_rate=l)) for l in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)]
)

# slide numbers for the comparison
SLIDE_OPT_4E6 = {0.1: 0.4901, 0.2: 0.4986, 0.3: 0.4974, 0.5: 0.5049, 0.8: 0.5253, 1.0: 0.5271}  # mode=opt table
SLIDE_FIXEDEMB_UNIF = {10: 0.329, 20: 0.381, 30: 0.396, 50: 0.412}                               # trained table
SLIDE_FIXEDEMB_STRAT = {0.1: 0.396, 0.2: 0.421, 0.3: 0.437, 0.5: 0.760, 0.8: 1.268, 1.0: 1.379}


def unigram_dataset(test_size=TEST_SIZE, chunk=CHUNK):
    """Slide's unigram setup. Returns V, ps, initial_bias(=log p), (nbatches,batchsize,seqlen)."""
    ps = torch.tensor(PS, dtype=torch.float64)
    ps = ps / ps.sum()
    nbatches = max(1, test_size // chunk)
    return ps.numel(), ps, ps.log(), (nbatches, chunk, 1)


def sample_proposal(N, kind="unif", exp_rate=1.0, unif_min=0.01, unif_max=10.0, generator=None):
    """(ts, weights) with E[L*w]=integral L dt. unif and stratified_exp only (slide spec)."""
    if kind == "unif":
        interval = float(unif_max - unif_min)
        ts = unif_min + interval * torch.rand(N, dtype=torch.float64, generator=generator)
        return ts, torch.full((N,), interval, dtype=torch.float64)
    if kind == "stratified_exp":
        u = (torch.arange(N, dtype=torch.float64)
             + torch.rand(N, dtype=torch.float64, generator=generator)) / N
        u = u[torch.randperm(N, generator=generator)].clamp(min=1e-12, max=1 - 1e-12)
        return -torch.log(u) / exp_rate, 1.0 / (exp_rate * u)   # weight == exp(rate*ts)/rate
    raise ValueError(f"unknown proposal {kind!r}")


class Accum:
    """Streaming accumulator for the weighted bridge NELBO over the test set."""
    def __init__(self, H, keep=5000):
        self.H = H; self.N = 0; self.keep = keep
        self.ws = self.wsq = self.ces = self.cesq = 0.0; self.maxw = 0.0
        self.top = torch.zeros(0, dtype=torch.float64)

    def update(self, wnelbo, logits, targets, weights):
        wn = wnelbo.double(); n = wn.numel(); self.N += n
        self.ws += float(wn.sum()); self.wsq += float(wn.square().sum())
        ce = torch.nn.functional.cross_entropy(logits.double(), targets, reduction="none")
        self.ces += float(ce.sum()); self.cesq += float(ce.square().sum())
        self.maxw = max(self.maxw, float(weights.max()))
        self.top = torch.cat([self.top, wn]).topk(min(self.keep, self.top.numel() + n)).values

    def summary(self):
        def st(sm, sq):
            m = sm / self.N
            return m, (max(sq - sm * sm / self.N, 0.0) / (self.N - 1)) ** 0.5
        wm, wstd = st(self.ws, self.wsq); cm, cstd = st(self.ces, self.cesq)
        sk = {f: (float(self.top[:max(1, int(self.N * f))].sum()) / self.ws if self.ws else float("nan"))
              for f in (1e-5, 1e-4, 1e-3)}
        return dict(wnelbo_mean=wm, wnelbo_std=wstd, ce_mean=cm, ce_std=cstd,
                    H=self.H, gap=wm - self.H, max_weight=self.maxw, skew=sk, N=self.N)


def dump_results(results, path=None):
    seeds = sorted({s for (_, s) in results})
    path = path or f"results_seed{'_'.join(map(str, seeds))}.json"
    out = {}
    for (p, s), r in results.items():
        row = {k: v for k, v in r.items() if k != "skew"}
        row["skew"] = {str(kk): vv for kk, vv in r["skew"].items()}
        out[f"{p}|{s}"] = row
    _json.dump({"rows": out}, open(path, "w"))
    return path


def load_all_results(globpat="results_seed*.json"):
    res = {}
    for f in sorted(_glob.glob(globpat)):
        for k, r in _json.load(open(f))["rows"].items():
            p, s = k.rsplit("|", 1); res[(p, int(s))] = r
    return res


def aggregate_and_plot(results, H, fig_path="elbo_vs_proposal.png", txt_path="slide_comparison.txt"):
    """Print the table + across-seed fluctuation + slide comparison; save the figure.
    `results` is {(proposal_name, seed): summary_dict}."""
    seeds = sorted({s for (_, s) in results})
    names = [n for n, _ in PROPOSALS if any((n, s) in results for s in seeds)]
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit(f"H = {H:.5f} nats   N = {results[(names[0], seeds[0])]['N']:,}   seeds = {seeds}")
    emit(f"\n=== Optimal-model bridge ELBO, equally-divided fixed embedding (seed {seeds[0]}) ===")
    emit(f"sanity: optimal-model test_ce should equal H={H:.5f} for every proposal")
    emit(f"{'proposal':<20}{'test_ce':>14}{'test_wnelbo':>22}{'gap':>10}{'max_w':>13}")
    emit("-" * 79)
    for n in names:
        r = results[(n, seeds[0])]
        gap = sum(results[(n, s)]['wnelbo_mean'] for s in seeds) / len(seeds) - H
        emit(f"{n:<20}{r['ce_mean']:>7.3f}+/-{r['ce_std']:<4.2f}"
             f"{r['wnelbo_mean']:>9.4f}+/-{r['wnelbo_std']:<7.2f}{gap:>+10.4f}{r['max_weight']:>13.1f}")

    if len(seeds) > 1:
        emit(f"\n=== Across-seed fluctuation of test_wnelbo ({len(seeds)} seeds) ===")
        emit(f"{'proposal':<20}{'mean':>9}{'seed-std':>10}{'min':>9}{'max':>9}{'gap(mean-H)':>13}")
        emit("-" * 70)
        for n in names:
            ms = [results[(n, s)]['wnelbo_mean'] for s in seeds]
            m = sum(ms) / len(ms); sd = (sum((x - m) ** 2 for x in ms) / (len(ms) - 1)) ** 0.5
            emit(f"{n:<20}{m:>9.4f}{sd:>10.4f}{min(ms):>9.4f}{max(ms):>9.4f}{m-H:>+13.4f}")

    emit(f"\n=== Comparison with unigram/slides/may31_2026/slides.md ===")
    emit("(A) vs OPTIMAL-model table ('ELBO under various Testing Dataset Size', 4e6 col)")
    emit(f"    {'lambda':>7}{'mine':>11}{'slide':>9}{'mine-slide':>12}")
    for lam in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        n = f"strat_exp(lam={lam})"
        if not any((n, s) in results for s in seeds):
            continue
        m = sum(results[(n, s)]['wnelbo_mean'] for s in seeds) / len(seeds)
        emit(f"    {lam:>7}{m:>11.4f}{SLIDE_OPT_4E6[lam]:>9.4f}{m - SLIDE_OPT_4E6[lam]:>+12.4f}")
    emit("(B) vs TRAINED fixed-embedding table (its test_ce was 0.57-2.4 = under-trained;")
    emit("    a close match is NOT expected -- shown for reference)")
    emit(f"    {'proposal':>16}{'mine':>11}{'slide':>9}")
    for t in (10, 20, 30, 50):
        n = f"unif[0.01,{t}]"
        if any((n, s) in results for s in seeds):
            m = sum(results[(n, s)]['wnelbo_mean'] for s in seeds) / len(seeds)
            emit(f"    {n:>16}{m:>11.4f}{SLIDE_FIXEDEMB_UNIF[t]:>9.4f}")
    for lam in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        n = f"strat_exp(lam={lam})"
        if any((n, s) in results for s in seeds):
            m = sum(results[(n, s)]['wnelbo_mean'] for s in seeds) / len(seeds)
            emit(f"    {('strat '+str(lam)):>16}{m:>11.4f}{SLIDE_FIXEDEMB_STRAT[lam]:>9.4f}")
    open(txt_path, "w").write("\n".join(lines) + "\n")

    _plot(results, H, names, seeds, fig_path)


def _plot(results, H, names, seeds, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(names))
    mean = [np.mean([results[(n, s)]['wnelbo_mean'] for s in seeds]) for n in names]
    std = [np.std([results[(n, s)]['wnelbo_mean'] for s in seeds], ddof=1) if len(seeds) > 1 else 0.0 for n in names]
    c = ["#1f77b4" if n.startswith("unif") else "#d62728" for n in names]
    lab = [n.replace("strat_exp(lam=", "strat l=").replace(")", "").replace("unif[0.01,", "unif<=").replace("]", "") for n in names]
    sx, sy = [], []
    for i, n in enumerate(names):
        if n.startswith("strat_exp"):
            lam = float(n.split("=")[1].rstrip(")"))
            if lam in SLIDE_OPT_4E6:
                sx.append(i); sy.append(SLIDE_OPT_4E6[lam])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(H, color="k", ls="--", lw=1.5, label=f"entropy H={H:.4f}")
    ax.errorbar(x, mean, yerr=std, fmt="o", ms=8, capsize=4, color="none", ecolor="gray", zorder=2)
    ax.scatter(x, mean, c=c, s=80, zorder=3, label="mine (optimal model, %d seeds)" % len(seeds))
    if sx:
        ax.scatter(sx, sy, marker="x", s=110, c="green", zorder=4, linewidths=2.5,
                   label="slide (optimal-model table, 4e6)")
    for xi, m in zip(x, mean):
        ax.annotate(f"{m-H:+.3f}", (xi, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("test_wnelbo (bridge ELBO)")
    ax.set_title("Notebook bridge ELBO (optimal model, equally-divided embedding) vs entropy\n"
                 "stratified_exp + unif  -  x=slide optimal-model values")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_path, dpi=130)
    print(f"\nwrote {fig_path}")
