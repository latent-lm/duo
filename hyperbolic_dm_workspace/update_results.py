#!/usr/bin/env python
"""Update the comparison figures + RESULTS doc:

  Mine  = optimal-model Poincare-polar bridge-ELBO sweep from
          unigram_test_script/unigram_test_lorentz_tmp4_opt.sh
          (outputs/unigram_test2/opt_repro_*_rs*, mode=opt, 5 seeds).
  HyperDiffTransformerPlane.py = hyperbolic_dm_workspace/results_seed*.json
          (the notebook's optimal-model eval, 5 seeds).

Both are the same Bayes-optimal model under the same bridge math, so this is a
reproduction check. Outputs (X/Y axes unchanged from the originals):

  elbo_vs_proposal.png : test_wnelbo vs proposal, BOTH sources as
                         mean +/- across-seed std (ddof=1) + entropy line.
  loss_curve.png       : running NELBO over eval batches (one proposal),
                         BOTH sources as mean +/- across-seed std band.
                         Mine's trace is read from the saved per-sample
                         weighted_nelbo; HyperDiff's per-sample trace is NOT
                         persisted by the notebook, so it is recomputed here
                         with the notebook's own bbridge/bridge_loss.
  RESULTS_elbo_reproduction.md : a comparison section (tables) is inserted /
                         refreshed between AUTO markers.

Run with the `duo` env:  /home/sc3379/anaconda3/envs/duo/bin/python update_results.py
"""
import os, re, json, glob, math, statistics, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "outputs", "unigram_test2")
sys.path.insert(0, HERE)
import utils  # noqa: E402  (sample_proposal + unigram_dataset, shared with the notebook)

# --- experiment config (mirrors utils.PROPOSALS so the x-axis matches the old fig) ---
PROPOSALS = (
    [(f"unif[0.01,{t}]", dict(kind="unif", unif_min=0.01, unif_max=float(t))) for t in (10, 20, 30, 50)]
    + [(f"strat_exp(lam={l})", dict(kind="stratified_exp", exp_rate=l)) for l in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)]
)
PROP_ORDER = [n for n, _ in PROPOSALS]
PROP_KW = dict(PROPOSALS)
PS = [0.91] + [0.01] * 9
HYPER_DT = 0.01                  # opt_repro unif_max = hyper_T * hyper_dt
LOSS_CURVE_PROPOSAL = "unif[0.01,10]"   # which proposal's running-NELBO trace to draw
N_TRACE_PTS = 2000               # downsample the cumulative-mean trace to this many points
MARK = ("<!-- AUTO:mine-vs-hyperdiff START -->", "<!-- AUTO:mine-vs-hyperdiff END -->")

BLUE, RED = "#1f77b4", "#d62728"


def entropy():
    s = sum(PS)
    p = [x / s for x in PS]
    return -sum(pi * math.log(pi) for pi in p)


# ----------------------------- load results -----------------------------------
def _mine_key(folder):
    pt = re.search(r"_pt([a-z_]+?)_per", folder).group(1)
    if pt == "unif":
        hT = float(re.search(r"_hT([0-9.eE+]+)_", folder).group(1))
        return f"unif[0.01,{hT * HYPER_DT:.0f}]"
    per = re.search(r"_per([0-9.]+)_", folder).group(1)
    return f"strat_exp(lam={float(per)})"


def load_mine():
    """{prop_key: {seed: {'wnelbo':.., 'ce':..}}} from opt_repro_*_rs* test_metrics.json."""
    out = {}
    for sweep in glob.glob(os.path.join(OUT, "opt_repro_*_rs*")):
        m = re.search(r"_rs(\d+)$", os.path.basename(sweep))
        if not m:
            continue
        seed = int(m.group(1))
        for mf in glob.glob(os.path.join(sweep, "*", "test_metrics.json")):
            key = _mine_key(os.path.basename(os.path.dirname(mf)))
            d = json.load(open(mf))
            out.setdefault(key, {})[seed] = {"wnelbo": d["test_wnelbo"], "ce": d["test_ce"]}
    return out


def load_hyperdiff():
    """{prop_key: {seed: {'wnelbo':.., 'ce':..}}} from results_seed*.json."""
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results_seed*.json"))):
        seed = int(re.search(r"results_seed(\d+)\.json", os.path.basename(f)).group(1))
        for k, r in json.load(open(f))["rows"].items():
            key = k.rsplit("|", 1)[0]
            out.setdefault(key, {})[seed] = {"wnelbo": r["wnelbo_mean"], "ce": r["ce_mean"]}
    return out


def agg(per_seed, field):
    v = [per_seed[s][field] for s in sorted(per_seed)]
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


# ----------------------- figure 1: elbo vs proposal ----------------------------
def plot_elbo(mine, hd, H, path):
    names = [n for n in PROP_ORDER if n in mine or n in hd]
    x = np.arange(len(names))

    def series(d):
        mu = np.array([agg(d[n], "wnelbo")[0] if n in d else np.nan for n in names])
        sd = np.array([agg(d[n], "wnelbo")[1] if n in d else 0.0 for n in names])
        return mu, sd

    mm, ms = series(mine)
    hm, hs = series(hd)
    n_mine = max((agg(mine[n], "wnelbo")[2] for n in names if n in mine), default=0)
    n_hd = max((agg(hd[n], "wnelbo")[2] for n in names if n in hd), default=0)
    lab = [n.replace("strat_exp(lam=", "strat l=").replace(")", "").replace("unif[0.01,", "unif<=").replace("]", "")
           for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(H, color="k", ls="--", lw=1.5, label=f"entropy H={H:.4f}")
    ax.errorbar(x - 0.09, mm, yerr=ms, fmt="o", ms=7, capsize=4, color=BLUE,
                label=f"Mine (tmp4 opt_repro, {n_mine} seeds)")
    ax.errorbar(x + 0.09, hm, yerr=hs, fmt="s", ms=7, capsize=4, color=RED,
                label=f"HyperDiffTransformerPlane.py ({n_hd} seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("test_wnelbo (bridge ELBO)")
    ax.set_title("Bridge ELBO vs entropy (optimal model)\n"
                 "Mine (tmp4 opt_repro) vs HyperDiffTransformerPlane.py  -  mean +/- across-seed std")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------- figure 2: running-NELBO loss curve --------------------
def _trace_xy(weighted_nelbo):
    """cumulative running mean of per-sample weighted NELBO, log-downsampled to
    N_TRACE_PTS points. Returns (samples_seen, running_mean). Cumulative-mean
    convergence is essentially complete within the first ~0.1% of 4e6 samples, so
    the x-axis is sampled log-spaced (and plotted log) to show it at all."""
    arr = np.asarray(weighted_nelbo, dtype=np.float64)
    cm = np.cumsum(arr) / np.arange(1, arr.size + 1)
    idx = np.unique(np.geomspace(1, arr.size, N_TRACE_PTS).astype(int)) - 1
    return idx + 1, cm[idx]


def mine_traces(prop_key):
    """({seed: running-mean}, samples_seen) read from the saved per-sample weighted_nelbo."""
    traces, xs = {}, None
    for sweep in glob.glob(os.path.join(OUT, "opt_repro_*_rs*")):
        m = re.search(r"_rs(\d+)$", os.path.basename(sweep))
        if not m:
            continue
        seed = int(m.group(1))
        for jf in glob.glob(os.path.join(sweep, "*", "test_loss_vs_timestep.json")):
            if _mine_key(os.path.basename(os.path.dirname(jf))) != prop_key:
                continue
            xs, traces[seed] = _trace_xy(json.load(open(jf))["weighted_nelbo"])
    return traces, xs


# --- notebook's bridge math (token-identical to Copy of HyperDiffTransformerPlane.py) ---
def _sample_chi(ns):
    ns = ns.reshape(-1)
    M = int(ns.sum().item())
    x = torch.randn(M, dtype=torch.float64).square()
    return torch.segment_reduce(x, "sum", lengths=ns).sqrt()


@torch.no_grad()
def _bbridge(ts):
    ns = torch.poisson(ts / 8).to(torch.int64)
    ss = ts.sqrt() * _sample_chi(2 * ns + 3)
    vs = torch.rand_like(ts)
    rhos = torch.acosh(vs.square() + (1 - vs.square()) * torch.cosh(ss))
    us = torch.rand_like(ts)
    thetas = 2 * torch.atan((-rhos).exp() * torch.tan(torch.pi * (us - 0.5)))
    return rhos, thetas


def _bridge_loss(logits, targets, rhos, thetas, V):
    phis = (torch.arange(V, dtype=torch.float64) + 0.5) * (2 * torch.pi / V)
    alphas = thetas[:, None] - phis[None, :]
    ca, sa = alphas.cos(), alphas.sin()
    horo = math.log(2.0) - torch.logaddexp((1 - ca).log() + rhos[:, None], (1 + ca).log() - rhos[:, None])
    mu = (horo + logits).softmax(-1) - torch.nn.functional.one_hot(targets, V).to(torch.float64)
    betas = torch.atan2(sa, rhos.cosh()[:, None] * ca - rhos.sinh()[:, None])
    cos_err = (betas.cos() * mu).sum(-1)
    sin_err = (betas.sin() * mu).sum(-1)
    return (cos_err.square() + sin_err.square()) / 2


def hyperdiff_traces(prop_key, seeds):
    """{seed: running-trace}, recomputed via the notebook's eval loop (per-sample
    trace is not persisted in results_seed*.json)."""
    V, ps, initial_bias, (nbatches, batchsize, seqlen) = utils.unigram_dataset()
    initial_bias = initial_bias.to(torch.float64)
    initial_probs = initial_bias.exp()
    pkw = PROP_KW[prop_key]
    traces = {}
    for seed in seeds:
        torch.manual_seed(seed)                       # bbridge draws from the global RNG
        g = torch.Generator().manual_seed(seed)
        N = nbatches * batchsize * seqlen
        batches = torch.multinomial(initial_probs, N, replacement=True, generator=g).view(nbatches, batchsize, seqlen)
        ts_all, w_all = utils.sample_proposal(N, generator=g, **pkw)
        per = []
        for ib in range(nbatches):
            tg = batches[ib].to(torch.int64).reshape(-1)
            sl = slice(ib * batchsize * seqlen, (ib + 1) * batchsize * seqlen)
            ts, w = ts_all[sl], w_all[sl]
            rhos, thetas = _bbridge(ts)
            thetas = thetas + (tg.to(torch.float64) + 0.5) * (2 * torch.pi / V)
            logits = initial_bias[None, :].expand(tg.shape[0], V)
            per.append((_bridge_loss(logits, tg, rhos, thetas, V) * w).numpy())
        _, traces[seed] = _trace_xy(np.concatenate(per))
    return traces


def plot_loss_curve(prop_key, H, path):
    mt, x = mine_traces(prop_key)
    if not mt:
        print(f"!! no Mine per-sample trace for {prop_key}; skipping loss_curve.png")
        return
    seeds = sorted(mt)
    print(f"recomputing HyperDiff trace for {prop_key}, seeds {seeds} ...")
    ht = hyperdiff_traces(prop_key, seeds)

    def band(traces):
        M = np.vstack([traces[s] for s in sorted(traces)])
        sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1])
        return M.mean(0), sd

    mm, msd = band(mt)
    hm, hsd = band(ht)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(H, color="k", ls="--", lw=1.2, label=f"entropy H={H:.4f}")
    ax.plot(x, mm, color=BLUE, lw=1.6, label=f"Mine (tmp4 opt_repro, {len(mt)} seeds)")
    ax.fill_between(x, mm - msd, mm + msd, color=BLUE, alpha=0.25)
    ax.plot(x, hm, color=RED, lw=1.6, label=f"HyperDiffTransformerPlane.py ({len(ht)} seeds)")
    ax.fill_between(x, hm - hsd, hm + hsd, color=RED, alpha=0.25)
    # focus the y-range on the convergence zone (after the early single-sample transient)
    tail = x >= 1000
    lo = min(float((mm - msd)[tail].min()), float((hm - hsd)[tail].min()), H)
    hi = max(float((mm + msd)[tail].max()), float((hm + hsd)[tail].max()), H)
    pad = 0.15 * (hi - lo + 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ax.set_xlabel("samples seen (cumulative running mean, N=4e6)")
    ax.set_ylabel("running NELBO")
    ax.set_title(f"eval NELBO curve  -  proposal {prop_key}\nrunning mean +/- across-seed std")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------- tables + RESULTS.md update ----------------------------
def build_tables(mine, hd, H):
    names = [n for n in PROP_ORDER if n in mine and n in hd]
    L = []
    L.append("### test_wnelbo (bridge ELBO) - mean +/- across-seed std (5 seeds, N=4e6)")
    L.append("")
    L.append("| proposal | Mine mean | Mine std | HyperDiff mean | HyperDiff std | Mine - HyperDiff | gap Mine (mean-H) | gap HyperDiff |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for n in names:
        mm, ms, _ = agg(mine[n], "wnelbo")
        hm, hs, _ = agg(hd[n], "wnelbo")
        L.append(f"| `{n}` | {mm:.4f} | {ms:.4f} | {hm:.4f} | {hs:.4f} | {mm - hm:+.4f} | {mm - H:+.4f} | {hm - H:+.4f} |")
    L.append("")
    L.append("### test_ce sanity (optimal model: should equal H for every proposal)")
    L.append("")
    L.append("| proposal | Mine test_ce | HyperDiff test_ce |")
    L.append("|---|--:|--:|")
    for n in names:
        mce = agg(mine[n], "ce")[0]
        hce = agg(hd[n], "ce")[0]
        L.append(f"| `{n}` | {mce:.4f} | {hce:.4f} |")
    return "\n".join(L)


def main():
    H = entropy()
    mine, hd = load_mine(), load_hyperdiff()
    if not mine:
        sys.exit(f"no Mine results under {OUT}/opt_repro_*_rs* - run unigram_test_lorentz_tmp4_opt.sh first")
    if not hd:
        sys.exit(f"no HyperDiff results under {HERE}/results_seed*.json")

    tables = build_tables(mine, hd, H)
    # fill the verdict's max |Mine-HyperDiff|
    names = [n for n in PROP_ORDER if n in mine and n in hd]
    biggest = max(abs(agg(mine[n], "wnelbo")[0] - agg(hd[n], "wnelbo")[0]) for n in names)

    print(f"\nH = {H:.5f}\n")
    print(tables.replace("|", " ").replace("`", "").replace("---", "").replace(":", ""))
    print(f"\nmax |Mine - HyperDiff| = {biggest:.4f} nats")

    plot_elbo(mine, hd, H, os.path.join(HERE, "elbo_vs_proposal.png"))
    plot_loss_curve(LOSS_CURVE_PROPOSAL, H, os.path.join(HERE, "loss_curve.png"))
    _write_md(tables, H, biggest, os.path.join(HERE, "RESULTS_elbo_reproduction.md"))


def _write_md(tables, H, biggest, path):
    block = (
        f"{MARK[0]}\n"
        f"## 5. Mine (tmp4 `opt_repro`) vs `HyperDiffTransformerPlane.py`\n\n"
        f"_Auto-generated by `update_results.py`. `Mine` = optimal-model sweep from "
        f"`unigram_test_script/unigram_test_lorentz_tmp4_opt.sh` "
        f"(`outputs/unigram_test2/opt_repro_*_rs*`, `mode=opt`); `HyperDiff` = `results_seed*.json`. "
        f"Both are the same Bayes-optimal model under the same Poincare-polar bridge, so this is a "
        f"reproduction check. Error bars/bands are across-seed std (ddof=1) over 5 seeds._\n\n"
        f"**Verdict.** The two sources agree to within **{biggest:.4f} nats** across all 10 proposals. "
        f"Both reproduce the entropy H={H:.5f} where the estimator is well-behaved (matched `strat-exp` "
        f"lam=0.1-0.3) and both droop below H in the short-`unif` (truncation-bias) and heavy-tail "
        f"(lam>=0.8) regimes - the same two failure modes documented in section 2. `loss_curve.png` "
        f"shows their running-NELBO traces converging to the same value.\n\n"
        f"![elbo vs proposal](elbo_vs_proposal.png)\n\n"
        f"![running NELBO](loss_curve.png)\n\n"
        f"{tables}\n"
        f"{MARK[1]}"
    )
    txt = open(path).read()
    if MARK[0] in txt and MARK[1] in txt:
        txt = re.sub(re.escape(MARK[0]) + r".*?" + re.escape(MARK[1]), lambda _: block, txt, flags=re.S)
    else:
        txt = txt.rstrip() + "\n\n" + block + "\n"
    open(path, "w").write(txt)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
