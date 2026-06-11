"""Aggregate the optimal-model bridge-ELBO sweep (unigram_test_lorentz_tmp4_opt.sh):
mean +/- seed-std of test_wnelbo per proposal, vs the data entropy H. Compares
to the hyperbolic_dm_workspace reproduction (RESULTS_elbo_reproduction.md)."""
import json, re, glob, os, math, statistics

ROOT = "outputs/unigram_test2"
PS = [0.91] + [0.01] * 9
sp = sum(PS); p = [x / sp for x in PS]
H = -sum(pi * math.log(pi) for pi in p)

# hyperbolic_dm_workspace reproduction (mean over 5 seeds, N=4e6) for side-by-side.
REPRO = {
    "unif[0.01,10]": 0.4359, "unif[0.01,20]": 0.4881, "unif[0.01,30]": 0.4963, "unif[0.01,50]": 0.4973,
    "strat-exp λ=0.1": 0.5009, "strat-exp λ=0.2": 0.4990, "strat-exp λ=0.3": 0.4970,
    "strat-exp λ=0.5": 0.4958, "strat-exp λ=0.8": 0.4724, "strat-exp λ=1.0": 0.4442,
}

def prop(folder):
    pt = re.search(r"_pt([a-z_]+?)_per", folder); per = re.search(r"_per([0-9.]+)_", folder); hT = re.search(r"_hT([0-9.eE+]+)_", folder)
    pt = pt.group(1) if pt else "?"
    if pt == "unif":
        h = float(hT.group(1)) if hT else 0
        return (0, h, f"unif[0.01,{h*0.01:.0f}]")
    return (1, float(per.group(1)) if per else 0, f"strat-exp λ={per.group(1) if per else '?'}")

data = {}
for sweep in glob.glob(f"{ROOT}/opt_repro_*_rs*"):
    seed = (re.search(r"_rs(\d+)$", os.path.basename(sweep)) or [None, "?"])[1]
    for mf in glob.glob(f"{sweep}/*/test_metrics.json"):
        folder = os.path.basename(os.path.dirname(mf))
        try: m = json.load(open(mf))
        except Exception: continue
        data.setdefault(prop(folder), []).append((seed, m.get("test_wnelbo"), m.get("test_ce")))

def finite(xs): return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]

print(f"\nH = {H:.5f}   (entropy of ps={PS})\n")
print(f"{'proposal':<18}{'n':>3}{'wnelbo mean':>13}{'seed-std':>10}{'gap(mean-H)':>13}{'repro':>9}{'mine-repro':>12}{'ce mean':>10}")
print("-" * 98)
for k in sorted(data):
    rows = data[k]; label = k[2]
    wn = finite([r[1] for r in rows]); ce = finite([r[2] for r in rows])
    if not wn:
        print(f"{label:<18}{len(rows):>3}   (no finite test_wnelbo)"); continue
    mwn = statistics.mean(wn); swn = statistics.pstdev(wn) if len(wn) > 1 else 0.0
    mce = statistics.mean(ce) if ce else float("nan")
    rep = REPRO.get(label)
    rep_s = f"{rep:>9.4f}" if rep is not None else f"{'-':>9}"
    diff_s = f"{mwn-rep:>+12.4f}" if rep is not None else f"{'-':>12}"
    print(f"{label:<18}{len(wn):>3}{mwn:>13.4f}{swn:>10.4f}{mwn-H:>+13.4f}{rep_s}{diff_s}{mce:>10.4f}")
print("\n(seeds detected:", sorted({r[0] for rows in data.values() for r in rows}), ")")
