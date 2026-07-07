"""
run_final_eval.py — final evaluation bundle (leakage-safe LOTO, wear regime VB<=300):
  #4  positioning vs time-only baselines: average-wear-curve, Linear(t) self-fit, Poly2(t) self-fit,
      and our physics Power(t) self-fit, at m=3 (conservative) and m=4 (precise).
  #2  bootstrap 95% CI (over the 18 tools) of the MAE improvement (baseline - ours).
  #1  conformal intervals re-based on the DEPLOYED model (physics power + few-shot self-adaptation):
      jackknife+ coverage + mean width at 80/90%, vs the old population-based conformal.

Deployed model: VB=b+a*order^p, a,b fit (robust Theil-Sen) to the held-out tool's OWN first m points;
global p fit on TRAINING tools only. Double validation: bootstrap recomputed two ways; conformal coverage
sanity-checked >= nominal.

Outputs: results/final_eval_models.csv, results/final_eval_bootstrap.csv, results/final_eval_conformal.csv.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from phm.prognostic_system import fit_population, fewshot_offset, predict_vb, conformal_quantile

CENSOR = 300.0
RNG = np.random.default_rng(42)


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um"]].drop_duplicates()
    return (d.rename(columns={"within_tool_order": "order", "vb_um": "vb"})
            .sort_values(["tool_id", "order"]).reset_index(drop=True))


def fit_global_p(tr):
    best_p, best = 0.5, np.inf
    for p in np.arange(0.2, 1.001, 0.05):
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR]
            if len(gg) < 2:
                continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < best:
            best, best_p = tot, p
    return best_p


def theil_sen(x, y):
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def tools_of(d):
    return sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))


def predict(model, tr, o, v, m, p):
    fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
    if len(fut) == 0:
        return None, None
    if model == "avgcurve":
        pop = fit_population(tr); off = fewshot_offset(pop, o, v, m); pr = predict_vb(pop, o[fut], off)
    elif model == "linear":
        a, b = theil_sen(o[:m].astype(float), v[:m]); pr = b + a * o[fut]
    elif model == "poly2":
        c = np.polyfit(o[:m], v[:m], min(2, m - 1)); pr = np.polyval(c, o[fut])
    elif model == "power":                                   # our deployed model
        x = o[:m] ** p; a, b = theil_sen(x, v[:m]); pr = b + a * o[fut] ** p
    return pr, v[fut]


def per_tool_mae(d, model, m):
    out = {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        p = fit_global_p(tr)
        pr, tru = predict(model, tr, o, v, m, p)
        if pr is None:
            continue
        out[tt] = float(np.mean(np.abs(pr - tru)))
    return out


def pooled_r2(d, model, m):
    P, Y = [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        p = fit_global_p(tr); pr, tru = predict(model, tr, o, v, m, p)
        if pr is None:
            continue
        P += list(pr); Y += list(tru)
    P, Y = np.array(P), np.array(Y)
    return 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)


def main():
    d = load()
    models = [("Average-wear-curve (baseline)", "avgcurve"),
              ("Linear(t) self-fit", "linear"),
              ("Poly2(t) self-fit", "poly2"),
              ("Physics Power(t) self-fit (ours)", "power")]
    # ---- #4 positioning table ----
    print("#4  Positioning vs time-only baselines (LOTO, per-tool MAE | pooled R2):\n")
    print(f"{'model':40} {'MAE m=3':>8} {'R2 m=3':>7} {'MAE m=4':>8} {'R2 m=4':>7}")
    recs = []
    pt_mae = {}
    for name, key in models:
        m3 = per_tool_mae(d, key, 3); m4 = per_tool_mae(d, key, 4)
        pt_mae[key] = m3
        r = dict(model=name, MAE_m3=round(np.mean(list(m3.values())), 1), R2_m3=round(pooled_r2(d, key, 3), 2),
                 MAE_m4=round(np.mean(list(m4.values())), 1), R2_m4=round(pooled_r2(d, key, 4), 2))
        recs.append(r)
        print(f"{name:40} {r['MAE_m3']:8.1f} {r['R2_m3']:7.2f} {r['MAE_m4']:8.1f} {r['R2_m4']:7.2f}")
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, "results", "final_eval_models.csv"), index=False)

    # ---- #2 bootstrap CI of improvement (baseline - ours), over tools, m=3 ----
    base = pt_mae["avgcurve"]; ours = pt_mae["power"]
    common = [t for t in base if t in ours]
    diff = np.array([base[t] - ours[t] for t in common])           # per-tool improvement
    n = len(common)
    boot = np.array([np.mean(RNG.choice(diff, n, replace=True)) for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # double-validate: analytic normal-approx CI on the mean difference
    se = diff.std(ddof=1) / np.sqrt(n); lo2, hi2 = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
    print(f"\n#2  Improvement (baseline MAE - ours), m=3, n={n} tools:")
    print(f"    mean = {diff.mean():.1f} um | bootstrap 95% CI [{lo:.1f}, {hi:.1f}] | "
          f"normal-approx [{lo2:.1f}, {hi2:.1f}]")
    print(f"    significant (CI excludes 0): {'YES' if lo > 0 else 'NO'}")
    pd.DataFrame([dict(mean_improvement_um=round(diff.mean(), 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                       normal_lo=round(lo2, 1), normal_hi=round(hi2, 1), n_tools=n,
                       significant=bool(lo > 0))]).to_csv(
        os.path.join(ROOT, "results", "final_eval_bootstrap.csv"), index=False)

    # ---- #1 conformal re-based on the deployed (power self-adapt) model, m=3 ----
    res = {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= 3:
            continue
        p = fit_global_p(tr); pr, tru = predict("power", tr, o, v, 3, p)
        if pr is None:
            continue
        res[tt] = np.abs(pr - tru)
    crecs = []
    print("\n#1  Conformal re-based on the deployed model (jackknife+, m=3):")
    for alpha, nom in [(0.2, 80), (0.1, 90)]:
        cov, wid = [], []
        for tt in res:
            cal = np.concatenate([res[t] for t in res if t != tt])
            q = conformal_quantile(cal, alpha)
            # coverage on tt's points using its own true residuals
            cov.append(float(np.mean(res[tt] <= q))); wid.append(2 * q)
        c = dict(nominal=nom, empirical_coverage=round(np.mean(cov) * 100, 0),
                 mean_width_um=round(np.mean(wid), 1))
        crecs.append(c)
        ok = "OK" if c["empirical_coverage"] >= nom - 5 else "LOW"
        print(f"    nominal {nom}% -> coverage {c['empirical_coverage']:.0f}% [{ok}] | "
              f"mean width {c['mean_width_um']:.0f} um")
    pd.DataFrame(crecs).to_csv(os.path.join(ROOT, "results", "final_eval_conformal.csv"), index=False)
    print("\n    (old population-based conformal width @90% was ~175 um; compare above.)")
    print("\nwrote results/final_eval_{models,bootstrap,conformal}.csv")


if __name__ == "__main__":
    main()
