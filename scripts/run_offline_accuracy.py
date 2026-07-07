"""run_offline_accuracy.py — P7: try to lower the few-shot MAE at small budgets (m=2,3) with
(a) empirical-Bayes shrinkage of the per-tool wear-rate toward the population, and
(b) a SENSOR-informed prior on the rate (sensors predict the rate scale, not VB directly).
Pre-stated rule: adopt a variant ONLY if it reduces LOTO MAE vs the deployed few-shot at the same m.
LOTO, leakage-safe, wear regime VB<=300."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
CENSOR = 300.0
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")


def tool_feature_means():
    """Per-tool aggregated sensor features (mean over that tool's cuts)."""
    f = pd.read_csv(FEAT)
    drop = {c for c in f.columns if f[c].dtype == object} | {"within_tool_order", "vb_um"}
    num = [c for c in f.columns if c not in drop and c != "tool_id"]
    g = f.groupby("tool_id")[num].mean()
    g = g.loc[:, g.std() > 1e-9]      # drop constant cols
    return g


def full_rate(g, p):
    """Theil-Sen rate a on a tool's full (censored) trajectory at exponent p."""
    gg = g[g.vb <= CENSOR]
    if len(gg) < 2:
        return np.nan
    a, _ = theil_sen(gg.order.to_numpy(float) ** p, gg.vb.to_numpy(float))
    return a


def evaluate(d, m, mode, lam=0.0, feats=None):
    """mode: 'base' | 'eb' (shrink to pop) | 'sensor' (shrink to sensor-predicted rate)."""
    per = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr)
        x = o[:m] ** p
        a_hat, _ = theil_sen(x, v[:m])
        if mode == "base":
            a_use = a_hat
        else:
            rates = {t: full_rate(gt.sort_values("order"), p) for t, gt in tr.groupby("tool_id")}
            rates = {t: r for t, r in rates.items() if np.isfinite(r)}
            a_pop = float(np.median(list(rates.values())))
            if mode == "eb":
                a_prior = a_pop
            else:  # sensor-informed prior on the rate
                ids = [t for t in rates if t in feats.index]
                X = feats.loc[ids].to_numpy(float); y = np.array([rates[t] for t in ids])
                sc = StandardScaler().fit(X)
                rg = Ridge(alpha=10.0).fit(sc.transform(X), y)
                a_prior = float(rg.predict(sc.transform(feats.loc[[tt]].to_numpy(float)))[0]) \
                    if tt in feats.index else a_pop
            a_use = (1 - lam) * a_hat + lam * a_prior
        b_use = float(np.median(v[:m] - a_use * x))
        pred = b_use + a_use * o[fut] ** p
        per.append(np.abs(pred - v[fut]).mean())
    return float(np.mean(per)), len(per)


def main():
    d = load(); feats = tool_feature_means()
    print("Deployed few-shot baseline vs shrinkage variants (LOTO MAE, µm). Lower is better.\n")
    print(f"{'m':>2} {'baseline':>9} | {'EB best (λ)':>14} | {'sensor best (λ)':>16}")
    rows = []
    for m in (2, 3):
        base, n = evaluate(d, m, "base")
        eb = [(lam, evaluate(d, m, "eb", lam)[0]) for lam in (0.2, 0.4, 0.6, 0.8)]
        sen = [(lam, evaluate(d, m, "sensor", lam, feats)[0]) for lam in (0.2, 0.4, 0.6, 0.8)]
        eb_best = min(eb, key=lambda z: z[1]); sen_best = min(sen, key=lambda z: z[1])
        print(f"{m:>2} {base:9.2f} | {eb_best[1]:8.2f} (λ={eb_best[0]}) | {sen_best[1]:9.2f} (λ={sen_best[0]})")
        rows.append(dict(m=m, baseline=round(base, 2), eb=round(eb_best[1], 2), eb_lam=eb_best[0],
                         sensor=round(sen_best[1], 2), sensor_lam=sen_best[0], n=n))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "offline_accuracy.csv"), index=False)
    print("\nVerdict (pre-stated rule: adopt only if it beats baseline at same m):")
    for r in rows:
        win = min(("EB", r["eb"]), ("sensor", r["sensor"]), key=lambda z: z[1])
        verdict = (f"{win[0]} improves {r['baseline']}→{win[1]}" if win[1] < r["baseline"] - 0.05
                   else "NO variant beats baseline — keep plain few-shot")
        print(f"  m={r['m']}: {verdict}")
    print("\nwrote results/offline_accuracy.csv")


if __name__ == "__main__":
    main()
