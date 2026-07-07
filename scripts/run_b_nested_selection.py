"""run_b_nested_selection.py — Reviewer-response B: selection-honest (nested double-LOTO) estimate of
the deployed configuration's accuracy.

Objection addressed: the joint-optimal configuration was selected by LOTO score on the same 18 tools
it is reported on (selection optimism). Remedy without new data: for each OUTER held-out tool, re-run
the configuration search using only the remaining 17 tools (each candidate scored by INNER LOTO with
the same validity constraint: inner Mondrian coverage >= 88%), then apply the inner winner — chosen
blind to the outer tool — to that outer tool. The outer mean is the selection-honest accuracy.

Config space mirrors the historical search (Sections 3.8 + record rounds):
  estimator  : theil_sen | siegel | ols | wls(tau^gamma), gamma in {1,2,3}
  p strategy : full-trajectory | m-matched
  local p    : off | on (quadratic-penalty local shrinkage)
  EB shrink  : lambda in {0, 0.2} (tool rate toward population rate)
  -> 48 configurations per budget m in {2,3,4}.

Outputs: results/b_nested_selection.csv (+ per-fold winners)
"""
import os, sys, itertools, collections
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, theil_sen, tools_of
from run_optimal_config_search import fit_p, FIT
from run_record_attempts import band_eval
CENSOR = 300.0; COV_MIN = 88.0
REFERENCE = {2: 12.7, 3: 11.0, 4: 5.6}   # in-search optima (Table 2 / m-curve)


def wls_gamma_fit(gamma):
    def f(tau, y):
        w = np.maximum(tau, 1e-9) ** gamma
        W = np.sqrt(w)
        A = np.column_stack([W, W * tau])
        c, *_ = np.linalg.lstsq(A, W * y, rcond=None)
        return float(c[1]), float(c[0])
    return f


ESTIMATORS = {"theil_sen": FIT["theil_sen"], "siegel": FIT["siegel"], "ols": FIT["ols"],
              "wls_g1": wls_gamma_fit(1.0), "wls_g2": wls_gamma_fit(2.0), "wls_g3": wls_gamma_fit(3.0)}
CONFIGS = [dict(est=e, ps=ps, local=lc, eb=eb)
           for e, ps, lc, eb in itertools.product(ESTIMATORS, ("full", "m_matched"),
                                                  (False, True), (0.0, 0.2))]


def local_p_grid(o, v, m, p_star, fitfn, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


class Cache:
    def __init__(self, d):
        self.d = d; self.p = {}; self.rates = {}

    def fit_p(self, train_ids, m, ps):
        key = (train_ids, m if ps == "m_matched" else None)
        if key not in self.p:
            tr = self.d[self.d.tool_id.isin(train_ids)]
            self.p[key] = fit_p(tr, m=(m if ps == "m_matched" else None))
        return self.p[key]

    def pop_rate(self, train_ids, p):
        key = (train_ids, round(p, 3))
        if key not in self.rates:
            rs = []
            for _, gt in self.d[self.d.tool_id.isin(train_ids)].groupby("tool_id"):
                gg = gt[gt.vb <= CENSOR].sort_values("order")
                if len(gg) >= 2:
                    a, _ = theil_sen(gg.order.to_numpy(float) ** p, gg.vb.to_numpy(float))
                    rs.append(a)
            self.rates[key] = float(np.median(rs))
        return self.rates[key]


def eval_pool(d, cache, pool, m, cfg):
    """LOTO of `cfg` within `pool`; returns res dict {tool: (signed, horizon, abs)}."""
    fitfn = ESTIMATORS[cfg["est"]]
    res = {}
    for u in pool:
        train_ids = frozenset(t for t in pool if t != u)
        g = d[d.tool_id == u].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = cache.fit_p(train_ids, m, cfg["ps"])
        if cfg["local"]:
            p = local_p_grid(o, v, m, p, fitfn)
        tau = o[:m] ** p
        a, b = fitfn(tau, v[:m])
        if cfg["eb"] > 0:
            a = (1 - cfg["eb"]) * a + cfg["eb"] * cache.pop_rate(train_ids, p)
            b = float(np.median(v[:m] - a * tau))
        sr = (b + a * o[fut] ** p) - v[fut]
        res[u] = (sr, (fut - (m - 1)).astype(float), np.abs(sr))
    return res


def mae_of(res):
    return float(np.mean([r[2].mean() for r in res.values()])) if res else np.inf


def main():
    d = load(); cache = Cache(d)
    alltools = tools_of(d)
    print("B: NESTED DOUBLE-LOTO CONFIGURATION SELECTION (selection-honest accuracy)")
    print(f"{len(CONFIGS)} configs x {len(alltools)} outer folds x inner LOTO(17) | validity: inner "
          f"Mondrian coverage >= {COV_MIN}%\n")
    out_rows, win_rows = [], []
    for m in (2, 3, 4):
        outer_maes, winners = [], []
        for tt in alltools:
            pool = tuple(t for t in alltools if t != tt)
            best, best_cfg = np.inf, None
            for cfg in CONFIGS:
                res = eval_pool(d, cache, pool, m, cfg)
                if not res:
                    continue
                sc = mae_of(res)
                if sc < best:
                    cov, _ = band_eval(res, False)
                    if cov >= COV_MIN:
                        best, best_cfg = sc, cfg
            # apply the (blind) inner winner to the outer tool
            fitfn = ESTIMATORS[best_cfg["est"]]
            g = d[d.tool_id == tt].sort_values("order")
            o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
            if len(o) <= m:
                continue
            fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
            if len(fut) == 0:
                continue
            train_ids = frozenset(pool)
            p = cache.fit_p(train_ids, m, best_cfg["ps"])
            if best_cfg["local"]:
                p = local_p_grid(o, v, m, p, fitfn)
            tau = o[:m] ** p
            a, b = fitfn(tau, v[:m])
            if best_cfg["eb"] > 0:
                a = (1 - best_cfg["eb"]) * a + best_cfg["eb"] * cache.pop_rate(train_ids, p)
                b = float(np.median(v[:m] - a * tau))
            mae = float(np.mean(np.abs((b + a * o[fut] ** p) - v[fut])))
            outer_maes.append(mae)
            wname = f'{best_cfg["est"]}|{best_cfg["ps"]}|{"loc" if best_cfg["local"] else "-"}|eb{best_cfg["eb"]}'
            winners.append(wname)
            win_rows.append(dict(m=m, outer_tool=tt, winner=wname, outer_mae=round(mae, 2)))
        nested = float(np.mean(outer_maes))
        top = collections.Counter(winners).most_common(3)
        print(f"m={m}: NESTED MAE {nested:6.2f} um  (in-search optimum {REFERENCE[m]})  "
              f"[n={len(outer_maes)} folds]")
        print(f"      winner stability: {top}\n")
        out_rows.append(dict(m=m, nested_MAE=round(nested, 2), insearch_MAE=REFERENCE[m],
                             optimism_um=round(nested - REFERENCE[m], 2),
                             top_winner=top[0][0], top_winner_share=round(top[0][1] / len(winners), 2)))
    pd.DataFrame(out_rows).to_csv(os.path.join(ROOT, "results", "b_nested_selection.csv"), index=False)
    pd.DataFrame(win_rows).to_csv(os.path.join(ROOT, "results", "b_nested_winners.csv"), index=False)
    print("wrote results/b_nested_selection.csv, results/b_nested_winners.csv")


if __name__ == "__main__":
    main()
