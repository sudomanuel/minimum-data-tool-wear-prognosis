# -*- coding: utf-8 -*-
"""run_p3_probe_estimator.py — PAPER 3 PIVOT · cheap probe of the estimator margin.

THE OPENING. Paper 1 states in its own record code:
    "gamma>3 lowers MAE further but is not adopted (conservative, avoids over-tuning)"
It stopped at gamma=3 because larger gamma keeps lowering MAE while DEGRADING the pooled R^2:
weighting hard toward the last support points sharpens the near horizon and destabilises the far
one. Paper 1 could not have both, so it stopped. That unresolved trade-off is the margin.

THE IDEA UNDER TEST: the trade-off is an artefact of ONE GLOBAL gamma. The optimal weighting is a
function of how far ahead we are predicting. A horizon-adaptive weight gamma(h) should sharpen the
near horizon (where Paper 1 pays for stopping at 3) without destabilising the far horizon (where
Paper 1's stopping rule was protecting the pooled R^2).

WHAT THIS PROBE MEASURES (no manuscript is written until it answers):
  1. reproduce the record exactly (sanity: gamma=3 -> 5.63 at m=4, 11.02 at m=3)
  2. the full gamma curve in BOTH metrics (MAE and pooled R^2) -> quantify the trade-off
  3. horizon-adaptive gamma(h) -> does it break the trade-off?
  4. an exponent chosen by EXTRAPOLATION skill instead of in-record fit (a second, independent
     margin: Paper 1 selects p by pooled SSE of the fit, never by forecast error)
All selection is done on training folds only; the held-out tool is never used to choose anything.
Output: results/p3_probe_estimator.csv
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from run_mcurve import load, theil_sen, tools_of
from run_optimal_config_search import fit_p
from run_record_attempts2 import wls_gamma, local_p_grid

CENSOR = 300.0
REC = {3: 11.02, 4: 5.63}


def pooled_r2(res):
    P = np.concatenate([r[0] + 0 for r in res.values()])      # signed residuals
    Y = np.concatenate([r[3] for r in res.values()])          # truths
    return 1 - np.sum(P ** 2) / np.sum((Y - Y.mean()) ** 2)


def evaluate(d, m, gamma_fn, ps="m_matched", use_local_p=True, p_mode="fit"):
    """gamma_fn(h) -> weighting exponent for horizon h (h = 1,2,... ahead of the support).
    p_mode: 'fit' = Paper-1 pooled-SSE exponent; 'extrap' = exponent chosen by forecast skill
    on the TRAINING tools only."""
    res = {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]
        g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o))
        fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        if p_mode == "extrap":
            p = fit_p_by_extrapolation(tr, m)
        else:
            p = fit_p(tr, m=(m if ps == "m_matched" else None))
        if use_local_p:
            p = local_p_grid(o, v, m, p, lambda t, y: wls_gamma(t, y, 1.0))
        tau = o[:m] ** p
        preds = np.empty(len(fut))
        for k, j in enumerate(fut):
            h = j - (m - 1)                      # steps ahead
            a, b = wls_gamma(tau, v[:m], gamma_fn(h))
            preds[k] = b + a * o[j] ** p
        sr = preds - v[fut]
        res[tt] = (sr, (fut - (m - 1)).astype(float), np.abs(sr), v[fut])
    return res


def fit_p_by_extrapolation(tr, m):
    """Choose the fleet exponent by FORECAST skill on the training tools (leave-one-in-train-out),
    not by in-record fit. Paper 1 never does this."""
    best_p, best = 0.20, np.inf
    tls = tools_of(tr)
    for p in np.arange(0.20, 1.001, 0.05):
        errs = []
        for t in tls:
            g = tr[tr.tool_id == t].sort_values("order")
            o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
            if len(o) <= m:
                continue
            tau = o[:m] ** p
            a, b = wls_gamma(tau, v[:m], 1.0)
            fut = np.arange(m, len(o))
            fut = fut[v[fut] <= CENSOR]
            if len(fut) == 0:
                continue
            errs.append(np.mean(np.abs(b + a * o[fut] ** p - v[fut])))
        if errs and np.mean(errs) < best:
            best, best_p = float(np.mean(errs)), p
    return best_p


def mae_of(res):
    return float(np.mean([r[2].mean() for r in res.values()]))


def main():
    d = load()
    rows = []

    def report(tag, m, res, note=""):
        mae, r2 = mae_of(res), pooled_r2(res)
        beats = mae < REC[m]
        rows.append(dict(tag=tag, m=m, MAE=round(mae, 2), pooled_R2=round(r2, 3),
                         record=REC[m], beats_record=bool(beats), note=note))
        print(f"  {tag:42s} m={m} MAE {mae:6.2f}  R2 {r2:6.3f}  "
              f"{'** BEATS RECORD **' if beats else ''}", flush=True)
        return mae, r2

    print("=== 1 · reproduce the Paper-1 record (sanity) ===", flush=True)
    for m in (3, 4):
        ps = "m_matched" if m == 4 else "full"
        report(f"record replication (gamma=3)", m,
               evaluate(d, m, lambda h: 3.0, ps=ps), "should match 11.02 / 5.63")

    print("\n=== 2 · the trade-off Paper 1 stopped at: gamma curve in BOTH metrics ===", flush=True)
    for m in (3, 4):
        ps = "m_matched" if m == 4 else "full"
        for gam in (1, 2, 3, 4, 5, 6, 8):
            report(f"global gamma={gam}", m, evaluate(d, m, lambda h, gg=gam: float(gg), ps=ps),
                   "P1 stopped at 3 to protect pooled R2")

    print("\n=== 3 · IDEA: horizon-adaptive gamma(h) — break the trade-off ===", flush=True)
    schemes = {
        "gamma(h)=6/h  (sharp near, anchored far)": lambda h: 6.0 / max(h, 1),
        "gamma(h)=8/h": lambda h: 8.0 / max(h, 1),
        "gamma(h)=6/sqrt(h)": lambda h: 6.0 / np.sqrt(max(h, 1)),
        "gamma(h)=max(3, 8-h)": lambda h: max(3.0, 8.0 - h),
        "gamma(h)=3+4*exp(-h/2)": lambda h: 3.0 + 4.0 * np.exp(-h / 2.0),
    }
    for m in (3, 4):
        ps = "m_matched" if m == 4 else "full"
        for name, fn in schemes.items():
            report(name, m, evaluate(d, m, fn, ps=ps), "horizon-adaptive weighting")

    print("\n=== 4 · second margin: exponent chosen by EXTRAPOLATION skill ===", flush=True)
    for m in (3, 4):
        ps = "m_matched" if m == 4 else "full"
        report("p by extrapolation + gamma=3", m,
               evaluate(d, m, lambda h: 3.0, ps=ps, p_mode="extrap"), "P1 selects p by fit only")
        report("p by extrapolation + gamma(h)=6/h", m,
               evaluate(d, m, lambda h: 6.0 / max(h, 1), ps=ps, p_mode="extrap"), "combined")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "p3_probe_estimator.csv"), index=False)
    print("\n=== VERDICT ===", flush=True)
    win = out[out.beats_record]
    if len(win):
        print(win.sort_values("MAE")[["tag", "m", "MAE", "pooled_R2", "record"]].to_string(index=False))
    else:
        print("no configuration beats the record — the margin is not here.")
    print("\nwrote results/p3_probe_estimator.csv")


if __name__ == "__main__":
    main()
