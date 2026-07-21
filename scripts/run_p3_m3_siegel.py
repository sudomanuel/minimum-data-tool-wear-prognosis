# -*- coding: utf-8 -*-
"""run_p3_m3_siegel.py — close the m=3 flank of the Paper-3 pivot.

WHY. The m=4 record (5.63 um) is a weighted least-squares fit, so horizon-adaptive weighting
applies to it directly and beat it (3.57 um nested). The m=3 record (11.02 um) is a different
estimator: a Siegel repeated-median with a locally shrunk exponent. Comparing a WLS variant against
it is not a like-for-like comparison, so the m=3 row cannot be claimed until the record is
reproduced with ITS OWN estimator and the same idea is applied to it.

WHAT IS TESTED
  1. exact replication of the m=3 record (Siegel + local p) -> must land on 11.02 um;
  2. WEIGHTED repeated-median: the same robust estimator, but each point's slope contributions are
     weighted by tau^gamma(h), so robustness is kept while the fit is sharpened toward the horizon
     being predicted. gamma(h)=const recovers the record as a special case;
  3. the same horizon-adaptive schemes that won at m=4.
Selection honesty is enforced by the nested audit in run_p3_nested_audit.py; this script measures
the in-search picture that decides whether m=3 is claimable at all.
Output: results/p3_m3_siegel.csv
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

from run_mcurve import load, tools_of, theil_sen
from run_optimal_config_search import fit_p, siegel
from run_record_attempts2 import local_p_grid, wls_gamma

CENSOR = 300.0
REC3 = 11.02


def weighted_repeated_median(x, y, w):
    """Siegel repeated-median with per-point weights.

    Siegel takes, for each point i, the median of the slopes to every other point, then the median
    of those per-point medians. Here each stage uses a WEIGHTED median: slopes from a point with a
    larger weight (a point closer to the forecast horizon) count more. w = 1 recovers Siegel.
    """
    n = len(x)
    med_i, wi = [], []
    for i in range(n):
        sl, sw = [], []
        for j in range(n):
            if j != i and x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
                sw.append(w[j])
        if sl:
            med_i.append(_wmedian(np.array(sl), np.array(sw)))
            wi.append(w[i])
    if not med_i:
        return 0.0, float(np.median(y))
    a = _wmedian(np.array(med_i), np.array(wi))
    b = _wmedian(y - a * x, w)
    return float(a), float(b)


def _wmedian(v, w):
    idx = np.argsort(v)
    v, w = v[idx], np.maximum(w[idx], 1e-12)
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])


def evaluate_m3(d, mode, gamma_fn=None):
    """mode: 'siegel_record' | 'wrm' (weighted repeated-median) | 'wls'."""
    m = 3
    per, res, hz, tru = [], [], [], []
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
        p = fit_p(tr, m=None)                     # 'full' strategy, as the record uses
        p = local_p_grid(o, v, m, p, lambda t, y: (siegel(t, y)))
        tau = o[:m] ** p
        preds = np.empty(len(fut))
        for k, j in enumerate(fut):
            h = j - (m - 1)
            if mode == "siegel_record":
                a, b = siegel(tau, v[:m])
            elif mode == "wrm":
                w = np.maximum(tau, 1e-9) ** gamma_fn(h)
                a, b = weighted_repeated_median(tau, v[:m], w)
            else:
                a, b = wls_gamma(tau, v[:m], gamma_fn(h))
            preds[k] = b + a * o[j] ** p
        sr = preds - v[fut]
        per.append(float(np.mean(np.abs(sr))))
        res.extend(sr); hz.extend((fut - (m - 1)).astype(float)); tru.extend(v[fut])
    res, tru = np.array(res), np.array(tru)
    r2 = 1 - np.sum(res ** 2) / np.sum((tru - tru.mean()) ** 2)
    return float(np.mean(per)), float(r2)


def main():
    d = load()
    rows = []

    def rep(tag, mae, r2, note=""):
        beats = mae < REC3
        rows.append(dict(tag=tag, m=3, MAE=round(mae, 2), pooled_R2=round(r2, 3),
                         record=REC3, beats_record=bool(beats), note=note))
        print(f"  {tag:46s} MAE {mae:6.2f}  R2 {r2:6.3f}  "
              f"{'** BEATS RECORD **' if beats else ''}", flush=True)

    print("=== 1 · replicate the m=3 record with ITS OWN estimator (Siegel + local p) ===",
          flush=True)
    mae, r2 = evaluate_m3(d, "siegel_record")
    rep("Siegel repeated-median (record replication)", mae, r2, "target 11.02")

    print("\n=== 2 · weighted repeated-median: same robustness, horizon-aware weights ===",
          flush=True)
    schemes = {
        "WRM gamma(h)=1 (weight-free control)": lambda h: 0.0,
        "WRM gamma(h)=3": lambda h: 3.0,
        "WRM gamma(h)=6/h": lambda h: 6.0 / max(h, 1),
        "WRM gamma(h)=8/h": lambda h: 8.0 / max(h, 1),
        "WRM gamma(h)=6/sqrt(h)": lambda h: 6.0 / np.sqrt(max(h, 1)),
        "WRM gamma(h)=max(3,8-h)": lambda h: max(3.0, 8.0 - h),
    }
    for name, fn in schemes.items():
        mae, r2 = evaluate_m3(d, "wrm", fn)
        rep(name, mae, r2, "weighted Siegel")

    print("\n=== 3 · WLS variants at m=3 (not the record's estimator; context only) ===",
          flush=True)
    for name, fn in [("WLS gamma(h)=6/sqrt(h)", lambda h: 6.0 / np.sqrt(max(h, 1))),
                     ("WLS gamma(h)=8/h", lambda h: 8.0 / max(h, 1))]:
        mae, r2 = evaluate_m3(d, "wls", fn)
        rep(name, mae, r2, "different estimator family")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "p3_m3_siegel.csv"), index=False)
    print("\n=== VERDICT m=3 ===")
    w = out[out.beats_record].sort_values("MAE")
    if len(w):
        print(w[["tag", "MAE", "pooled_R2", "record"]].to_string(index=False))
    else:
        print("nothing beats the m=3 record -> the paper claims m=4 only, and says so.")
    print("\nwrote results/p3_m3_siegel.csv")


if __name__ == "__main__":
    main()
