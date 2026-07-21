# -*- coding: utf-8 -*-
"""run_p3_nested_audit.py — the two checks that decide whether the horizon-adaptive result is real.

The probe found gamma(h)=8/h at 3.63 um against the Paper-1 record of 5.63 um with the pooled R^2
preserved. But the scheme was chosen by looking at the outer scores, which is precisely the
selection optimism Paper 1 audited for its own record. Nothing is claimed until:

  CHECK 1 · NESTED SELECTION. For each held-out tool, the whole scheme search is re-run blind on
    the remaining 17 tools and the inner winner is applied to the outer tool. If the nested MAE
    matches the in-search MAE, the result is selection-robust; if it degrades, it was over-tuned.

  CHECK 2 · CONFORMAL VALIDITY. Paper 1's pre-stated adoption rule requires VALID coverage, not
    just lower error. The horizon-binned (Mondrian) band is recalibrated on the new estimator's
    out-of-fold residuals and its empirical coverage is measured against the 90% target.

Output: results/p3_nested_audit.csv
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

from run_mcurve import load, tools_of
from run_optimal_config_search import fit_p
from run_record_attempts2 import wls_gamma, local_p_grid

CENSOR = 300.0
REC = {3: 11.02, 4: 5.63}
ALPHA = 0.10

# the candidate family the search ranges over (identical inner and outer)
SCHEMES = {
    "global gamma=3 (Paper-1 record)": lambda h: 3.0,
    "global gamma=4": lambda h: 4.0,
    "global gamma=6": lambda h: 6.0,
    "gamma(h)=6/h": lambda h: 6.0 / max(h, 1),
    "gamma(h)=8/h": lambda h: 8.0 / max(h, 1),
    "gamma(h)=6/sqrt(h)": lambda h: 6.0 / np.sqrt(max(h, 1)),
    "gamma(h)=max(3,8-h)": lambda h: max(3.0, 8.0 - h),
    "gamma(h)=3+4*exp(-h/2)": lambda h: 3.0 + 4.0 * np.exp(-h / 2.0),
}


def forecast_tool(d_train, o, v, m, gamma_fn, ps="m_matched"):
    """Predict the sealed continuation of one tool. Everything fitted on d_train only."""
    p = fit_p(d_train, m=(m if ps == "m_matched" else None))
    p = local_p_grid(o, v, m, p, lambda t, y: wls_gamma(t, y, 1.0))
    tau = o[:m] ** p
    fut = np.arange(m, len(o))
    fut = fut[v[fut] <= CENSOR]
    if len(fut) == 0:
        return None, None, None
    preds = np.empty(len(fut))
    for k, j in enumerate(fut):
        a, b = wls_gamma(tau, v[:m], gamma_fn(j - (m - 1)))
        preds[k] = b + a * o[j] ** p
    return preds, v[fut], (fut - (m - 1)).astype(float)


def score(d, tools, m, gamma_fn, ps):
    per, res, hz, tru = [], [], [], []
    for tt in tools:
        tr = d[d.tool_id != tt]
        g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        pr, tv, h = forecast_tool(tr, o, v, m, gamma_fn, ps)
        if pr is None:
            continue
        per.append(float(np.mean(np.abs(pr - tv))))
        res.extend(pr - tv); hz.extend(h); tru.extend(tv)
    return np.array(per), np.array(res), np.array(hz), np.array(tru)


def hbin(h):
    return 0 if h <= 1 else (1 if h <= 3 else 2)


def main():
    d = load()
    tools = tools_of(d)
    rows = []

    for m in (3, 4):
        ps = "m_matched" if m == 4 else "full"
        print(f"\n{'='*74}\nm = {m}   (record {REC[m]} um)\n{'='*74}", flush=True)

        # ---------- in-search scores (what the probe reported) ----------
        insearch = {}
        for name, fn in SCHEMES.items():
            per, _, _, _ = score(d, tools, m, fn, ps)
            insearch[name] = float(np.mean(per))
        best_name = min(insearch, key=insearch.get)
        print(f"  in-search winner: {best_name}  ->  {insearch[best_name]:.2f} um", flush=True)

        # ---------- CHECK 1: nested selection ----------
        nested_err, winners = [], []
        for tt in tools:
            inner_tools = [t for t in tools if t != tt]
            d_inner = d[d.tool_id != tt]
            best, bname = np.inf, None
            for name, fn in SCHEMES.items():
                per, _, _, _ = score(d_inner, inner_tools, m, fn, ps)   # blind to tt
                if len(per) and np.mean(per) < best:
                    best, bname = float(np.mean(per)), name
            winners.append(bname)
            g = d[d.tool_id == tt].sort_values("order")
            o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
            if len(o) <= m:
                continue
            pr, tv, _ = forecast_tool(d[d.tool_id != tt], o, v, m, SCHEMES[bname], ps)
            if pr is None:
                continue
            nested_err.append(float(np.mean(np.abs(pr - tv))))
        nested = float(np.mean(nested_err))
        share = pd.Series(winners).value_counts(normalize=True)
        print(f"  NESTED  MAE {nested:6.2f} um   (optimism {nested - insearch[best_name]:+.2f})",
              flush=True)
        print(f"  nested winner share: {share.index[0]} in {share.iloc[0]*100:.0f}% of folds",
              flush=True)

        # ---------- CHECK 2: conformal validity of the new estimator ----------
        per, res, hz, tru = score(d, tools, m, SCHEMES[best_name], ps)
        qs = {}
        for bnum in (0, 1, 2):
            r = np.abs(res[[hbin(h) == bnum for h in hz]])
            qs[bnum] = float(np.quantile(r, 1 - ALPHA)) if len(r) else np.nan
        widths, covered = [], []
        for r, h in zip(res, hz):
            q = qs[hbin(h)]
            if np.isnan(q):
                continue
            widths.append(2 * q); covered.append(abs(r) <= q)
        cov = 100.0 * float(np.mean(covered))
        mw = float(np.mean(widths))
        print(f"  CONFORMAL  coverage {cov:.1f}%  (target 90, gate >=88)   mean width {mw:.1f} um",
              flush=True)

        valid = cov >= 88.0
        beats = nested < REC[m]
        rows.append(dict(m=m, best_scheme=best_name,
                         in_search_MAE=round(insearch[best_name], 2),
                         nested_MAE=round(nested, 2),
                         optimism=round(nested - insearch[best_name], 2),
                         nested_winner_share=round(float(share.iloc[0]), 2),
                         coverage_pct=round(cov, 1), mean_width_um=round(mw, 1),
                         record=REC[m], beats_record_nested=bool(beats),
                         coverage_valid=bool(valid),
                         ADOPTED=bool(beats and valid)))
        print(f"  >>> beats record (nested): {beats}   coverage valid: {valid}   "
              f"ADOPT: {beats and valid}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "p3_nested_audit.csv"), index=False)
    print("\n" + "=" * 74)
    print(out.to_string(index=False))
    print("\nwrote results/p3_nested_audit.csv")


if __name__ == "__main__":
    main()
