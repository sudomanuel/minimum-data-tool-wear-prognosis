"""run_record_attempts.py — record-beating suite. Current records (LOTO, leakage-safe):
  m=3 MAE 11.26 (Siegel, p full)   |   m=4 MAE 9.12 (Theil-Sen, p m-matched)
  Mondrian band 52.5 um @ 90.1%    |   online one-step 4.0 um
Five UNTESTED levers (everything previously rejected is excluded):
  A) horizon-binned signed BIAS CORRECTION of the point prediction (debias, LOTO-calibrated);
  B) ASYMMETRIC Mondrian band (two-sided signed quantiles per horizon bin) — exploits the measured
     under-prediction bias; symmetric |r| wastes width on the side where mass is thin;
  C) estimator ENSEMBLE (median of Theil-Sen / Siegel / OLS predictions);
  D) extrapolation-WEIGHTED least squares (weights ∝ tau: later early-points count more for the future);
  E) per-tool exponent with SHRINKAGE to the fleet p* (local grid ± penalty).
Pre-stated rule: adopt a lever ONLY if it beats the record at the same m (or tightens the band at
coverage >= 88%). Everything evaluated per-tool-averaged, identical folds.
Outputs: results/record_attempts.csv, results/record_band.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of
from run_optimal_config_search import fit_p, FIT
CENSOR = 300.0; AL = 0.10
RECORDS = {3: 11.26, 4: 9.12}
BAND_RECORD = (90.1, 52.5)


def siegel(x, y):
    return FIT["siegel"](x, y)


def ols(x, y):
    return FIT["ols"](x, y)


def wls_tau(x, y):
    """Weighted LS with weights ∝ x (=tau): later points dominate -> extrapolation-oriented."""
    w = np.maximum(x, 1e-9)
    W = np.sqrt(w)
    A = np.column_stack([W, W * x])
    c, *_ = np.linalg.lstsq(A, W * y, rcond=None)
    return float(c[1]), float(c[0])


def fewshot_pred(d, m, estimator, p_strategy="full", p_local=None):
    """Per-tool predictions. Returns dict tool -> (signed_resid, horizons, |resid|)."""
    out = {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_p(tr, m=(m if p_strategy == "m_matched" else None))
        if p_local is not None:                      # lever E: local p with shrinkage
            lam = p_local
            best_p, best = p, np.inf
            for pc in np.arange(max(p - 0.15, 0.05), p + 0.1501, 0.05):
                tau = o[:m] ** pc
                a, b = theil_sen(tau, v[:m])
                sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p) ** 2
                if sse < best:
                    best, best_p = sse, pc
            p = best_p
        tau = o[:m] ** p
        if estimator == "theil_sen":
            a, b = theil_sen(tau, v[:m])
        elif estimator == "siegel":
            a, b = siegel(tau, v[:m])
        elif estimator == "ols":
            a, b = ols(tau, v[:m])
        elif estimator == "wls_tau":
            a, b = wls_tau(tau, v[:m])
        elif estimator == "ensemble":
            preds = []
            for f in (theil_sen, siegel, ols):
                aa, bb = f(tau, v[:m]); preds.append(bb + aa * o[fut] ** p)
            pr = np.median(np.vstack(preds), axis=0)
            sr = pr - v[fut]
            out[tt] = (sr, (fut - (m - 1)).astype(float), np.abs(sr))
            continue
        pr = b + a * o[fut] ** p
        sr = pr - v[fut]
        out[tt] = (sr, (fut - (m - 1)).astype(float), np.abs(sr))
    return out


def mae_of(res):
    return float(np.mean([r[2].mean() for r in res.values()]))


def hbin(h):
    return 0 if h <= 1 else (1 if h <= 3 else 2)


def debias(res):
    """Lever A: per held-out tool, subtract the median SIGNED residual of the OTHER tools, per horizon
    bin (LOTO-honest). Returns corrected result dict."""
    tools = list(res)
    out = {}
    for tt in tools:
        cal_s = np.concatenate([res[t][0] for t in tools if t != tt])
        cal_h = np.concatenate([res[t][1] for t in tools if t != tt])
        sr, hh, _ = res[tt]
        corr = np.array([np.median(cal_s[np.array([hbin(x) for x in cal_h]) == hbin(h)])
                         if (np.array([hbin(x) for x in cal_h]) == hbin(h)).sum() >= 5
                         else np.median(cal_s) for h in hh])
        sr2 = sr - corr
        out[tt] = (sr2, hh, np.abs(sr2))
    return out


def cq(a, q):
    a = np.sort(np.asarray(a, float)); n = len(a)
    k = int(np.ceil((n + 1) * q))
    return float(a[min(max(k, 1), n) - 1])


def band_eval(res, asymmetric):
    """Mondrian band per horizon bin; symmetric (|r|, 1-α) vs asymmetric (signed, α/2 & 1-α/2).
    Per-tool coverage/width averaging (project convention)."""
    tools = list(res); cov, wid = [], []
    for tt in tools:
        cal_s = np.concatenate([res[t][0] for t in tools if t != tt])
        cal_h = np.array([hbin(x) for x in np.concatenate([res[t][1] for t in tools if t != tt])])
        sr, hh, _ = res[tt]
        tc, tw = [], []
        for s, h in zip(sr, hh):
            sel = cal_h == hbin(h)
            pool_s = cal_s[sel] if sel.sum() >= 8 else cal_s
            if asymmetric:
                lo, hi = cq(pool_s, AL / 2), cq(pool_s, 1 - AL / 2)   # signed residual quantiles
                tc.append(lo <= s <= hi); tw.append(hi - lo)
            else:
                q = cq(np.abs(pool_s), 1 - AL)
                tc.append(abs(s) <= q); tw.append(2 * q)
        cov.append(np.mean(tc)); wid.append(np.mean(tw))
    return float(np.mean(cov)) * 100, float(np.mean(wid))


def main():
    d = load()
    print("RECORD ATTEMPTS — pre-stated rule: adopt only what beats the record on identical folds.\n")
    print(f"records: m=3 {RECORDS[3]}  |  m=4 {RECORDS[4]}  |  band {BAND_RECORD[1]}um @ {BAND_RECORD[0]}%\n")
    rows = []

    # reference configs (the record holders)
    ref = {3: ("siegel", "full", None), 4: ("theil_sen", "m_matched", None)}
    res_ref = {}
    for m in (3, 4):
        est, ps, pl = ref[m]
        res_ref[m] = fewshot_pred(d, m, est, ps, pl)
        print(f"  [check] record config m={m}: MAE {mae_of(res_ref[m]):.2f} (record {RECORDS[m]})")

    print("\n--- levers on point MAE ---")
    for m in (3, 4):
        est, ps, _ = ref[m]
        candidates = {
            "A_debias(record cfg)": debias(res_ref[m]),
            "C_ensemble": fewshot_pred(d, m, "ensemble", ps),
            "D_wls_tau": fewshot_pred(d, m, "wls_tau", ps),
            "E_local_p(lam=200)": fewshot_pred(d, m, est, ps, p_local=200.0),
            "E_local_p(lam=50)": fewshot_pred(d, m, est, ps, p_local=50.0),
            "A+C_debias(ensemble)": debias(fewshot_pred(d, m, "ensemble", ps)),
        }
        for name, r in candidates.items():
            v = mae_of(r)
            beat = v < RECORDS[m] - 0.05
            rows.append(dict(m=m, lever=name, MAE=round(v, 2), record=RECORDS[m], beats=beat))
            print(f"  m={m} {name:24} MAE {v:6.2f}  {'** BEATS RECORD **' if beat else ''}")

    print("\n--- lever B: asymmetric Mondrian band (m=3, record config) ---")
    brows = []
    for tag, res in [("raw", res_ref[3]), ("debiased", debias(res_ref[3]))]:
        for asym in (False, True):
            c, w = band_eval(res, asym)
            name = f"{tag}+{'asym' if asym else 'sym'}"
            ok = c >= 88.0 and w < BAND_RECORD[1] - 0.5
            brows.append(dict(variant=name, PICP=round(c, 1), width=round(w, 1), beats=ok))
            print(f"  {name:16} PICP {c:5.1f}%  width {w:5.1f} um  {'** TIGHTER AT VALID COVERAGE **' if ok else ''}")

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "record_attempts.csv"), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(ROOT, "results", "record_band.csv"), index=False)

    winners = [r for r in rows if r["beats"]] + [b for b in brows if b["beats"]]
    print("\nVERDICT:", ("ADOPT: " + "; ".join(str(w.get("lever", w.get("variant"))) for w in winners))
          if winners else "no lever beats the records — the pipeline stands at its measured floor.")
    print("wrote results/record_attempts.csv, results/record_band.csv")


if __name__ == "__main__":
    main()
