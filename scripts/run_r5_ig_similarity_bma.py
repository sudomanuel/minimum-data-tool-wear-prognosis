"""run_r5_ig_similarity_bma.py — Round-5 A2/A5/A4 adjudication under one LOTO harness.

A2 INVERSE GAUSSIAN PROCESS with random effects (Ye & Chen 2014): monotone-by-construction
   stochastic process; drift eta random across the fleet, updated per tool from its m early
   increments (precision-weighted normal approximation); mean prediction b + eta_post * tau.
A5 SIMILARITY-BASED PROGNOSTICS (Wang, Yu, Siegel & Lee, PHM 2008): library of the 17 training
   curves; match the new tool's first m inspections (offset-aligned), inherit the continuation of
   the k nearest donors (k in {1,2,3}); donors shorter than the needed horizon are extended by
   their last observed slope.
A4 BAYESIAN MODEL AVERAGING over wear-law forms {power(fleet p), Archard p=0.5, linear}: fleet-BIC
   weights (pooled training SSE), per-tool few-shot fit of each form, weighted prediction.

Protocol: leakage-safe LOTO, m in {3,4}; references base 11.57/9.67, records 11.02/5.63.
Pre-stated rule: adopt only if a variant beats the record at the same m.
Outputs: results/r5_ig_similarity_bma.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, theil_sen, tools_of
from run_optimal_config_search import fit_p
CENSOR = 300.0
RECORDS = {3: 11.02, 4: 5.63}
BASE = {3: 11.57, 4: 9.67}


# ---------------- A2: IG process ----------------
def ig_pred(tr, o, v, m, p):
    tau = o ** p
    # fleet random-effect prior on the drift eta (rate in the linearized clock)
    etas = []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) >= 2:
            tt_, vv_ = gg.order.to_numpy(float), gg.vb.to_numpy(float)
            etas.append(max((vv_[-1] - vv_[0]) / max(tt_[-1] ** p - tt_[0] ** p, 1e-9), 1e-3))
    etas = np.array(etas)
    mu0, s0 = float(etas.mean()), max(float(etas.std(ddof=1)), 1e-6)
    # tool's increments over its m early inspections
    dY = np.diff(v[:m]); dT = np.diff(tau[:m])
    ok = dT > 1e-9
    dY, dT = np.clip(dY[ok], 0.0, None), dT[ok]
    if len(dY) == 0:
        return None
    eta_hat = float(np.sum(dY) / np.sum(dT))
    # observation variance of eta_hat: IG increment var = mu^3*dt/lambda ~ approx via fleet residual
    resid = dY - eta_hat * dT
    s_obs = max(float(np.std(resid, ddof=0)) / max(np.sum(dT), 1e-9) * np.sqrt(len(dY)), 1e-6)
    w = (1 / s_obs ** 2) / (1 / s_obs ** 2 + 1 / s0 ** 2)
    eta_post = w * eta_hat + (1 - w) * mu0
    b = float(v[0] - eta_post * tau[0])
    return lambda idx: b + eta_post * tau[idx]


# ---------------- A5: similarity ----------------
def donor_value_at(oc, vc, order):
    """VB of donor curve at a given cut order; linear inside, last-slope extension beyond."""
    if order <= oc[-1]:
        return float(np.interp(order, oc, vc))
    if len(oc) >= 2:
        sl = (vc[-1] - vc[-2]) / max(oc[-1] - oc[-2], 1e-9)
    else:
        sl = 0.0
    return float(vc[-1] + sl * (order - oc[-1]))


def similarity_pred(tr, o, v, m, k):
    dons = []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        oc, vc = gg.order.to_numpy(float), gg.vb.to_numpy(float)
        if len(oc) < 2:
            continue
        early = np.array([donor_value_at(oc, vc, x) for x in o[:m]])
        off = float(np.mean(v[:m] - early))
        dist = float(np.sqrt(np.mean((v[:m] - (early + off)) ** 2)))
        dons.append((dist, oc, vc, off))
    dons.sort(key=lambda d: d[0])
    sel = dons[:k]
    def f(idx):
        return np.array([np.mean([donor_value_at(oc, vc, o[j]) + off for _, oc, vc, off in sel])
                         for j in idx])
    return f


# ---------------- A4: BMA over wear-law forms ----------------
def bma_weights(tr, p_fleet):
    """Fleet BIC per model form (pooled SSE over training tools, per-tool 2-param fits)."""
    forms = {"power": p_fleet, "archard": 0.5, "linear": 1.0}
    out = {}
    for name, p in forms.items():
        sse, n = 0.0, 0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR].sort_values("order")
            if len(gg) < 3:
                continue
            oc, vc = gg.order.to_numpy(float), gg.vb.to_numpy(float)
            a, b = theil_sen(oc ** p, vc)
            sse += float(np.sum((b + a * oc ** p - vc) ** 2)); n += len(oc)
        sig2 = max(sse / max(n, 1), 1e-9)
        bic = n * np.log(sig2)  # + same k*log(n) for all -> cancels
        out[name] = (p, bic)
    bics = np.array([b for _, b in out.values()])
    w = np.exp(-(bics - bics.min()) / 2); w = w / w.sum()
    return {name: (out[name][0], float(wi)) for name, wi in zip(out, w)}


def bma_pred(tr, o, v, m, p_fleet):
    wts = bma_weights(tr, p_fleet)
    fits = {}
    for name, (p, w) in wts.items():
        a, b = theil_sen(o[:m] ** p, v[:m])
        fits[name] = (p, a, b, w)
    def f(idx):
        return np.sum([w * (b + a * o[idx] ** p) for p, a, b, w in fits.values()], axis=0)
    return f, {n: round(w, 3) for n, (_, _, _, w) in fits.items()}


def main():
    d = load()
    print("R5-A2/A5/A4: IG process | similarity library | BMA — adopt only if beats record.\n")
    rows = []
    for m in (3, 4):
        res = {"IG_process": [], "sim_k1": [], "sim_k2": [], "sim_k3": [], "BMA": []}
        wlog = []
        for tt in tools_of(d):
            tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
            o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
            if len(o) <= m:
                continue
            fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
            if len(fut) == 0:
                continue
            p_fleet = fit_p(tr)
            igf = ig_pred(tr, o, v, m, p_fleet)
            if igf is not None:
                res["IG_process"].append(np.abs(igf(fut) - v[fut]).mean())
            for k in (1, 2, 3):
                sf = similarity_pred(tr, o, v, m, k)
                res[f"sim_k{k}"].append(np.abs(sf(fut) - v[fut]).mean())
            bf, wts = bma_pred(tr, o, v, m, p_fleet)
            res["BMA"].append(np.abs(bf(fut) - v[fut]).mean())
            wlog.append(wts)
        print(f"--- m={m} (base {BASE[m]}, record {RECORDS[m]}) ---")
        for name, vals in res.items():
            mae = float(np.mean(vals))
            beat = "** BEATS **" if mae < RECORDS[m] - 0.05 else ""
            rows.append(dict(m=m, method=name, MAE=round(mae, 2)))
            print(f"  {name:11}: {mae:6.2f} {beat}")
        wmean = pd.DataFrame(wlog).mean().round(3).to_dict()
        print(f"  BMA mean fleet weights: {wmean}\n")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r5_ig_similarity_bma.csv"), index=False)
    winners = df[df.apply(lambda r: r.MAE < RECORDS[r.m] - 0.05, axis=1)]
    print("WINNERS:")
    print(winners.to_string(index=False) if len(winners) else "  none — records stand.")
    print("wrote results/r5_ig_similarity_bma.csv")


if __name__ == "__main__":
    main()
