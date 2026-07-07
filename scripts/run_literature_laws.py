"""run_literature_laws.py — test the CANONICAL literature wear equations under our LOTO protocol.

Q: do the paper-specific physics equations beat our own law? Pre-stated rule: adopt only if a form
beats the record at the same m; otherwise they are cited for comparison in the manuscript.

Forms tested (all leakage-safe, identical folds):
  ARCHARD   : VB = b + a*t^0.5  (Archard's constant-pressure sliding wear => p = 1/2, fixed)
  TAYLOR    : extended Taylor life V*T^n*f^a = C  =>  log(rate) = c0 + c1*log(vc) + c2*log(fz) + c3*cool
              fitted on the fleet's full-trajectory rates; new tool's rate from ITS CONDITION
              (zero-shot rate; level b anchored on the m early points), plus a few-shot blend grid.
  USUI-proxy: thermally-activated rate dW/dt ∝ exp(theta*T); temperature proxied by the tool's mean
              rotational vibration energy (standardized on the fleet):  log a = d0 + d1*E_rot.
              Zero-shot rate from energy + blend grid. (True Usui needs force/temperature — not
              instrumented in this campaign; the proxy is the closest testable surrogate.)
Reference points: base few-shot m=3 11.57 / m=4 9.67; records m=3 11.02 / m=4 5.63.
Outputs: results/literature_laws.csv
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
CENSOR = 300.0
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")
RECORDS = {3: 11.02, 4: 5.63}
BASE = {3: 11.57, 4: 9.67}


def tool_conditions():
    f = pd.read_csv(FEAT)
    g = f.groupby("tool_id").first()
    cool = (g["cooling"].astype(str).str.lower()
            .map(lambda s: 0.0 if ("dry" in s or "no" in s or s in ("0", "nan")) else 1.0))
    erot = None
    for c in ["R_energy__mean", "R_rms__mean"]:
        if c in g.columns:
            erot = g[c].astype(float); break
    return pd.DataFrame(dict(vc=g.vc.astype(float), fz=g.fz.astype(float), cool=cool,
                             erot=erot)).fillna(0.0)


def fleet_rates(tr, p):
    """Robust full-trajectory rate per training tool at exponent p."""
    out = {}
    for tt, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) < 2:
            continue
        a, _ = theil_sen(gg.order.to_numpy(float) ** p, gg.vb.to_numpy(float))
        out[tt] = max(a, 1e-3)
    return out


def rate_model(rates, cond, kind):
    """Fit log(rate) on condition (TAYLOR) or on rotational energy (USUI proxy). Return predict(tool)."""
    ids = [t for t in rates if t in cond.index]
    y = np.log([rates[t] for t in ids])
    if kind == "taylor":
        X = np.column_stack([np.ones(len(ids)),
                             np.log(cond.loc[ids, "vc"]), np.log(cond.loc[ids, "fz"]),
                             cond.loc[ids, "cool"]])
        def feats(t):
            return np.array([1.0, np.log(cond.loc[t, "vc"]), np.log(cond.loc[t, "fz"]),
                             cond.loc[t, "cool"]])
    else:  # usui proxy
        e = cond.loc[ids, "erot"].to_numpy(float)
        mu, sd = e.mean(), max(e.std(), 1e-9)
        X = np.column_stack([np.ones(len(ids)), (e - mu) / sd])
        def feats(t):
            return np.array([1.0, (cond.loc[t, "erot"] - mu) / sd])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return lambda t: float(np.exp(feats(t) @ c))


def evaluate(d, cond, m, form, lam=0.0):
    per = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        if form == "archard":
            p = 0.5
            tau = o[:m] ** p
            a, b = theil_sen(tau, v[:m])
        else:
            p = fit_global_p(tr)
            tau = o[:m] ** p
            a_fs, _ = theil_sen(tau, v[:m])
            pred_rate = rate_model(fleet_rates(tr, p), cond,
                                   "taylor" if form.startswith("taylor") else "usui")(tt)
            a = (1 - lam) * a_fs + lam * pred_rate
            b = float(np.median(v[:m] - a * tau))
        if form == "archard":
            pred = b + a * o[fut] ** p
        else:
            pred = b + a * o[fut] ** p
        per.append(np.abs(pred - v[fut]).mean())
    return float(np.mean(per))


def main():
    d = load(); cond = tool_conditions()
    print("LITERATURE WEAR EQUATIONS under LOTO — adopt only if a form beats the record at the same m.\n")
    print(f"reference: base m=3 {BASE[3]} / m=4 {BASE[4]}  |  records m=3 {RECORDS[3]} / m=4 {RECORDS[4]}\n")
    rows = []
    for m in (3, 4):
        v = evaluate(d, cond, m, "archard")
        rows.append(dict(m=m, form="ARCHARD p=0.5 (fixed)", lam="-", MAE=round(v, 2)))
        print(f"  m={m} ARCHARD p=0.5           : {v:6.2f}")
        for lam in (1.0, 0.5, 0.25):
            v = evaluate(d, cond, m, "taylor", lam)
            tag = "zero-shot (condition only)" if lam == 1.0 else f"blend λ={lam}"
            rows.append(dict(m=m, form=f"TAYLOR rate {tag}", lam=lam, MAE=round(v, 2)))
            print(f"  m={m} TAYLOR {tag:26}: {v:6.2f}")
        for lam in (1.0, 0.5, 0.25):
            v = evaluate(d, cond, m, "usui", lam)
            tag = "zero-shot (energy only)" if lam == 1.0 else f"blend λ={lam}"
            rows.append(dict(m=m, form=f"USUI-proxy rate {tag}", lam=lam, MAE=round(v, 2)))
            print(f"  m={m} USUI-proxy {tag:22}: {v:6.2f}")
        print()
    df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results", "literature_laws.csv"), index=False)
    winners = [(r["m"], r["form"], r["MAE"]) for _, r in df.iterrows()
               if r["MAE"] < RECORDS[r["m"]] - 0.05]
    if winners:
        print("WINNERS (beat the record):", winners)
    else:
        best = df.loc[df.groupby("m").MAE.idxmin()]
        print("VERDICT: no canonical literature form beats the record — cite for comparison.")
        for _, r in best.iterrows():
            print(f"  best literature form at m={r['m']}: {r['form']} = {r['MAE']} "
                  f"(record {RECORDS[r['m']]}, base {BASE[r['m']]})")
    print("\nwrote results/literature_laws.csv")


if __name__ == "__main__":
    main()
