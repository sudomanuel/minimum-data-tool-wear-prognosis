"""run_online_accuracy.py — P8: try to lower the online one-step-ahead MAE below the constant-velocity
Kalman baseline using a PHYSICS-constrained recursive predictor (recursive Theil-Sen on the linearised
power law τ = order^p), warm-started from the training population slope. Optionally a robust blend.
Pre-stated rule: adopt only if it beats the Kalman one-step MAE. Leakage-safe LOTO, VB<=300."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_online_monitor import load, fit_global_p, theil_sen, tr_params, kf_online_onestep
CENSOR = 300.0


def physics_online(tr, o, v, p, warm=True):
    """One-step-ahead: after observing points 0..k-1, fit the power law and predict point k."""
    _, pod, _ = tr_params(tr, p)
    tau = o ** p; preds = []
    for k in range(1, len(o)):
        if k >= 2:
            a, b = theil_sen(tau[:k], v[:k])
        else:
            a = pod if warm else (v[1] - v[0]) / max(tau[1] - tau[0], 1e-9)
            b = v[0] - a * tau[0]
        pr = b + a * tau[k]
        if v[k] <= CENSOR:
            preds.append((pr, float(v[k])))
    return preds


def mae(preds):
    return float(np.mean([abs(pr - tu) for pr, tu in preds])) if preds else np.nan


def main():
    d = load()
    tools = sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))
    kf, phys, phys_cold = [], [], []
    for tt in tools:
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= 2:
            continue
        p = fit_global_p(tr)
        kf += [(pr, tu) for pr, tu, _ in kf_online_onestep(tr, o, v, p)]
        phys += physics_online(tr, o, v, p, warm=True)
        phys_cold += physics_online(tr, o, v, p, warm=False)
    res = dict(kalman_cv=round(mae(kf), 2), physics_warm=round(mae(phys), 2),
               physics_cold=round(mae(phys_cold), 2))
    print("Online one-step-ahead MAE (µm), pooled over all next-cut predictions. Lower is better.\n")
    for k, val in res.items():
        print(f"  {k:16s} {val:6.2f}")
    pd.DataFrame([res]).to_csv(os.path.join(ROOT, "results", "online_accuracy.csv"), index=False)
    best = min(res, key=res.get)
    print("\nVerdict (pre-stated rule: adopt only if it beats the Kalman baseline):")
    if best != "kalman_cv" and res[best] < res["kalman_cv"] - 0.05:
        print(f"  {best} improves {res['kalman_cv']} → {res[best]} µm — adopt as the online predictor.")
    else:
        print(f"  Kalman CV remains best ({res['kalman_cv']} µm) — keep it; physics-recursive ties/does not beat it.")
    print("\nwrote results/online_accuracy.csv")


if __name__ == "__main__":
    main()
