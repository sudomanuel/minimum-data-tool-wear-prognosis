"""run_e_phm2010_external.py — Reviewer-response E: external sanity check of the few-shot machinery
on the public PHM2010 milling campaign (appendix material).

Scope declared BEFORE running (report-as-comes rule):
  - Data: per-cut flank wear (max over 3 flutes, um) of cutters c1/c4/c6, 315 cuts each
    (data/external/phm2010/, provenance in its README). No signals used.
  - The method is designed for sparse ex-situ inspection, so the dense record is subsampled to an
    inspection cadence: PRIMARY K=10 equally spaced inspections per tool; SENSITIVITY K=5.
  - LOTO across the 3 cutters (train on 2, hold out 1); few-shot budgets m=3 and m=4 on the held-out
    cutter's first inspections; every later inspection is sealed and used only for scoring.
  - Configurations: the paper's deployed ones — m=3 Siegel + locally-shrunk exponent; m=4
    extrapolation-weighted WLS (gamma=3) + locally-shrunk exponent, m-matched p. Baselines: the
    average-wear-curve of the training cutters (few-shot offset) and a linear-in-t robust fit.
  - Nothing from this dataset touches the 18-tool campaign's training, tuning, or calibration.
Outputs: results/e_phm2010_external.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
DATA = os.path.join(ROOT, "data", "external", "phm2010", "phm2010_wear_percut.csv")
P_GRID = np.arange(0.05, 0.9001, 0.01)


def theil_sen(x, y):
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))
          if x[j] != x[i]]
    a = float(np.median(sl))
    return a, float(np.median(y - a * x))


def siegel(x, y):
    n = len(x)
    med = []
    for i in range(n):
        s = [(y[j] - y[i]) / (x[j] - x[i]) for j in range(n) if j != i and x[j] != x[i]]
        med.append(np.median(s))
    a = float(np.median(med))
    return a, float(np.median(y - a * x))


def wls_gamma(tau, y, gamma=3.0):
    w = np.maximum(tau, 1e-9) ** gamma
    W = np.sqrt(w)
    A = np.column_stack([W, W * tau])
    c, *_ = np.linalg.lstsq(A, W * y, rcond=None)
    return float(c[1]), float(c[0])


def fit_global_p(curves, m=None):
    """Pooled-SSE exponent over training curves ((o,v) pairs); optionally m-matched."""
    best_p, best = 0.2, np.inf
    for p in P_GRID:
        sse = 0.0
        for o, v in curves:
            oo, vv = (o[:m], v[:m]) if m else (o, v)
            tau = oo ** p
            a, b = theil_sen(tau, vv)
            sse += float(np.sum((b + a * tau - vv) ** 2))
        if sse < best:
            best, best_p = sse, p
    return best_p


def local_p_grid(o, v, m, p_star, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def subsample(df, K):
    """K equally spaced inspections per cutter (always including the last cut)."""
    out = {}
    for c, g in df.groupby("cutter"):
        g = g.sort_values("cut")
        idx = np.unique(np.round(np.linspace(1, len(g), K)).astype(int)) - 1
        out[c] = (g.cut.to_numpy(float)[idx], g.max_wear.to_numpy(float)[idx])
    return out


def evaluate(curves, m, model):
    """LOTO over the 3 cutters; returns per-cutter future-inspection MAE + pooled R2."""
    names = sorted(curves)
    per, P, Y = {}, [], []
    for held in names:
        tr = [curves[c] for c in names if c != held]
        o, v = curves[held]
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o))
        if model == "physics":
            if m == 3:
                p = local_p_grid(o, v, m, fit_global_p(tr))
                a, b = siegel(o[:m] ** p, v[:m])
            else:
                p = local_p_grid(o, v, m, fit_global_p(tr, m=m))
                a, b = wls_gamma(o[:m] ** p, v[:m], 3.0)
            pred = b + a * o[fut] ** p
        elif model == "avgcurve":
            L = min(len(tr[0][1]), len(tr[1][1]))
            pop = np.mean([t[1][:L] for t in tr], axis=0)
            off = float(np.mean(v[:m] - pop[:m]))
            pred = pop[fut] + off
        else:  # linear in t
            a, b = theil_sen(o[:m], v[:m])
            pred = b + a * o[fut]
        per[held] = float(np.mean(np.abs(pred - v[fut])))
        P += list(pred); Y += list(v[fut])
    P, Y = np.array(P), np.array(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return per, float(np.mean(list(per.values()))), float(r2)


def main():
    df = pd.read_csv(DATA)
    print("E: EXTERNAL SANITY CHECK — PHM2010 (3 labeled cutters, 315 cuts each, wear = max flute, um)")
    print("protocol: LOTO over cutters; inspection cadence K=10 primary / K=5 sensitivity; "
          "deployed configs; report-as-comes.\n")
    rows = []
    for K in (10, 5):
        curves = subsample(df, K)
        print(f"--- cadence K={K} inspections/tool ---")
        for m in (3, 4):
            if K == 5 and m == 4:
                pass  # still valid: predict the single remaining inspection
            res = {}
            for model in ("physics", "avgcurve", "linear"):
                per, mae, r2 = evaluate(curves, m, model)
                res[model] = (per, mae, r2)
                rows.append(dict(K=K, m=m, model=model, MAE=round(mae, 2), R2_pooled=round(r2, 2),
                                 **{f"MAE_{c}": round(v, 2) for c, v in per.items()}))
            ph, av, ln = res["physics"], res["avgcurve"], res["linear"]
            beat = "BEATS baseline" if ph[1] < av[1] else "does NOT beat baseline"
            print(f"  m={m}: physics {ph[1]:6.2f} um (R2 {ph[2]:+.2f}) | avg-curve {av[1]:6.2f} "
                  f"(R2 {av[2]:+.2f}) | linear {ln[1]:6.2f} (R2 {ln[2]:+.2f})  -> {beat}")
        print()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "e_phm2010_external.csv"), index=False)
    print("wrote results/e_phm2010_external.csv")


if __name__ == "__main__":
    main()
