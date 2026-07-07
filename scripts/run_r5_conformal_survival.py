"""run_r5_conformal_survival.py — Round-5 A1: conformalized survival for the safe stop.

Idea (Candès, Lei & Ren, JRSS-B 2023, adapted to our fully-observed terminal events): instead of a
symmetric RUL window, produce a CALIBRATED LOWER BOUND on the time-to-chipping — the exact quantity a
safe-stop decision needs, with a finite-sample guarantee P(T_true >= t_lb) >= 1-alpha.

Construction (LOTO, project convention):
  - Every tool's record ends in an observed chipping at t_fail (18 real events; no active censoring
    among terminals, so the general censored machinery reduces to one-sided conformal).
  - Point predictor: the deployed m=3 few-shot law's crossing of VB_safe = 167 um.
  - Signed nonconformity: s_i = t_hat_i - t_fail_i  (positive = OVER-estimation = dangerous side).
  - For held-out tool u:  t_lb(u) = t_hat_u - q_{1-alpha}({s_i : i != u}) with the finite-sample
    quantile index ceil((n+1)(1-alpha))/n.
Scoring: empirical coverage of t_lb (target >= 90%), median slack (t_fail - t_lb) in cuts, and the
naive (uncalibrated) coverage of t_hat for contrast. Pre-stated rule: report as comes; adopt into the
manuscript only if coverage holds and the bound is decision-useful (slack finite and interpretable).
Outputs: results/r5_conformal_survival.csv
"""
import os, sys, math
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, theil_sen, tools_of
from run_optimal_config_search import fit_p, FIT
CENSOR = 300.0; M = 3; ALPHA = 0.10; VB_SAFE = 167.0


def local_p_grid(o, v, m, p_star, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def t_hat_crossing(d, tt):
    """Deployed m=3 config: predicted crossing time of VB_safe for tool tt (LOTO)."""
    tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
    o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
    if len(o) <= M:
        return None, None
    p = local_p_grid(o, v, M, fit_p(tr))
    a, b = FIT["siegel"](o[:M] ** p, v[:M])
    if a <= 0 or VB_SAFE <= b:
        return None, o[-1]
    return float(((VB_SAFE - b) / a) ** (1.0 / p)), o[-1]


def main():
    d = load()
    tools = tools_of(d)
    print("R5-A1: CONFORMALIZED SURVIVAL — calibrated lower bound on time-to-chipping "
          f"(alpha={ALPHA}, VB_safe={VB_SAFE:.0f} um, deployed m={M} predictor)\n")
    pred = {}
    for tt in tools:
        th, tf = t_hat_crossing(d, tt)
        if th is not None:
            pred[tt] = (th, tf)
    names = sorted(pred)
    n_all = len(names)
    print(f"usable tools (defined crossing prediction): {n_all}/18\n")

    rows, cover, slack, naive_cover = [], [], [], []
    for u in names:
        cal = [pred[c][0] - pred[c][1] for c in names if c != u]   # signed scores t_hat - t_fail
        ncal = len(cal)
        k = math.ceil((ncal + 1) * (1 - ALPHA))
        if k > ncal:
            q = float(np.inf)
        else:
            q = float(np.sort(cal)[k - 1])
        th, tf = pred[u]
        tlb = th - q
        cov = tlb <= tf
        cover.append(cov); slack.append(tf - tlb); naive_cover.append(th <= tf)
        rows.append(dict(tool=u, t_fail=tf, t_hat=round(th, 2), q90=round(q, 2),
                         t_lb=round(tlb, 2), covered=int(cov), slack_cuts=round(tf - tlb, 2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r5_conformal_survival.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\ncoverage of calibrated lower bound: {100*np.mean(cover):.0f}%  (target >= {100*(1-ALPHA):.0f}%)")
    print(f"naive coverage of raw t_hat (P(t_hat <= t_fail)): {100*np.mean(naive_cover):.0f}%  "
          f"-> the raw crossing OVER-estimates life; calibration is what makes it safe")
    print(f"slack (t_fail - t_lb): median {np.median(slack):.1f} cuts | min {np.min(slack):.1f} | "
          f"max {np.max(slack):.1f}")
    print("\nREADING: t_lb is a finite-sample-guaranteed 'do not run past this' time computed from "
          "three early inspections — the decision object the safe stop needs.")
    print("wrote results/r5_conformal_survival.csv")


if __name__ == "__main__":
    main()
