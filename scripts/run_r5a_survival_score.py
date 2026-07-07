"""run_r5a_survival_score.py — R5a: right-censored survival log-likelihood over the FULL fleet.

Every tool contributes evidence: observed 200um crossings via log f(t_true), censored tools via
log(1 - F(t_cens)) — models that predict failure before the censoring time are penalised. The
model-implied failure-time distribution T ~ Normal(t_hat, sigma_T) comes from the calibrated Mondrian
window mapped through the crossing (window = central 90% interval -> sigma_T = halfwidth/1.645).

Models compared (identical LOTO folds, m=3 few-shot):
  ours_k0      : record config (Siegel + local-p), concave law (kappa=0)
  ours_k002    : same, two-phase continuation kappa=0.02 (envelope minimum variant)
  avg_curve    : average-wear-curve population baseline + few-shot offset
  linear_t     : linear-in-t self fit
Report mean per-tool censored log-likelihood (higher better) + counts. Outputs: results/r5a_survival.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of
from run_optimal_config_search import FIT
from run_f1_multithreshold_rul import fleet_residuals_by_horizon, mondrian_q, true_crossing
from run_r1_twophase_law import two_phase, crossing as tp_crossing
from phm.prognostic_system import fit_population, fewshot_offset, predict_vb
CENSOR = 300.0; VB_FAIL = 200.0; M = 3; Z90 = 1.6449


def local_p(o, v, m, p_star, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def crossing_grid(fn, t_max=300.0):
    ts = np.linspace(0.5, t_max, 6000)
    vb = fn(ts)
    idx = np.argmax(vb >= VB_FAIL)
    return float(ts[idx]) if vb[idx] >= VB_FAIL else np.nan


def main():
    d = load(); R = fleet_residuals_by_horizon(d)
    rows_detail = []
    scores = {k: [] for k in ("ours_k0", "ours_k002", "avg_curve", "linear_t")}
    n_obs = n_cens = 0
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M or v[0] >= VB_FAIL:              # born-failed: left-censored, excluded (noted)
            continue
        p_star = fit_global_p(tr)
        p = local_p(o, v, M, p_star)
        tau = o[:M] ** p
        a, b = FIT["siegel"](tau, v[:M])
        t_ref = o[M - 1]
        # calibrated sigma_T from the Mondrian window at the predicted crossing
        cal_r = np.concatenate([R[t][0] for t in R if t != tt])
        cal_h = np.concatenate([R[t][1] for t in R if t != tt])

        def sigma_T(t_hat, slope):
            hh = max(t_hat - t_ref, 1.0)
            q = mondrian_q(cal_r, cal_h, hh)
            return max(q / max(slope, 1e-6), 0.5) / Z90

        preds = {}
        # ours kappa=0 / kappa=0.02
        for name, kap in (("ours_k0", 0.0), ("ours_k002", 0.02)):
            th = tp_crossing(b, a, p, 150.0, kap, VB_FAIL, t_max=300.0)
            if not np.isnan(th):
                slope = a * p * max(th, 1.0) ** (p - 1)
                preds[name] = (th, sigma_T(th, slope))
        # avg-curve baseline
        pop = fit_population(tr); off = fewshot_offset(pop, o, v, M)
        th = crossing_grid(lambda ts: predict_vb(pop, ts, off))
        if not np.isnan(th):
            preds["avg_curve"] = (th, sigma_T(th, max((predict_vb(pop, np.array([th + 1]), off)[0]
                                                       - predict_vb(pop, np.array([th - 1]), off)[0]) / 2, 1e-6)))
        # linear in t
        al, bl = theil_sen(o[:M], v[:M])
        if al > 0:
            th = (VB_FAIL - bl) / al
            if th > 0:
                preds["linear_t"] = (th, sigma_T(th, al))

        tc = true_crossing(o, v, VB_FAIL)
        observed = tc is not None and tc > t_ref
        if observed:
            n_obs += 1
        else:
            n_cens += 1
        t_cens = o[-1]
        for name in scores:
            if name not in preds:
                # model predicts no crossing within horizon: survival prob ~1 -> ll=0 if censored,
                # heavily penalised if a real crossing was observed
                ll = 0.0 if not observed else np.log(1e-6)
            else:
                mu, sd = preds[name]
                if observed:
                    ll = float(norm.logpdf(tc, mu, sd))
                else:
                    ll = float(np.log(max(1.0 - norm.cdf(t_cens, mu, sd), 1e-12)))
            scores[name].append(ll)
            rows_detail.append(dict(tool=tt, model=name, observed=int(observed), ll=round(ll, 3)))

    print(f"R5a censored survival log-likelihood (VB_fail=200): {n_obs} observed crossings, "
          f"{n_cens} right-censored tools contribute via survival terms.\n")
    print(f"{'model':12} {'mean ll/tool':>13}  (higher is better)")
    summary = []
    for name, ll in scores.items():
        print(f"{name:12} {np.mean(ll):13.2f}")
        summary.append(dict(model=name, mean_ll=round(float(np.mean(ll)), 3), n=len(ll)))
    pd.DataFrame(summary).to_csv(os.path.join(ROOT, "results", "r5a_survival.csv"), index=False)
    pd.DataFrame(rows_detail).to_csv(os.path.join(ROOT, "results", "r5a_survival_detail.csv"), index=False)
    best = max(summary, key=lambda r: r["mean_ll"])
    print(f"\nBest under the censoring-correct rule: {best['model']} ({best['mean_ll']}).")
    print("wrote results/r5a_survival.csv, results/r5a_survival_detail.csv")


if __name__ == "__main__":
    main()
