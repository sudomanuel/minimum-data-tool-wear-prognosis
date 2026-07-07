"""run_r7_bayes_particle.py — Round-7 directive A adjudication: replace the deterministic few-shot
fit with full Bayesian updating (hierarchical fleet priors + particle/importance filtering).

Implementation is exactly what the directive asks for, in its strongest cheap form:
  - Hierarchical fleet priors from the 17 training tools (LOTO): log a ~ N, b ~ N, p ~ N truncated
    to (0.05, 0.60) — per-tool shape uncertainty INCLUDED (more freedom than the deployed local grid).
  - Monotonicity by construction: every particle has a = exp(la) > 0, p in (0,1) -> monotone curve.
  - 20,000 particles; Gaussian measurement likelihood on the m early points (sigma_meas from the
    fleet's own full-trajectory fit residuals, plus a declared grid {4, 6, 8} um); with static m
    points a particle filter reduces to importance sampling from the prior — this IS the posterior
    predictive, i.e. the exact object the directive requests. Divergence impossible (weights
    normalized; ESS reported).
  - Prediction: posterior-predictive mean at the sealed future cuts. LOTO, m in {3,4}.
Pre-stated rule: adopt only if it beats the record at the same m (11.02 / 5.63).
Context already on record: the linear-Gaussian sequential limit of this idea (Kalman as forecaster)
was tested and rejected (12.2 um, 621 um bands); EB fleet shrinkage = power-prior posterior mode
(round 5). Outputs: results/r7_bayes_particle.csv
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
N_PART = 20000
RNG = np.random.default_rng(7)


def fleet_prior(tr, p_star):
    """Per-tool (log a, b) at candidate p plus residual scale, over training tools."""
    las, bs, res = [], [], []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) < 3:
            continue
        o, v = gg.order.to_numpy(float), gg.vb.to_numpy(float)
        a, b = theil_sen(o ** p_star, v)
        las.append(np.log(max(a, 1e-3))); bs.append(b)
        res += list(v - (b + a * o ** p_star))
    return (float(np.mean(las)), max(float(np.std(las, ddof=1)), 1e-3),
            float(np.mean(bs)), max(float(np.std(bs, ddof=1)), 1e-3),
            float(np.std(res, ddof=1)))


def evaluate(d, m, sig_mode):
    per, ess_all = [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p_star = fit_p(tr, m=(m if m == 4 else None))
        mu_la, s_la, mu_b, s_b, sig_fleet = fleet_prior(tr, p_star)
        sig = sig_fleet if sig_mode == "fleet" else float(sig_mode)
        # particles
        la = RNG.normal(mu_la, s_la, N_PART)
        b = RNG.normal(mu_b, s_b, N_PART)
        p = np.clip(RNG.normal(p_star, 0.05, N_PART), 0.05, 0.60)
        a = np.exp(la)
        # importance weights from the m early points
        pred_early = b[:, None] + a[:, None] * (o[:m][None, :] ** p[:, None])
        ll = -0.5 * np.sum(((pred_early - v[:m][None, :]) / sig) ** 2, axis=1)
        w = np.exp(ll - ll.max()); w = w / w.sum()
        ess_all.append(1.0 / np.sum(w ** 2))
        # posterior-predictive mean on future cuts
        pred_fut = b[:, None] + a[:, None] * (o[fut][None, :] ** p[:, None])
        mean_fut = w @ pred_fut
        per.append(np.abs(mean_fut - v[fut]).mean())
    return float(np.mean(per)), float(np.median(ess_all))


def main():
    d = load()
    print("R7-A: BAYESIAN POSTERIOR PREDICTIVE / PARTICLE FEW-SHOT — adopt only if beats record.\n"
          f"priors: hierarchical fleet (log a, b) + truncated-normal shape p; {N_PART} particles; "
          "monotone by construction.\n")
    rows = []
    for m in (3, 4):
        print(f"--- m={m} (base {BASE[m]}, record {RECORDS[m]}) ---")
        for sig_mode in ("fleet", 4.0, 6.0, 8.0):
            mae, ess = evaluate(d, m, sig_mode)
            tag = f"sigma={sig_mode}" if sig_mode != "fleet" else "sigma=fleet-resid"
            beat = "** BEATS **" if mae < RECORDS[m] - 0.05 else ""
            rows.append(dict(m=m, sigma=str(sig_mode), MAE=round(mae, 2), median_ESS=round(ess, 0)))
            print(f"  {tag:18}: MAE {mae:6.2f}  (median ESS {ess:6.0f}) {beat}")
        print()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r7_bayes_particle.csv"), index=False)
    win = df[df.apply(lambda r: r.MAE < RECORDS[int(r.m)] - 0.05, axis=1)]
    print("WINNERS:" if len(win) else "VERDICT: the full Bayesian posterior predictive does not beat "
          "the deployed estimator — the fleet prior is the same information the EB/power-prior layer "
          "already uses, and averaging over shape uncertainty does not outperform the "
          "extrapolation-weighted point fit at these budgets.")
    if len(win):
        print(win.to_string(index=False))
    print("wrote results/r7_bayes_particle.csv")


if __name__ == "__main__":
    main()
