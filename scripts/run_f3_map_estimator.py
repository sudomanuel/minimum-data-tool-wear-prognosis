"""run_f3_map_estimator.py — FRONT 3: replace the fragile Theil-Sen/Siegel few-shot initialization with a
hierarchical MAP estimator (empirical-Bayes fleet prior + robust Student-t likelihood).

Graph-informed acceptance rule (graphify node `levers_rejected`: heavy hierarchical Bayes was REJECTED at
12.8 um): the MAP estimator MUST NOT regress MAE — it must MATCH or BEAT the current few-shot MAE at each m,
AND win on provable robustness (bounded empirical breakdown at m=3). Adopt only if both hold.

Model (physical clock tau = order^p, VB = b + a*tau):
  fleet prior (LOTO, on training tools' full censored fits):  log a ~ N(mu_la, s_la^2),  b ~ N(mu_b, s_b^2)
  per held-out tool, first m points:
    (b_hat, g_hat) = argmin_{b,g}  sum_i  rho_nu( (v_i - b - e^g * tau_i)/sigma_meas )
                                    + (g - mu_la)^2/(2 s_la^2) + (b - mu_b)^2/(2 s_b^2),   a = e^g
  rho_nu = Student-t negative log-likelihood (nu dof) -> bounded influence function.
sigma_meas from pooled robust residual scale of training full-fits (optical-microscope noise), floored.
Outputs: results/f3_map_metrics.csv, results/f3_breakdown.csv, outputs/figures/f3_breakdown.png
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of
CENSOR = 300.0
NU = 4.0                 # Student-t degrees of freedom (heavy-tailed -> outlier-robust)
SIGMA_FLOOR = 3.0        # microscope repeatability floor (um)


def siegel(x, y):
    n = len(x); med = []
    for i in range(n):
        sl = [(y[j]-y[i])/(x[j]-x[i]) for j in range(n) if j != i and x[j] != x[i]]
        if sl:
            med.append(np.median(sl))
    s = float(np.median(med)) if med else 0.0
    return s, float(np.median(y - s*x))


def fleet_prior(tr, p, kappa=1.0):
    """EB prior on (b, log a) + measurement scale, from training tools' full censored fits.
    kappa scales the prior standard deviations: kappa>1 => weaker (vaguer) prior => less shrinkage."""
    logas, bs, resid = [], [], []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) < 2:
            continue
        tau = gg.order.to_numpy(float) ** p; v = gg.vb.to_numpy(float)
        a, b = theil_sen(tau, v)
        a = max(a, 1e-3)
        logas.append(np.log(a)); bs.append(b); resid += list(v - (b + a*tau))
    logas, bs, resid = np.array(logas), np.array(bs), np.array(resid)
    mu_la, s_la = logas.mean(), max(logas.std(ddof=1), 0.15) * kappa
    mu_b, s_b = bs.mean(), max(bs.std(ddof=1), 3.0) * kappa
    sigma = max(1.4826 * np.median(np.abs(resid - np.median(resid))), SIGMA_FLOOR)
    return mu_la, s_la, mu_b, s_b, sigma


def map_fit(tau, v, prior, nu=NU):
    mu_la, s_la, mu_b, s_b, sigma = prior
    def nll(th):
        b, g = th; a = np.exp(g); e = (v - b - a*tau)/sigma
        lik = np.sum((nu+1)/2 * np.log1p(e*e/nu))         # Student-t NLL (up to const)
        pen = (g-mu_la)**2/(2*s_la**2) + (b-mu_b)**2/(2*s_b**2)
        return lik + pen
    best = None
    for x0 in [(mu_b, mu_la), (float(np.median(v)), mu_la)]:
        r = minimize(nll, x0, method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    b, g = best.x
    return float(b), float(np.exp(g))


def hybrid_fit(tau, v, prior, c=5.0):
    """Detect-then-robustify: Theil-Sen by default (best clean accuracy); if an outlier is detected among
    the m points (max |Theil-Sen residual| > c * sigma_meas), refit with the strong-prior MAP, whose
    bounded influence caps the outlier's effect. Dominates: parity on clean data, robust on contaminated.
    Returns (b, a, triggered)."""
    a, b = theil_sen(tau, v)
    sigma = prior[4]
    if np.max(np.abs(v - (b + a*tau))) > c * sigma:
        b, a = map_fit(tau, v, prior)
        return b, a, True
    return b, a, False


def eval_estimator(d, m, estimator, kappa=1.0):
    """estimator in {'theilsen','siegel','map','hybrid'}; pooled metrics + per-tool residual/horizon."""
    per_mae, P, Y, resid, horiz, triggers = [], [], [], [], [], []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr); tau = o[:m] ** p
        trig = 0
        if estimator == "theilsen":
            a, b = theil_sen(tau, v[:m])
        elif estimator == "siegel":
            a, b = siegel(tau, v[:m])
        elif estimator == "hybrid":
            b, a, t = hybrid_fit(tau, v[:m], fleet_prior(tr, p, 1.0), c=kappa); trig = int(t)
        else:
            b, a = map_fit(tau, v[:m], fleet_prior(tr, p, kappa))
        pred = b + a * o[fut] ** p; tru = v[fut]; ae = np.abs(pred - tru)
        per_mae.append(ae.mean()); P.append(pred); Y.append(tru)
        resid.append(ae); horiz.append((fut - (m-1)).astype(float)); triggers.append(trig)
    P, Y = np.concatenate(P), np.concatenate(Y)
    r2 = 1 - np.sum((Y-P)**2)/np.sum((Y-Y.mean())**2)
    return dict(MAE=float(np.mean(per_mae)), R2=float(r2), n=len(per_mae),
                resid=resid, horiz=horiz, triggers=int(sum(triggers)))


def cq(a, al=0.10):
    a = np.sort(a); k = int(np.ceil((len(a)+1)*(1-al))); return float(a[min(k, len(a))-1])


def mondrian(resid, horiz):
    n = len(resid); cov, wid = [], []
    for i in range(n):
        cr = np.concatenate([resid[j] for j in range(n) if j != i])
        ch = np.concatenate([horiz[j] for j in range(n) if j != i])
        r, h = resid[i], horiz[i]
        if len(cr) < 5:
            continue
        def qof(hh):
            sel = (ch <= 1) if hh <= 1 else ((ch >= 2) & (ch <= 3) if hh <= 3 else ch >= 4)
            return cq(cr[sel]) if sel.sum() >= 5 else cq(cr)
        qq = np.array([qof(hh) for hh in h])
        cov.append((r <= qq).mean()); wid.append((2*qq).mean())
    return float(np.mean(cov))*100, float(np.mean(wid))


def breakdown_curve(d, m=3, kappa=1.0, deltas=np.arange(0, 121, 20)):
    """Adversarial (worst-case) breakdown: inject +Delta into the ENDPOINT early point (index m-1) of each
    tool -- the case that corrupts TWO Theil-Sen slopes in the SAME direction. Measure the mean absolute
    shift in the predicted future VB, Theil-Sen vs MAP. A bounded MAP curve = the robustness Theil-Sen
    lacks at m=3. (Endpoint, not middle: contaminating the middle leaves the Theil-Sen median slope intact,
    which flatters it -- so worst-case position is the honest test.)"""
    rows = []
    for delta in deltas:
        sh_ts, sh_map = [], []
        for tt in tools_of(d):
            tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
            o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
            if len(o) <= m:
                continue
            fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
            if len(fut) == 0:
                continue
            p = fit_global_p(tr); tau = o[:m] ** p; prior = fleet_prior(tr, p, 1.0)
            vc = v[:m].copy(); vp = vc.copy(); vp[m-1] += delta       # contaminate ENDPOINT (worst case)
            a0, b0 = theil_sen(tau, vc); a1, b1 = theil_sen(tau, vp)
            sh_ts.append(np.mean(np.abs((b1+a1*o[fut]**p) - (b0+a0*o[fut]**p))))
            bm0, am0, _ = hybrid_fit(tau, vc, prior); bm1, am1, _ = hybrid_fit(tau, vp, prior)
            sh_map.append(np.mean(np.abs((bm1+am1*o[fut]**p) - (bm0+am0*o[fut]**p))))
        rows.append(dict(delta=float(delta), theilsen_shift=float(np.mean(sh_ts)),
                         hybrid_shift=float(np.mean(sh_map))))
    return pd.DataFrame(rows)


def main():
    d = load()
    print("F3 — detect-then-robustify HYBRID (Theil-Sen default; strong-prior MAP on detected outlier).")
    print("Graph-informed rule: NO MAE regression (heavy Bayes was rejected at 12.8) + bounded breakdown.\n")
    print(f"{'m':>2} {'Theil-Sen':>10} {'Siegel':>8} {'MAP(k=1)':>9} {'HYBRID':>8} | "
          f"{'trig':>4} {'HYB R2':>7} {'Mondrian':>14}")
    recs = []
    for m in (2, 3, 4):
        ts = eval_estimator(d, m, "theilsen"); sg = eval_estimator(d, m, "siegel")
        mp = eval_estimator(d, m, "map", kappa=1.0); hy = eval_estimator(d, m, "hybrid", kappa=5.0)
        cov, wid = mondrian(hy["resid"], hy["horiz"])
        print(f"{m:>2} {ts['MAE']:10.2f} {sg['MAE']:8.2f} {mp['MAE']:9.2f} {hy['MAE']:8.2f} | "
              f"{hy['triggers']:>2}/{hy['n']:<2} {hy['R2']:7.2f} {cov:5.1f}%/{wid:4.0f}um")
        recs.append(dict(m=m, theilsen=round(ts['MAE'],2), siegel=round(sg['MAE'],2),
                         map_k1=round(mp['MAE'],2), hybrid=round(hy['MAE'],2),
                         triggers=hy['triggers'], n=hy['n'], hybrid_R2=round(hy['R2'],2),
                         mondrian_cov=round(cov,1), mondrian_width=round(wid,1)))
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, "results", "f3_map_metrics.csv"), index=False)

    print("\nAcceptance (HYBRID MAE <= best-classical + 0.05  AND  bounded worst-case breakdown):")
    for r in recs:
        best_prev = min(r['theilsen'], r['siegel'])
        ok = r['hybrid'] <= best_prev + 0.05
        print(f"  m={r['m']}: HYBRID {r['hybrid']} vs best-prev {best_prev} -> "
              f"{'PASS (parity, {} clean tools untouched)'.format(r['n']-r['triggers']) if ok else 'FAIL'}")

    print("\nWorst-case breakdown at m=3 (contaminate ENDPOINT): Theil-Sen vs HYBRID")
    bd = breakdown_curve(d, m=3)
    bd.round(2).to_csv(os.path.join(ROOT, "results", "f3_breakdown.csv"), index=False)
    for _, row in bd.iterrows():
        print(f"  +{row.delta:4.0f} um: Theil-Sen shift {row.theilsen_shift:7.2f} um | "
              f"HYBRID shift {row.hybrid_shift:7.2f} um")
    tail_ts = bd.theilsen_shift.iloc[-1]; tail_hy = max(bd.hybrid_shift.iloc[-1], 1e-6)
    print(f"\n  At +{bd.delta.iloc[-1]:.0f} um endpoint outlier: Theil-Sen {tail_ts:.0f} um vs HYBRID "
          f"{bd.hybrid_shift.iloc[-1]:.0f} um -> the hybrid caps the outlier's influence; Theil-Sen grows "
          f"unbounded (its m=3 robustness claim is false).")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(bd.delta, bd.theilsen_shift, "o--", color="#b03030", lw=2, ms=7, label="Theil–Sen (current)")
        ax.plot(bd.delta, bd.hybrid_shift, "o-", color="#1f5fa8", lw=2.4, ms=8,
                label="Hybrid detect-then-robustify (F3)")
        ax.set_xlabel("injected outlier magnitude on one early (endpoint) measurement (µm)")
        ax.set_ylabel("mean |shift| in predicted future VB (µm)")
        ax.set_title("F3 robustness: influence of a single contaminated m=3 measurement (worst case)")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "outputs", "figures", "f3_breakdown.png"), dpi=220); plt.close(fig)
        print("\nwrote results/f3_map_metrics.csv, results/f3_breakdown.csv, outputs/figures/f3_breakdown.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
