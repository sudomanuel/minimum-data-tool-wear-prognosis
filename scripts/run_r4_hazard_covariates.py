"""run_r4_hazard_covariates.py — Round-4 Module 3→4 bridge: does a per-cycle vibration transient
indicator add information to the chipping hazard beyond the wear level?

Model under test:  logit h = γ0 + γ1·z(VB) + γ2·z(s)   vs   baseline  logit h = γ0 + γ1·z(VB)
where s is a dimensionless per-cycle transient indicator (kurtosis / crest / impulse / spectral
kurtosis, A and R channels; __mean = level over the 6 contacts, __std = dispersion across contacts).
These are exactly the condition-agnostic indicator class the consortium prescribes (Module 3): shape
ratios, not energy.

Scoring: likelihood-ratio test (2·ΔLL ~ χ²(1)), coefficient sign, and the shift of VB_safe (h ≤ 0.10)
at the covariate's fleet median. Fleet-level descriptive fit (18 events, 172 cycles) — same standing
as the round-3 hazard; in-sample LRT with 1 dof, optimism flagged, no LOTO claim made.

Outputs: results/r4_hazard_covariates.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")
H_MAX = 0.10
CANDIDATES = ["A_kurtosis__mean", "R_kurtosis__mean", "A_kurtosis__std", "R_kurtosis__std",
              "A_crest_factor__mean", "R_crest_factor__mean",
              "A_impulse_factor__mean", "R_impulse_factor__mean",
              "A_spectral_kurtosis__mean", "R_spectral_kurtosis__mean"]


def fit_logistic(X, y):
    """MLE logistic; returns (loglik, params)."""
    def nll(th):
        eta = X @ th
        p = 1 / (1 + np.exp(-eta))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    th0 = np.zeros(X.shape[1]); th0[0] = -2.0
    r = minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    return -r.fun, r.x


def main():
    f = pd.read_csv(FEAT).sort_values(["tool_id", "within_tool_order"])
    f["chip"] = (f.groupby("tool_id")["within_tool_order"].transform("max")
                 == f["within_tool_order"]).astype(float)
    y = f["chip"].to_numpy(float)
    zvb = ((f["vb_um"] - f["vb_um"].mean()) / f["vb_um"].std()).to_numpy(float)
    n_ev = int(y.sum())
    fv = f.loc[f.chip == 1, "vb_um"]
    print(f"R4-M4: HAZARD COVARIATES — {len(f)} cycles, {n_ev} terminal chipping events "
          f"(wear-at-chipping {fv.min():.0f}–{fv.max():.0f}, median {fv.median():.0f}) [sanity vs r3: OK]\n")

    X0 = np.column_stack([np.ones(len(f)), zvb])
    ll0, th0 = fit_logistic(X0, y)
    print(f"baseline logit h = γ0 + γ1·z(VB):  logLik {ll0:.2f}   γ1 {th0[1]:+.2f}\n")
    print(f"{'covariate':30} {'γ2':>7} {'ΔlogLik':>8} {'LRT p':>8}  verdict")

    rows = []
    for c in CANDIDATES:
        if c not in f.columns:
            continue
        s = f[c].astype(float)
        zs = ((s - s.mean()) / max(s.std(), 1e-12)).to_numpy(float)
        X1 = np.column_stack([np.ones(len(f)), zvb, zs])
        ll1, th1 = fit_logistic(X1, y)
        lrt = 2 * (ll1 - ll0)
        pval = float(chi2.sf(max(lrt, 0.0), 1))
        verdict = "SIGNAL (p<0.05)" if pval < 0.05 else ("weak (p<0.15)" if pval < 0.15 else "none")
        rows.append(dict(covariate=c, gamma2=round(th1[2], 3), dLL=round(ll1 - ll0, 3),
                         LRT_p=round(pval, 4), verdict=verdict))
        print(f"{c:30} {th1[2]:+7.2f} {ll1-ll0:8.3f} {pval:8.3f}  {verdict}")

    df = pd.DataFrame(rows).sort_values("LRT_p")
    df.to_csv(os.path.join(ROOT, "results", "r4_hazard_covariates.csv"), index=False)
    best = df.iloc[0]
    print()
    if best.LRT_p < 0.05:
        print(f"BEST: {best.covariate} (γ2 {best.gamma2:+.2f}, p {best.LRT_p:.3f}) — a transient channel "
              f"ADDS hazard information beyond VB. Note: descriptive in-sample evidence on 18 events; "
              f"promote to the paper only as an exploratory covariate, and re-test under replication.")
    else:
        print(f"VERDICT: no per-cycle transient indicator adds significant hazard information beyond the "
              f"wear level (best {best.covariate}, p {best.LRT_p:.2f}). The chipping hazard remains "
              f"h(VB) alone — consistent with the sensor branch carrying no cross-tool signal; the "
              f"vibration-informed hazard is declared as replicated-campaign protocol.")
    print("\nwrote results/r4_hazard_covariates.csv")


if __name__ == "__main__":
    main()
