"""run_c_hazard_robustness.py — Reviewer-response C: robustness of the exploratory hazard covariate.

The round-4 finding (R_spectral_kurtosis adds hazard information beyond VB, LRT p=0.010 nominal) was
fit on 172 pooled cycles ignoring within-tool correlation, with chi-square asymptotics on 18 events.
Both are legitimate reviewer objections. This script upgrades the inference WITHOUT changing the
claim's exploratory status:
  1. cluster-robust (sandwich, CR1) standard errors clustering by tool -> z-test for gamma2;
  2. exact permutation LRT (s permuted WITHIN each tool, preserving tool-level distributions while
     breaking the cycle-level association; 10,000 permutations);
  3. leave-one-tool-out influence: does nominal significance hinge on any single tool?

Run for the two round-4 signals: R_spectral_kurtosis__mean and R_kurtosis__std.
Outputs: results/c_hazard_robustness.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")
N_PERM = 10000
RNG = np.random.default_rng(42)


def irls(X, y, max_iter=60, tol=1e-10):
    """Logistic MLE by IRLS; returns (beta, loglik)."""
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        z = eta + (y - p) / W
        A = X.T @ (W[:, None] * X) + 1e-9 * np.eye(X.shape[1])
        bn = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(bn - b)) < tol:
            b = bn; break
        b = bn
    eta = X @ b
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return b, ll


def sandwich_cluster(X, y, b, clusters):
    """CR1 cluster-robust covariance for logistic MLE."""
    eta = X @ b
    p = 1 / (1 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-10, None)
    A = X.T @ (W[:, None] * X)
    G = np.unique(clusters)
    B = np.zeros((X.shape[1], X.shape[1]))
    scores = X * (y - p)[:, None]
    for g in G:
        sg = scores[clusters == g].sum(axis=0)
        B += np.outer(sg, sg)
    g_, n_, k_ = len(G), len(y), X.shape[1]
    cr1 = (g_ / (g_ - 1)) * ((n_ - 1) / (n_ - k_))
    Ainv = np.linalg.inv(A + 1e-9 * np.eye(k_))
    return cr1 * (Ainv @ B @ Ainv)


def lrt(X0, X1, y):
    _, l0 = irls(X0, y)
    b1, l1 = irls(X1, y)
    return 2 * (l1 - l0), b1


def main():
    f = pd.read_csv(FEAT).sort_values(["tool_id", "within_tool_order"]).reset_index(drop=True)
    f["chip"] = (f.groupby("tool_id")["within_tool_order"].transform("max")
                 == f["within_tool_order"]).astype(float)
    y = f["chip"].to_numpy(float)
    zvb = ((f["vb_um"] - f["vb_um"].mean()) / f["vb_um"].std()).to_numpy(float)
    tools = f["tool_id"].to_numpy()
    X0 = np.column_stack([np.ones(len(f)), zvb])
    print(f"C: HAZARD-COVARIATE ROBUSTNESS — {len(f)} cycles, {int(y.sum())} events, "
          f"{len(np.unique(tools))} tool clusters, {N_PERM} permutations\n")

    rows = []
    for cov in ("R_spectral_kurtosis__mean", "R_kurtosis__std"):
        s = f[cov].astype(float)
        zs = ((s - s.mean()) / max(s.std(), 1e-12)).to_numpy(float)
        X1 = np.column_stack([np.ones(len(f)), zvb, zs])
        lrt_obs, b1 = lrt(X0, X1, y)
        p_chi2 = float(chi2.sf(max(lrt_obs, 0), 1))

        # 1. cluster-robust z-test
        V = sandwich_cluster(X1, y, b1, tools)
        se = float(np.sqrt(V[2, 2]))
        z = b1[2] / se
        p_cr = 2 * float(norm.sf(abs(z)))

        # 2. within-tool permutation LRT
        idx_by_tool = [np.where(tools == t)[0] for t in np.unique(tools)]
        cnt = 0
        for _ in range(N_PERM):
            zp = zs.copy()
            for idx in idx_by_tool:
                zp[idx] = zp[RNG.permutation(idx)]
            Xp = np.column_stack([np.ones(len(f)), zvb, zp])
            lp, _ = lrt(X0, Xp, y)
            if lp >= lrt_obs:
                cnt += 1
        p_perm = (1 + cnt) / (1 + N_PERM)

        # 3. leave-one-tool-out influence
        loo_p = []
        for t in np.unique(tools):
            keep = tools != t
            l, _ = lrt(X0[keep], X1[keep], y[keep])
            loo_p.append(float(chi2.sf(max(l, 0), 1)))
        loo_max = max(loo_p); frac_sig = np.mean([p < 0.05 for p in loo_p])

        print(f"{cov}:")
        print(f"  gamma2 {b1[2]:+.2f} | LRT chi2 p = {p_chi2:.4f} (round-4 reference)")
        print(f"  1. cluster-robust (18 tool clusters, CR1): se {se:.3f}, z {z:+.2f}, p = {p_cr:.4f}")
        print(f"  2. within-tool permutation LRT: p = {p_perm:.4f}")
        print(f"  3. leave-one-tool influence: max p {loo_max:.3f} | folds nominally significant "
              f"{frac_sig*100:.0f}%\n")
        rows.append(dict(covariate=cov, gamma2=round(float(b1[2]), 3), p_chi2=round(p_chi2, 4),
                         p_cluster_robust=round(p_cr, 4), p_permutation=round(p_perm, 4),
                         loo_max_p=round(loo_max, 4), loo_frac_sig=round(float(frac_sig), 2)))

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "c_hazard_robustness.csv"), index=False)
    print("wrote results/c_hazard_robustness.csv")


if __name__ == "__main__":
    main()
