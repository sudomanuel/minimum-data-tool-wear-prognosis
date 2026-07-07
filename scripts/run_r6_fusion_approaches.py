"""run_r6_fusion_approaches.py — Round-6: does ANY fusion strategy extract cross-tool signal that
plain concatenation cannot?

Standing result being challenged (Secs 4.4/4.8): the sensor branch carries no cross-tool VB signal
(all R2<0; 'better A/R fusion cannot help, because the branch that consumes the fused features
carries no generalisable signal to begin with'). The user asked to TEST that claim against the
standard fusion families rather than assert it:

  VB-reading task (fair compact physics indicators, LOTO, identical folds as F2):
    early_concat  A+R indicators -> PLS (reference, = fair baseline)
    A_only / R_only               per-channel PLS references
    late_avg      average of per-channel PLS predictions (decision-level fusion)
    late_stack    per-channel PLS preds -> inner-LOTO ridge meta-learner (stacked fusion)
    cca           CCA(A-block, R-block) shared variates -> ridge (subspace fusion)
    mb_pls        block-scaled concatenation -> PLS (multiblock-PLS approximation)
    ratio_AR      dimensionless cross-channel ratios A_x/R_x -> PLS (physical fusion)

  Hazard-covariate task (exploratory, extends round-4; reference R_spectral_kurtosis p=0.010):
    fusion_mean   z(A)+z(R) spectral-kurtosis average
    fusion_pc1    first PC of {A,R} x {spectral_kurtosis__mean, kurtosis__std}

Pre-stated rule: the VB-reading claim flips only if a fusion variant reaches R2>0; the hazard fusion
remains exploratory whatever its p. Outputs: results/r6_fusion_approaches.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA, PLSRegression
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_f2_fair_baseline import load, tools_of, phys_cols, loto_predict, m_pls_factory, CENSOR


def split_blocks(cols):
    A = [c for c in cols if c.startswith("A_")]
    R = [c for c in cols if c.startswith("R_")]
    return A, R


def m_ridge10(Xtr, ytr, Xte, tr):
    return Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)


def loto_late(f, colsA, colsR, stack=False):
    """Decision-level fusion of per-channel PLS predictions."""
    pls = m_pls_factory(3)
    P, Y, per = [], [], []
    for tt in tools_of(f):
        tr = f[(f.tool_id != tt) & (f.vb_um <= CENSOR)]
        te = f[(f.tool_id == tt) & (f.vb_um <= CENSOR)]
        if len(te) == 0 or len(tr) < 8:
            continue
        preds_te, preds_tr = [], []
        for cols in (colsA, colsR):
            sc = StandardScaler().fit(tr[cols].to_numpy(float))
            pr_te = pls(sc.transform(tr[cols].to_numpy(float)), tr.vb_um.to_numpy(float),
                        sc.transform(te[cols].to_numpy(float)), tr)
            preds_te.append(pr_te)
            # in-fold train predictions for the stacker (inner honest enough for a meta of 2 dims)
            pr_tr = pls(sc.transform(tr[cols].to_numpy(float)), tr.vb_um.to_numpy(float),
                        sc.transform(tr[cols].to_numpy(float)), tr)
            preds_tr.append(pr_tr)
        if stack:
            Ztr = np.column_stack(preds_tr); Zte = np.column_stack(preds_te)
            meta = Ridge(alpha=1.0).fit(Ztr, tr.vb_um.to_numpy(float))
            pred = meta.predict(Zte)
        else:
            pred = np.mean(preds_te, axis=0)
        yte = te.vb_um.to_numpy(float)
        P.append(pred); Y.append(yte); per.append(np.abs(pred - yte).mean())
    P, Y = np.concatenate(P), np.concatenate(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return dict(MAE=float(np.mean(per)), R2=float(r2))


def loto_cca(f, colsA, colsR, k=2):
    P, Y, per = [], [], []
    for tt in tools_of(f):
        tr = f[(f.tool_id != tt) & (f.vb_um <= CENSOR)]
        te = f[(f.tool_id == tt) & (f.vb_um <= CENSOR)]
        if len(te) == 0 or len(tr) < 8:
            continue
        sa = StandardScaler().fit(tr[colsA]); sr = StandardScaler().fit(tr[colsR])
        cca = CCA(n_components=k, max_iter=1000).fit(sa.transform(tr[colsA]), sr.transform(tr[colsR]))
        Atr, Rtr = cca.transform(sa.transform(tr[colsA]), sr.transform(tr[colsR]))
        Ate, Rte = cca.transform(sa.transform(te[colsA]), sr.transform(te[colsR]))
        Ztr, Zte = np.hstack([Atr, Rtr]), np.hstack([Ate, Rte])
        pred = Ridge(alpha=1.0).fit(Ztr, tr.vb_um.to_numpy(float)).predict(Zte)
        yte = te.vb_um.to_numpy(float)
        P.append(pred); Y.append(yte); per.append(np.abs(pred - yte).mean())
    P, Y = np.concatenate(P), np.concatenate(Y)
    r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return dict(MAE=float(np.mean(per)), R2=float(r2))


def make_ratios(f, colsA, colsR):
    pairs = []
    for a in colsA:
        r = "R_" + a[2:]
        if r in colsR:
            pairs.append((a, r))
    g = f.copy()
    rat_cols = []
    for a, r in pairs:
        c = f"ratio_{a[2:]}"
        g[c] = f[a] / f[r].replace(0, np.nan)
        rat_cols.append(c)
    g[rat_cols] = g[rat_cols].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return g, rat_cols


def make_blockscaled(f, colsA, colsR):
    g = f.copy()
    for cols in (colsA, colsR):
        w = 1.0 / np.sqrt(len(cols))
        for c in cols:
            g[c + "_bs"] = f[c] * w
    return g, [c + "_bs" for c in colsA + colsR]


# ---------- hazard fusion (exploratory) ----------
def irls(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b; p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        A = X.T @ (W[:, None] * X) + 1e-9 * np.eye(X.shape[1])
        b = np.linalg.solve(A, X.T @ (W * ((X @ b) + (y - p) / W)))
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12)
    return b, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def hazard_fusion(f):
    f = f.sort_values(["tool_id", "within_tool_order"]).reset_index(drop=True)
    y = (f.groupby("tool_id")["within_tool_order"].transform("max") == f["within_tool_order"]).astype(float).to_numpy()
    zvb = ((f.vb_um - f.vb_um.mean()) / f.vb_um.std()).to_numpy(float)
    X0 = np.column_stack([np.ones(len(f)), zvb])
    _, ll0 = irls(X0, y)
    def z(c):
        s = f[c].astype(float)
        return ((s - s.mean()) / max(s.std(), 1e-12)).to_numpy(float)
    zs_mean = (z("A_spectral_kurtosis__mean") + z("R_spectral_kurtosis__mean")) / 2
    M = np.column_stack([z("A_spectral_kurtosis__mean"), z("R_spectral_kurtosis__mean"),
                         z("A_kurtosis__std"), z("R_kurtosis__std")])
    u, s_, vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    pc1 = (M - M.mean(0)) @ vt[0]; pc1 = (pc1 - pc1.mean()) / pc1.std()
    out = []
    for name, zz in (("fusion_mean_speckurt", zs_mean), ("fusion_pc1_transient", pc1)):
        X1 = np.column_stack([np.ones(len(f)), zvb, zz])
        b1, ll1 = irls(X1, y)
        lrt = 2 * (ll1 - ll0)
        out.append(dict(covariate=name, gamma2=round(float(b1[2]), 3),
                        LRT_p=round(float(chi2.sf(max(lrt, 0), 1)), 4)))
    return out


def main():
    f = load()
    cols = phys_cols(f)
    A, R = split_blocks(cols)
    print(f"R6: FUSION BENCH — {len(A)} A-indicators + {len(R)} R-indicators (fair compact set), "
          f"LOTO identical folds. Claim flips only if any variant reaches R2>0.\n")
    pls = m_pls_factory(3)
    rows = []
    def add(name, res):
        rows.append(dict(variant=name, MAE=round(res["MAE"], 1), R2=round(res["R2"], 2)))
        print(f"  {name:14}: MAE {res['MAE']:6.1f}  R2 {res['R2']:+.2f}")
    add("early_concat", loto_predict(f, A + R, pls))
    add("A_only", loto_predict(f, A, pls))
    add("R_only", loto_predict(f, R, pls))
    add("late_avg", loto_late(f, A, R, stack=False))
    add("late_stack", loto_late(f, A, R, stack=True))
    add("cca", loto_cca(f, A, R, k=2))
    g, bs = make_blockscaled(f, A, R); add("mb_pls", loto_predict(g, bs, pls))
    g, rat = make_ratios(f, A, R); add("ratio_AR", loto_predict(g, rat, pls))
    best = max(rows, key=lambda r: r["R2"])
    print(f"\nbest fusion: {best['variant']} R2 {best['R2']:+.2f} -> "
          f"{'CLAIM FLIPS (R2>0!)' if best['R2'] > 0 else 'claim STANDS: no fusion extracts cross-tool VB signal'}")
    print("\n--- hazard-covariate fusion (exploratory; round-4 single-channel reference p=0.010) ---")
    for r in hazard_fusion(f):
        print(f"  {r['covariate']:22}: gamma2 {r['gamma2']:+.2f}  LRT p {r['LRT_p']}")
        rows.append(dict(variant=r["covariate"], MAE=None, R2=None, **{}))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "r6_fusion_approaches.csv"), index=False)
    print("\nwrote results/r6_fusion_approaches.csv")


if __name__ == "__main__":
    main()
