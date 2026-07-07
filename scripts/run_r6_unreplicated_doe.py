"""run_r6_unreplicated_doe.py — Round-6: classical unreplicated-factorial inference on the 3x3x2 DOE.

The predictive question (condition -> new tool's wear, LOTO) is adjudicated NEGATIVE (Taylor round-2,
p(x) round-4). This script asks the OTHER, so-far-untested statistical question — the inferential
one: with one tool per condition, WHICH factors have detectable effects on the wear responses, with
what percentage contribution? Classical machinery for exactly this regime:
  - orthogonal polynomial contrasts of the 3x3x2 (vc lin/quad, fz lin/quad, cooling; 2-factor
    interactions), normalized so all 17 effect estimates share one scale;
  - Lenth (1989) pseudo-standard-error test on the 17 effects (no replicates needed);
  - classical ANOVA using the 3-way interaction (4 df) as the error term (Taguchi-style pooling);
  - half-normal ordering (Daniel 1959) reported as ranks.
Responses (one value per tool, 18 tools): log wear-rate log(a_k) at the fleet exponent; per-tool
exponent p_k (caveat: noisy); break-in level VB0; wear-at-chipping VB_chip.
Multiplicity: 4 responses x 13 effects — Lenth p-values reported raw AND with Holm within response.
Pre-stated: report as comes; integrate into the paper only if clean actives emerge.
Outputs: results/r6_unreplicated_doe.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, theil_sen, tools_of
from run_optimal_config_search import fit_p
CENSOR = 300.0
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")
P_GRID = np.arange(0.05, 0.6001, 0.01)


def per_tool_responses(d):
    p_fleet = fit_p(d)
    rows = {}
    for tt, g in d.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        o, v = gg.order.to_numpy(float), gg.vb.to_numpy(float)
        if len(o) < 3:
            continue
        a, _ = theil_sen(o ** p_fleet, v)
        best_p, best = p_fleet, np.inf
        for pc in P_GRID:
            aa, bb = theil_sen(o ** pc, v)
            sse = float(np.sum((bb + aa * o ** pc - v) ** 2))
            if sse < best:
                best, best_p = sse, pc
        g_full = g.sort_values("order")
        rows[tt] = dict(log_rate=np.log(max(a, 1e-3)), p_hat=best_p,
                        VB0=float(g_full.vb.iloc[0]), VB_chip=float(g_full.vb.iloc[-1]))
    return pd.DataFrame(rows).T


def orth_contrasts(cond):
    """Orthonormal contrast matrix for the 3x3x2 (17 effects), columns unit-norm."""
    lv = {55: -1, 70: 0, 80: 1}
    lf = {0.08: -1, 0.2: 0, 0.3: 1}
    vc_l = cond.vc.map(lv).to_numpy(float)
    fz_l = cond.fz.map(lf).to_numpy(float)
    co = np.where(cond.cool.to_numpy() > 0.5, 1.0, -1.0)
    def quad(x):  # orthogonal quadratic for 3 equally-weighted levels
        return 3 * x ** 2 - 2
    cols = {
        "vc_lin": vc_l, "vc_quad": quad(vc_l),
        "fz_lin": fz_l, "fz_quad": quad(fz_l),
        "cool": co,
        "vcL:fzL": vc_l * fz_l, "vcL:fzQ": vc_l * quad(fz_l),
        "vcQ:fzL": quad(vc_l) * fz_l, "vcQ:fzQ": quad(vc_l) * quad(fz_l),
        "vcL:cool": vc_l * co, "vcQ:cool": quad(vc_l) * co,
        "fzL:cool": fz_l * co, "fzQ:cool": quad(fz_l) * co,
        # 3-way (error pool for the ANOVA variant)
        "3w_LLc": vc_l * fz_l * co, "3w_LQc": vc_l * quad(fz_l) * co,
        "3w_QLc": quad(vc_l) * fz_l * co, "3w_QQc": quad(vc_l) * quad(fz_l) * co,
    }
    X = pd.DataFrame(cols, index=cond.index)
    X = X / np.sqrt((X ** 2).sum(axis=0))          # unit-norm -> comparable effect scale
    return X


def lenth_test(effects):
    """Lenth (1989): PSE + t-like p-values (df ~ m/3)."""
    e = np.asarray(effects, float)
    s0 = 1.5 * np.median(np.abs(e))
    pse = 1.5 * np.median(np.abs(e)[np.abs(e) < 2.5 * s0])
    df = len(e) / 3.0
    t = e / max(pse, 1e-12)
    p = 2 * stats.t.sf(np.abs(t), df)
    return pse, t, p


def main():
    d = load()
    resp = per_tool_responses(d)
    f = pd.read_csv(FEAT).groupby("tool_id").first()
    cond = pd.DataFrame(dict(
        vc=f.vc.astype(float), fz=f.fz.astype(float),
        cool=f["cooling"].astype(str).str.lower().map(
            lambda s: 0.0 if ("dry" in s or "no" in s) else 1.0))).loc[resp.index]
    X = orth_contrasts(cond)
    err_cols = [c for c in X.columns if c.startswith("3w_")]
    eff_cols = [c for c in X.columns if not c.startswith("3w_")]

    print("R6: UNREPLICATED 3x3x2 DOE INFERENCE — Lenth PSE + ANOVA (3-way pooled as error)\n"
          f"tools: {len(resp)} | effects tested: {len(eff_cols)} | error df (ANOVA): {len(err_cols)}\n")
    all_rows = []
    for yname in ("log_rate", "p_hat", "VB0", "VB_chip"):
        y = resp[yname].to_numpy(float)
        yc = y - y.mean()
        est = X.T.to_numpy() @ yc                      # orthonormal effect estimates
        est = pd.Series(est, index=X.columns)
        # Lenth on the 17 effects
        pse, tvals, pl = lenth_test(est.values)
        lenth = pd.Series(pl, index=X.columns)
        # ANOVA: SS per effect = est^2 (orthonormal); error = mean SS of 3-way pool
        ss = est ** 2
        ms_err = ss[err_cols].mean()
        F = ss[eff_cols] / ms_err
        pF = pd.Series(stats.f.sf(F, 1, len(err_cols)), index=eff_cols)
        contrib = 100 * ss[eff_cols] / ss.sum()
        # Holm within response over the 13 non-error effects (Lenth p)
        pl13 = lenth[eff_cols].sort_values()
        holm = {}
        for i, (nm, pv) in enumerate(pl13.items()):
            holm[nm] = min(pv * (len(pl13) - i), 1.0)
        holm = pd.Series(holm)
        print(f"--- response: {yname} (sd {y.std(ddof=1):.3f}) ---")
        tab = pd.DataFrame(dict(effect=eff_cols,
                                estimate=[round(est[c], 3) for c in eff_cols],
                                contrib_pct=[round(contrib[c], 1) for c in eff_cols],
                                p_Lenth=[round(lenth[c], 4) for c in eff_cols],
                                p_Holm=[round(holm[c], 4) for c in eff_cols],
                                p_ANOVA=[round(pF[c], 4) for c in eff_cols]))
        tab = tab.sort_values("p_Lenth")
        act = tab[(tab.p_Lenth < 0.05)]
        print(tab.head(6).to_string(index=False))
        sig_holm = tab[tab.p_Holm < 0.05].effect.tolist()
        sig_both = tab[(tab.p_Lenth < 0.05) & (tab.p_ANOVA < 0.05)].effect.tolist()
        print(f"  activos (Lenth<.05): {act.effect.tolist() or 'ninguno'} | sobreviven Holm: "
              f"{sig_holm or 'ninguno'} | Lenth&ANOVA: {sig_both or 'ninguno'}\n")
        for _, r in tab.iterrows():
            all_rows.append(dict(response=yname, **r))
    pd.DataFrame(all_rows).to_csv(os.path.join(ROOT, "results", "r6_unreplicated_doe.csv"), index=False)
    print("wrote results/r6_unreplicated_doe.csv")


if __name__ == "__main__":
    main()
