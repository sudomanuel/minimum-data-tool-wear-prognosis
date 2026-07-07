"""run_r7_kernel_conformal.py — Round-7 directive C adjudication: kernel-smoothed conformal
prediction with residuals weighted by KINEMATIC distance in the DOE space, versus the deployed
horizon-Mondrian band.

Construction (as demanded by the panel): for held-out tool u, every calibration residual from tool c
receives weight w_c = exp(-0.5 ||z_u - z_c||^2 / h^2) with z = standardized (ln v_c, ln f_z, cooling);
the band half-width is the weighted 90% quantile of |residuals| (weighted-quantile version of split
conformal; finite-sample validity holds only approximately under weighting — checked EMPIRICALLY per
project convention: per-tool coverage averaging, same residual set as the deployed band).
Variants: bandwidth grid h in {0.5, 1.0, 2.0, inf(=global)}; and kernel x horizon (the soft version
of severity-Mondrian, rejected in round 2 by bin budget).
Reference to beat: Mondrian 90.1% PICP / 52.5 um mean width (m=3 deployed config).
Pre-stated rule: adopt only if PICP >= 90 AND width < 52.5. Outputs: results/r7_kernel_conformal.csv
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
from run_optimal_config_search import fit_p, FIT
CENSOR = 300.0; AL = 0.10
FEAT = os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv")
REF = dict(PICP=90.1, width=52.5)


def local_p_grid(o, v, m, p_star, lam=200.0):
    best_p, best = p_star, np.inf
    for pc in np.arange(max(p_star - 0.15, 0.05), p_star + 0.1501, 0.05):
        tau = o[:m] ** pc
        a, b = theil_sen(tau, v[:m])
        sse = float(np.sum((b + a * tau - v[:m]) ** 2)) + lam * (pc - p_star) ** 2
        if sse < best:
            best, best_p = sse, pc
    return best_p


def residuals(d, m=3):
    """Deployed m=3 config residuals per tool: (signed, horizon)."""
    res = {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = local_p_grid(o, v, m, fit_p(tr))
        a, b = FIT["siegel"](o[:m] ** p, v[:m])
        sr = (b + a * o[fut] ** p) - v[fut]
        res[tt] = (sr, (fut - (m - 1)).astype(float))
    return res


def kin_coords():
    f = pd.read_csv(FEAT).groupby("tool_id").first()
    z = pd.DataFrame(dict(
        lvc=np.log(f.vc.astype(float)), lfz=np.log(f.fz.astype(float)),
        cool=f["cooling"].astype(str).str.lower().map(
            lambda s: 0.0 if ("dry" in s or "no" in s) else 1.0)))
    return (z - z.mean()) / z.std()


def wquant(x, w, q):
    idx = np.argsort(x)
    x, w = np.asarray(x)[idx], np.asarray(w)[idx]
    cw = np.cumsum(w) / np.sum(w)
    k = np.searchsorted(cw, q)
    return float(x[min(k, len(x) - 1)])


def hbin(h):
    return 0 if h <= 1 else (1 if h <= 3 else 2)


def evaluate(res, Z, h, per_horizon):
    tools = list(res)
    cov, wid = [], []
    for u in tools:
        zu = Z.loc[u].to_numpy(float)
        rs, hs, ws = [], [], []
        for c in tools:
            if c == u:
                continue
            w = 1.0 if not np.isfinite(h) else float(np.exp(-0.5 * np.sum((zu - Z.loc[c].to_numpy(float)) ** 2) / h ** 2))
            sr, hh = res[c]
            rs += list(np.abs(sr)); hs += [hbin(x) for x in hh]; ws += [w] * len(sr)
        rs, hs, ws = np.array(rs), np.array(hs), np.array(ws)
        sr_u, hh_u = res[u]
        tc, tw = [], []
        for s, hz in zip(sr_u, hh_u):
            if per_horizon:
                sel = hs == hbin(hz)
                if ws[sel].sum() < 1e-9 or sel.sum() < 8:
                    sel = np.ones(len(rs), bool)
            else:
                sel = np.ones(len(rs), bool)
            q = wquant(rs[sel], ws[sel], 1 - AL)
            tc.append(abs(s) <= q); tw.append(2 * q)
        cov.append(np.mean(tc)); wid.append(np.mean(tw))
    return 100 * float(np.mean(cov)), float(np.mean(wid))


def main():
    d = load()
    res = residuals(d, m=3)
    Z = kin_coords()
    print("R7-C: KERNEL-SMOOTHED (KINEMATIC-DISTANCE) CONFORMAL vs deployed Mondrian "
          f"({REF['PICP']}% / {REF['width']} um). Rule: adopt only if PICP>=90 AND width<52.5.\n")
    rows = []
    for per_h in (False, True):
        for h in (0.5, 1.0, 2.0, np.inf):
            picp, w = evaluate(res, Z, h, per_h)
            tag = f"kernel h={h}" + (" x horizon" if per_h else " (global)")
            ok = "** ADOPTABLE **" if (picp >= 90 and w < REF["width"]) else ""
            rows.append(dict(scheme=tag, PICP=round(picp, 1), width_um=round(w, 1)))
            print(f"  {tag:26}: PICP {picp:5.1f}%  width {w:6.1f} um  {ok}")
        print()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "r7_kernel_conformal.csv"), index=False)
    win = df[(df.PICP >= 90) & (df.width_um < REF["width"])]
    print("WINNERS:" if len(win) else
          "VERDICT: kinematic-kernel weighting does not dominate the horizon-Mondrian band — the "
          "error scale is driven by the FORECAST HORIZON, not by DOE proximity (the same physics that "
          "rejected severity-Mondrian and the normalized scores).")
    if len(win):
        print(win.to_string(index=False))
    print("wrote results/r7_kernel_conformal.csv")


if __name__ == "__main__":
    main()
