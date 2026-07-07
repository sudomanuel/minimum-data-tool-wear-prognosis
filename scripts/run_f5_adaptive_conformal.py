"""run_f5_adaptive_conformal.py — FRONT 5: shrink the conformal interval WITHOUT breaking 90% coverage.

Replace the absolute nonconformity score |r| by a NORMALIZED score S = |r| / sigma_hat(x), calibrated
leakage-safe, then band = q * sigma_hat (finite-sample conformal q on the normalized scores -> valid by
construction). A half-width FLOOR (microscope repeatability) prevents the band from collapsing below the
measurement noise -- the failure mode that made naive normalized conformal under-cover (82%).

We do NOT assume which scale is right (our power law is concave/decelerating, so a physics-slope scale is
anti-correlated with forecast difficulty). We compare candidate scales empirically and keep the tightest
that stays valid:
  global      : sigma_hat = 1                         (= absolute score; the baseline)
  mondrian    : per-horizon-bin absolute quantile      (current deployed band)
  norm_horizon: sigma_hat = median|r| as a function of forecast horizon h  (data-driven)
  norm_level  : sigma_hat proportional to predicted VB level
  norm_physics: sigma_hat proportional to |dVB/dorder| = a*p*order^(p-1)   (proposal's suggestion)
Metrics: PICP (target 0.90), MPIW, NMPIW=MPIW/200um, mean Winkler interval score. Adopt tightest valid.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of
CENSOR = 300.0; M = 3; AL = 0.10; VB_RANGE = 200.0
HW_FLOOR = 4.0     # prediction-interval half-width floor (um) ~ optical-microscope repeatability


def collect(d, m=M):
    """Per future point (LOTO few-shot): abs residual + scale covariates."""
    rows = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m:
            continue
        fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr); a, b = theil_sen(o[:m]**p, v[:m])
        for k in fut:
            pred = b + a*o[k]**p
            rows.append(dict(tool=tt, r=abs(pred - v[k]), h=float(k-(m-1)), order=float(o[k]),
                             level=max(pred, 1.0), slope=max(a*p*o[k]**(p-1), 1e-6)))
    return pd.DataFrame(rows)


def mondrian_widths(df, al):
    """Per-tool Mondrian at level (1-al): return (PICP, mean width, NEAR-bin width)."""
    cov, wid, near = [], [], []
    tools = df.tool.unique()
    for tt in tools:
        cal = df[df.tool != tt]; te = df[df.tool == tt]
        if len(cal) < 5 or len(te) == 0:
            continue
        ch, cr = cal.h.to_numpy(), cal.r.to_numpy()
        tc, tw = [], []
        for _, row in te.iterrows():
            hh = row["h"]
            sel = (ch <= 1) if hh <= 1 else ((ch >= 2) & (ch <= 3) if hh <= 3 else ch >= 4)
            q = cq(cr[sel], al) if sel.sum() >= 5 else cq(cr, al)
            hw = max(q, HW_FLOOR); tc.append(row["r"] <= hw); tw.append(2*hw)
            if hh <= 1:
                near.append(2*hw)
        cov.append(np.mean(tc)); wid.append(np.mean(tw))
    return np.mean(cov)*100, np.mean(wid), (np.mean(near) if near else np.nan)


def cq(arr, al=AL):
    arr = np.sort(np.asarray(arr, float))
    k = int(np.ceil((len(arr)+1)*(1-al)))
    return float(arr[min(k, len(arr))-1])


def scale_fn(method, cal):
    """Return a function x_row-> sigma_hat (shape only; overall scale absorbed by q). Fit on calibration."""
    if method in ("global", "mondrian"):
        return lambda row: 1.0
    if method == "norm_horizon":
        # median |r| as a smooth power of horizon: log m = c + gamma*log h
        hs = cal.h.to_numpy(); rr = np.maximum(cal.r.to_numpy(), 1e-3)
        A = np.column_stack([np.ones(len(hs)), np.log(hs)])
        coef, *_ = np.linalg.lstsq(A, np.log(rr), rcond=None)
        return lambda row: float(np.exp(coef[0] + coef[1]*np.log(row["h"])))
    if method == "norm_level":
        return lambda row: float(row["level"])
    if method == "norm_physics":
        return lambda row: float(row["slope"])
    raise ValueError(method)


def evaluate(df, method, floor=HW_FLOOR):
    """Per-tool aggregation (matches run_cqr_cv_test.py so the Mondrian baseline reproduces 52um)."""
    cov, wid, wink = [], [], []
    tools = df.tool.unique()
    for tt in tools:
        cal = df[df.tool != tt]; te = df[df.tool == tt]
        if len(cal) < 5 or len(te) == 0:
            continue
        if method == "mondrian":
            ch = cal.h.to_numpy(); cr = cal.r.to_numpy()
            def hw_of(row):
                hh = row["h"]
                sel = (ch <= 1) if hh <= 1 else ((ch >= 2) & (ch <= 3) if hh <= 3 else ch >= 4)
                q = cq(cr[sel]) if sel.sum() >= 5 else cq(cr)
                return max(q, floor)
        else:
            sf = scale_fn(method, cal)
            s_cal = cal.r.to_numpy() / np.array([sf(r) for _, r in cal.iterrows()])
            q = cq(s_cal)
            hw_of = lambda row: max(q * sf(row), floor)
        tc, tw, tk = [], [], []
        for _, row in te.iterrows():
            hw = hw_of(row)
            tc.append(row["r"] <= hw); tw.append(2*hw)
            tk.append(2*hw + (2/AL)*max(0.0, row["r"]-hw))
        cov.append(np.mean(tc)); wid.append(np.mean(tw)); wink.append(np.mean(tk))
    return (np.mean(cov)*100, np.mean(wid), np.mean(wid)/VB_RANGE*100, np.mean(wink))


def main():
    df = collect(load())
    methods = ["global", "mondrian", "norm_horizon", "norm_level", "norm_physics"]
    print(f"F5 adaptive conformal (LOTO, m={M}, target {int((1-AL)*100)}% coverage, "
          f"half-width floor {HW_FLOOR:.0f}um). {len(df)} future points.\n")
    print(f"{'method':13} {'PICP':>6} {'MPIW(um)':>9} {'NMPIW':>7} {'Winkler':>8}")
    recs = []
    for mth in methods:
        picp, mpiw, nmpiw, wink = evaluate(df, mth)
        print(f"{mth:13} {picp:5.1f}% {mpiw:9.1f} {nmpiw:6.1f}% {wink:8.1f}")
        recs.append(dict(method=mth, PICP=round(picp,1), MPIW_um=round(mpiw,1),
                         NMPIW_pct=round(nmpiw,1), Winkler=round(wink,1)))
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, "results", "f5_adaptive_conformal.csv"), index=False)

    valid = [r for r in recs if r["PICP"] >= 88.0]
    cur = next(r for r in recs if r["method"] == "mondrian")
    best = min(valid, key=lambda r: r["MPIW_um"]) if valid else None
    print("\nAcceptance (PICP>=88 AND MPIW < current Mondrian %.1fum):" % cur["MPIW_um"])
    if best and best["method"] != "mondrian" and best["MPIW_um"] < cur["MPIW_um"] - 0.5:
        print(f"  ADOPT '{best['method']}': {cur['MPIW_um']}um -> {best['MPIW_um']}um "
              f"({100*(cur['MPIW_um']-best['MPIW_um'])/cur['MPIW_um']:.0f}% tighter) at PICP {best['PICP']}%.")
    else:
        print(f"  Keep Mondrian ({cur['MPIW_um']}um @ {cur['PICP']}%). No adaptive scale is tighter at fixed "
              f"coverage -> the score-engineering lever is exhausted; the band is at its data floor.")

    # ---- honest tightening frontier: the levers that DO narrow the band ----
    print("\nTightening frontier (Mondrian) — width vs coverage target vs budget m:")
    print(f"{'m':>2} {'target':>7} {'PICP':>6} {'mean width':>11} {'NEAR-horizon width':>19}")
    d = load(); fr = []
    for m in (3, 4):
        dm = collect(d, m)
        for al, tgt in [(0.10, "90%"), (0.20, "80%")]:
            picp, mw, nw = mondrian_widths(dm, al)
            print(f"{m:>2} {tgt:>7} {picp:5.1f}% {mw:10.1f}u {nw:16.1f}u")
            fr.append(dict(m=m, target=tgt, PICP=round(picp,1), mean_width=round(mw,1),
                           near_width=round(nw,1)))
    pd.DataFrame(fr).to_csv(os.path.join(ROOT, "results", "f5_tightening_frontier.csv"), index=False)
    print("\n  Operational note: in rolling deployment only the NEAR-horizon band is used each step; it is "
          "already the tightest usable interval. Genuine width reduction comes from coverage target, "
          "more early points (m), or replication — not from nonconformity-score engineering.")
    print("\nwrote results/f5_adaptive_conformal.csv, results/f5_tightening_frontier.csv")


if __name__ == "__main__":
    main()
