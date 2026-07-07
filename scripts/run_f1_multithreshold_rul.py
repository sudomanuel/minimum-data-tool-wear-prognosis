"""run_f1_multithreshold_rul.py — FRONT 1: dissolve the N=3 RUL-validation problem WITHOUT new data.

Idea: instead of validating a single crossing of VB_fail=200 um (only ~3 tools reach it after their
few-shot window), validate the crossing-prediction OPERATOR at a ladder of wear thresholds
VB_c in {120,150,175,200} um. Every (tool, threshold) whose TRUE crossing lies after the few-shot window
and within the observed record is a validatable interval-censored event -> N grows from 3 to ~dozens.

For each event, from the first m=3 measurements (LOTO, leakage-safe exponent), we:
  - fit the physics few-shot curve VB=b+a*order^p,
  - predict the crossing order t_hat_c  (solve b+a*o^p = VB_c),
  - form a first-order RUL window from the Mondrian band (width q / local slope),
and compare against the TRUE crossing t_c (linear-interpolated between bracketing measured points).

Reliability metrics (Saxena et al. + coverage):
  RA = 1 - |RUL_true - RUL_pred|/RUL_true ;  CRA = mean RA ;  alpha-accuracy (|rel err|<=0.2) ;
  PICP = fraction of events whose RUL window contains the true crossing.
Outputs: results/f1_rul_events.csv, results/f1_rul_summary.csv, outputs/figures/f1_rul_alpha_lambda.png
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
CENSOR = 300.0; M = 3; ALPHA = 0.20; THRESHOLDS = [120.0, 150.0, 175.0, 200.0]


def true_crossing(o, v, vb_c):
    """First order at which the measured trajectory reaches vb_c, linearly interpolated between the two
    bracketing measured points. Returns None if never reached within the record."""
    for i in range(1, len(o)):
        if v[i-1] < vb_c <= v[i]:
            frac = (vb_c - v[i-1]) / (v[i] - v[i-1])
            return o[i-1] + frac * (o[i] - o[i-1])
    if v[0] >= vb_c:                 # already above at first point (born-failed) -> not validatable
        return 0.0
    return None


def pred_crossing(b, a, p, vb_c):
    if a <= 0 or vb_c <= b:
        return None
    return ((vb_c - b) / a) ** (1.0 / p)


def cq(arr, al=0.10):
    arr = np.sort(arr); k = int(np.ceil((len(arr)+1)*(1-al))); return float(arr[min(k, len(arr))-1])


def fleet_residuals_by_horizon(d):
    """LOTO residuals + horizon of the few-shot model, pooled across tools, for the Mondrian quantile."""
    R = {}
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        fut = np.arange(M, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr); a, b = theil_sen(o[:M]**p, v[:M])
        r = np.abs(b + a*o[fut]**p - v[fut]); h = (fut - (M-1)).astype(float)
        R[tt] = (r, h)
    return R


def mondrian_q(cal_r, cal_h, hh, al=0.10):
    sel = (cal_h <= 1) if hh <= 1 else ((cal_h >= 2) & (cal_h <= 3) if hh <= 3 else cal_h >= 4)
    return cq(cal_r[sel], al) if sel.sum() >= 5 else cq(cal_r, al)


def main():
    d = load(); R = fleet_residuals_by_horizon(d)
    events = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        p = fit_global_p(tr); a, b = theil_sen(o[:M]**p, v[:M])
        t_ref = o[M-1]
        cal_r = np.concatenate([R[t][0] for t in R if t != tt])
        cal_h = np.concatenate([R[t][1] for t in R if t != tt])
        for vb_c in THRESHOLDS:
            tc = true_crossing(o, v, vb_c)
            if tc is None or tc <= t_ref:          # must cross AFTER the few-shot window, within record
                continue
            thc = pred_crossing(b, a, p, vb_c)
            if thc is None:
                continue
            rul_true = tc - t_ref; rul_pred = thc - t_ref
            if rul_true <= 0:
                continue
            hh = max(thc - t_ref, 1.0)
            q = mondrian_q(cal_r, cal_h, hh)
            slope = max(a * p * max(thc, 1e-6) ** (p - 1), 1e-6)   # dVB/d(order) at predicted crossing
            w = q / slope                                          # band (um) -> order units
            lo, hi = thc - w, thc + w
            events.append(dict(tool=tt, vb_c=vb_c, t_ref=round(t_ref,2), t_true=round(tc,2),
                               t_pred=round(thc,2), RUL_true=round(rul_true,2), RUL_pred=round(rul_pred,2),
                               abs_err=round(abs(rul_pred-rul_true),2),
                               RA=round(1-abs(rul_pred-rul_true)/rul_true,3),
                               alpha_hit=int(abs(rul_pred-rul_true)/rul_true <= ALPHA),
                               in_window=int(lo <= tc <= hi), win_lo=round(lo,2), win_hi=round(hi,2)))
    ev = pd.DataFrame(events)
    ev.to_csv(os.path.join(ROOT, "results", "f1_rul_events.csv"), index=False)

    print(f"Validatable RUL events: {len(ev)} (vs 3 with a single 200 um threshold)\n")
    print(f"{'VB_c':>6} {'events':>7} {'CRA':>7} {'alpha-acc':>10} {'window PICP':>12} {'mean|err|':>10}")
    rows = []
    for vb_c in THRESHOLDS:
        s = ev[ev.vb_c == vb_c]
        if len(s) == 0:
            continue
        print(f"{vb_c:>6.0f} {len(s):>7} {s.RA.mean():7.2f} {s.alpha_hit.mean()*100:9.0f}% "
              f"{s.in_window.mean()*100:11.0f}% {s.abs_err.mean():9.1f}")
        rows.append(dict(vb_c=vb_c, events=len(s), CRA=round(s.RA.mean(),3),
                         alpha_acc=round(s.alpha_hit.mean(),3), window_PICP=round(s.in_window.mean(),3),
                         mean_abs_err=round(s.abs_err.mean(),2)))
    print("-"*55)
    print(f"{'ALL':>6} {len(ev):>7} {ev.RA.mean():7.2f} {ev.alpha_hit.mean()*100:9.0f}% "
          f"{ev.in_window.mean()*100:11.0f}% {ev.abs_err.mean():9.1f}")
    rows.append(dict(vb_c="ALL", events=len(ev), CRA=round(ev.RA.mean(),3),
                     alpha_acc=round(ev.alpha_hit.mean(),3), window_PICP=round(ev.in_window.mean(),3),
                     mean_abs_err=round(ev.abs_err.mean(),2)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "f1_rul_summary.csv"), index=False)
    n_tools = ev.tool.nunique()
    print(f"\nCoverage of the design: {n_tools}/18 tools now contribute >=1 validatable RUL event.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cmap = {120:"#1f5fa8",150:"#2ca02c",175:"#d08a00",200:"#b03030"}
        s = ev.sort_values(["vb_c","t_ref"]).reset_index(drop=True)
        y = np.arange(len(s))
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        # calibrated RUL windows (horizontal bars) with the true crossing marked
        for i, row in s.iterrows():
            lo = row.win_lo - row.t_ref; hi = row.win_hi - row.t_ref
            col = cmap[int(row.vb_c)]
            ax.plot([lo, hi], [i, i], color=col, lw=3, alpha=.55, solid_capstyle="round")
            ax.plot(row.RUL_pred, i, "o", color=col, ms=5, zorder=3)
            covered = row.win_lo <= row.t_true <= row.win_hi
            ax.plot(row.RUL_true, i, "x" if covered else "D", color="k" if covered else "#c00",
                    ms=7 if covered else 6, mew=1.8, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([f"{r.tool}·{int(r.vb_c)}µm" for _, r in s.iterrows()],
                                             fontsize=9)
        ax.set_xlabel("RUL (cuts from few-shot window)", fontsize=10)
        picp = ev.in_window.mean()*100
        ax.set_title(f"Calibrated RUL windows: {picp:.0f}% of {len(ev)} events contain the truth\n"
                     "(bar = 90% window · o = predicted · × = covered · ◇ = missed)", fontsize=12)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=cmap[int(t)],
                  label=f'VB_c={int(t)}µm', ms=7) for t in THRESHOLDS], fontsize=9.5, loc="lower right")
        ax.grid(alpha=.25, axis="x"); fig.tight_layout()
        fig.savefig(os.path.join(ROOT,"outputs","figures","f1_rul_windows.png"), dpi=220); plt.close(fig)
        print("wrote results/f1_rul_events.csv, results/f1_rul_summary.csv, "
              "outputs/figures/f1_rul_windows.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
