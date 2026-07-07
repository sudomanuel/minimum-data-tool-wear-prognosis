"""
run_online_monitor.py — two deliverables:
  (1) ONLINE Kalman monitor (one-step-ahead next-cut prediction) — MAE + tracking figure.
  (2) NORMALIZED (horizon-adaptive) conformal band vs the current GLOBAL conformal band — does it tighten
      the CI at valid coverage? (the user's "reduce the IC" goal). ADOPT only if mean width drops with
      coverage >= 88%.
Leakage-safe LOTO, wear regime VB<=300, m=3. Outputs: results/online_monitor.csv,
results/normalized_conformal.csv, outputs/figures/kalman_online.png.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from phm.prognostic_system import conformal_quantile
CENSOR = 300.0; M = 3


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um"]].drop_duplicates()
    return (d.rename(columns={"within_tool_order": "order", "vb_um": "vb"})
            .sort_values(["tool_id", "order"]).reset_index(drop=True))


def fit_global_p(tr):
    bp, be = 0.5, np.inf
    for p in np.arange(0.2, 1.001, 0.05):
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR]
            if len(gg) < 2:
                continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < be:
            be, bp = tot, p
    return bp


def theil_sen(x, y):
    s = np.median([(y[j] - y[i]) / (x[j] - x[i])
                   for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]])
    return float(s), float(np.median(y - s * x))


def tr_params(tr, p):
    slopes, resid = [], []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) < 2:
            continue
        tau = gg.order.to_numpy(float) ** p; v = gg.vb.to_numpy(float)
        A = np.column_stack([np.ones(len(tau)), tau]); c, *_ = np.linalg.lstsq(A, v, rcond=None)
        slopes.append(c[1]); resid += list(v - A @ c)
    return max(np.var(resid), 1.0), float(np.mean(slopes)), max(np.var(slopes, ddof=1), 1e-6)


def kf_online_onestep(tr, o, v, p):
    """Online: after each observed point, predict the NEXT. Returns list of (pred, true)."""
    R, pod, dv = tr_params(tr, p); sa2 = dv; tau = o ** p
    H = np.array([[1.0, 0.0]]); preds = []
    x = np.array([v[0], pod]); P = np.array([[R, 0.0], [0.0, dv]])
    for k in range(1, len(o)):
        dt = tau[k] - tau[k - 1]
        F = np.array([[1.0, dt], [0.0, 1.0]]); Q = sa2 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        xp = F @ x; Pp = F @ P @ F.T + Q                       # predict next (one-step)
        if v[k] <= CENSOR:
            preds.append((float(xp[0]), float(v[k]), k))
        S = (H @ Pp @ H.T)[0, 0] + R; K = (Pp @ H.T / S).ravel()
        x = xp + K * (v[k] - (H @ xp)[0]); P = (np.eye(2) - np.outer(K, H)) @ Pp
    return preds


def main():
    d = load(); tools = sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))

    # ---------- (1) online Kalman one-step ----------
    on = []
    for tt in tools:
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= 2:
            continue
        p = fit_global_p(tr); on += kf_online_onestep(tr, o, v, p)
    on_mae = float(np.mean([abs(pr - tu) for pr, tu, _ in on]))
    pd.DataFrame([dict(task="Kalman online one-step-ahead (monitoring)", MAE_um=round(on_mae, 1),
                       n=len(on))]).to_csv(os.path.join(ROOT, "results", "online_monitor.csv"), index=False)
    print(f"(1) ONLINE Kalman one-step-ahead MAE = {on_mae:.1f} um  (n={len(on)})  — for live monitoring")

    # figure: tracking on one representative long tool (T14)
    tt = "T14"; tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
    o, v = g.order.to_numpy(float), g.vb.to_numpy(float); p = fit_global_p(tr)
    pr = kf_online_onestep(tr, o, v, p)
    ks = [k for _, _, k in pr]; preds = [a for a, _, _ in pr]; trues = [b for _, b, _ in pr]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(o[v <= CENSOR], v[v <= CENSOR], "o-", color="#5A6B7B", label="actual VB  ·  VB real")
    ax.plot(ks, preds, "s--", color="#2E8B57", label="Kalman next-cut prediction  ·  predicción próximo corte")
    ax.set_title("Online monitoring — next-cut prediction  ·  Monitoreo online — próximo corte", fontsize=12)
    ax.set_xlabel("cut order  ·  nº de corte"); ax.set_ylabel("flank wear VB (µm)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.text(0.5, 0.01, f"one-step-ahead MAE = {on_mae:.1f} µm across 18 tools", ha="center",
             fontsize=10, color="#2E8B57", weight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "kalman_online.png"), dpi=220); plt.close(fig)

    # ---------- (2) normalized (horizon-adaptive) conformal vs global ----------
    res, hor, sp = {}, {}, {}
    for tt in tools:
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        fut = np.arange(M, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut) == 0:
            continue
        p = fit_global_p(tr); x = o[:M] ** p; a, b = theil_sen(x, v[:M])
        pred = b + a * o[fut] ** p
        res[tt] = np.abs(pred - v[fut]); hor[tt] = (fut - (M - 1)).astype(float); sp[tt] = (pred, v[fut])

    def sigma(h):
        return np.sqrt(1.0 + h)                                    # difficulty proxy = sqrt(horizon)

    g_cov, g_w, n_cov, n_w = [], [], [], []
    for tt in res:
        cal_r = np.concatenate([res[t] for t in res if t != tt])
        cal_h = np.concatenate([hor[t] for t in res if t != tt])
        qg = conformal_quantile(cal_r, 0.1)                        # global
        qn = conformal_quantile(cal_r / sigma(cal_h), 0.1)         # normalized
        r = res[tt]; h = hor[tt]
        g_cov.append(np.mean(r <= qg)); g_w.append(np.mean(2 * qg * np.ones_like(r)))
        n_cov.append(np.mean(r <= qn * sigma(h))); n_w.append(np.mean(2 * qn * sigma(h)))
    rows = [dict(method="GLOBAL conformal (current)", coverage_pct=round(np.mean(g_cov) * 100, 0),
                 mean_width_um=round(np.mean(g_w), 0)),
            dict(method="NORMALIZED conformal (horizon-adaptive)", coverage_pct=round(np.mean(n_cov) * 100, 0),
                 mean_width_um=round(np.mean(n_w), 0))]
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "normalized_conformal.csv"), index=False)
    print("\n(2) Conformal band — global vs normalized (horizon-adaptive):")
    for r in rows:
        print(f"    {r['method']:42} coverage {r['coverage_pct']:.0f}%  | mean width {r['mean_width_um']:.0f} um")
    adopt = (np.mean(n_cov) * 100 >= 88) and (np.mean(n_w) < np.mean(g_w) - 1)
    print(f"\n    VERDICT (band): {'ADOPT normalized — tighter at valid coverage.' if adopt else 'keep global.'}")
    print("\nwrote results/online_monitor.csv, results/normalized_conformal.csv, outputs/figures/kalman_online.png")


if __name__ == "__main__":
    main()
