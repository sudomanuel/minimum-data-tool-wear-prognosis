"""
run_kalman_test.py — SEPARATE test of a Kalman filter (state-space, constant-velocity in tau=order^p)
vs the current model. Two settings:
  (A) OFFLINE forecast from the first m=3 points (same protocol as current) -> MAE/R2/band.
  (B) ONLINE one-step-ahead (predict point k+1 after seeing 0..k) -> where a filter is supposed to shine.

State x=[vb, drift]; transition over d.tau:  vb' = vb + drift*d.tau, drift' = drift.
Params from TRAINING tools only (leakage-safe): R = measurement-noise var; Q = white-noise-accel; init
drift = population mean slope in tau-space. Pre-stated rule: ADOPT for the few-shot task only if (A) beats
current MAE 11.6 with a calibrated tighter band; otherwise REJECT (note any online benefit separately).
Output: results/kalman_test.csv.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
CENSOR = 300.0; M = 3; Z90 = 1.6448536


def load():
    f = pd.read_csv(os.path.join(ROOT, "data", "input", "derived", "features_experiment.csv"))
    d = f[["tool_id", "within_tool_order", "vb_um"]].drop_duplicates()
    return (d.rename(columns={"within_tool_order": "order", "vb_um": "vb"})
            .sort_values(["tool_id", "order"]).reset_index(drop=True))


def fit_global_p(tr):
    best_p, best = 0.5, np.inf
    for p in np.arange(0.2, 1.001, 0.05):
        tot = 0.0
        for _, g in tr.groupby("tool_id"):
            gg = g[g.vb <= CENSOR]
            if len(gg) < 2:
                continue
            A = np.column_stack([np.ones(len(gg)), gg.order.to_numpy(float) ** p])
            c, *_ = np.linalg.lstsq(A, gg.vb.to_numpy(float), rcond=None)
            tot += float(np.sum((A @ c - gg.vb.to_numpy(float)) ** 2))
        if tot < best:
            best, best_p = tot, p
    return best_p


def train_params(tr, p):
    """R (meas noise var), pop_drift, drift_var, sigma_a2 (process noise intensity) from training tools."""
    slopes, resid = [], []
    for _, g in tr.groupby("tool_id"):
        gg = g[g.vb <= CENSOR].sort_values("order")
        if len(gg) < 2:
            continue
        tau = gg.order.to_numpy(float) ** p; v = gg.vb.to_numpy(float)
        A = np.column_stack([np.ones(len(tau)), tau]); c, *_ = np.linalg.lstsq(A, v, rcond=None)
        slopes.append(c[1]); resid += list(v - A @ c)
    R = max(float(np.var(resid)), 1.0)
    pop_drift = float(np.mean(slopes)); drift_var = max(float(np.var(slopes, ddof=1)), 1e-6)
    sigma_a2 = drift_var                       # data-driven process-noise intensity (not tuned on test)
    return R, pop_drift, drift_var, sigma_a2


def kf_filter(tau, v, R, pop_drift, drift_var, sigma_a2, n_obs):
    """Filter the first n_obs points; return state x, cov P after the last observed point."""
    x = np.array([v[0], pop_drift]); P = np.array([[R, 0.0], [0.0, drift_var]])
    H = np.array([[1.0, 0.0]])
    for k in range(1, n_obs):
        dt = tau[k] - tau[k - 1]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = sigma_a2 * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x = F @ x; P = F @ P @ F.T + Q
        S = (H @ P @ H.T)[0, 0] + R; K = (P @ H.T / S).ravel()
        x = x + K * (v[k] - (H @ x)[0]); P = (np.eye(2) - np.outer(K, H)) @ P
    return x, P


def main():
    d = load()
    tools = sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))

    # ---- (A) OFFLINE forecast from first m ----
    P_, Y_, LO_, HI_, perr = [], [], [], [], []
    # ---- (B) ONLINE one-step-ahead ----
    on_err = []
    for tt in tools:
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= M:
            continue
        p = fit_global_p(tr); R, pod, dv, sa2 = train_params(tr, p); tau = o ** p
        fut = np.arange(M, len(o)); fut = fut[v[fut] <= CENSOR]
        if len(fut):
            x, Pc = kf_filter(tau, v, R, pod, dv, sa2, M)           # filter first m
            H = np.array([[1.0, 0.0]])
            for j in fut:                                            # predict-only to each future order
                dt = tau[j] - tau[M - 1]
                F = np.array([[1.0, dt], [0.0, 1.0]]); Q = sa2 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
                xp = F @ x; Pp = F @ Pc @ F.T + Q
                pred = xp[0]; var = (H @ Pp @ H.T)[0, 0] + R; sd = np.sqrt(max(var, 0))
                P_.append(pred); Y_.append(v[j]); LO_.append(pred - Z90 * sd); HI_.append(pred + Z90 * sd)
            perr.append(float(np.mean([abs(P_[-len(fut) + i] - Y_[-len(fut) + i]) for i in range(len(fut))])))
        # online one-step-ahead (wear regime), k>=1
        for k in range(1, len(o) - 1):
            if v[k + 1] > CENSOR:
                break
            x, Pc = kf_filter(tau, v, R, pod, dv, sa2, k + 1)
            dt = tau[k + 1] - tau[k]; pred = x[0] + x[1] * dt
            on_err.append(abs(pred - v[k + 1]))

    P_, Y_, LO_, HI_ = map(np.array, (P_, Y_, LO_, HI_))
    a_mae = float(np.mean(perr)); a_r2 = 1 - np.sum((Y_ - P_) ** 2) / np.sum((Y_ - Y_.mean()) ** 2)
    a_cov = float(np.mean((Y_ >= LO_) & (Y_ <= HI_))) * 100; a_w = float(np.mean(HI_ - LO_))
    b_mae = float(np.mean(on_err))

    rows = [dict(setting="CURRENT (physics+few-shot, offline m=3)", MAE_um=11.6, cov_pct=98, width_um=92),
            dict(setting="KALMAN offline (forecast from m=3)", MAE_um=round(a_mae, 1),
                 cov_pct=round(a_cov, 0), width_um=round(a_w, 0)),
            dict(setting="KALMAN online (1-step-ahead)", MAE_um=round(b_mae, 1), cov_pct="-", width_um="-")]
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "kalman_test.csv"), index=False)
    print(f"[validate] Kalman offline R2={a_r2:.2f} | n_offline_pts={len(Y_)} | n_online={len(on_err)}\n")
    print(f"{'setting':42} {'MAE':>6} {'cov%':>6} {'width':>7}")
    for r in rows:
        print(f"{r['setting']:42} {str(r['MAE_um']):>6} {str(r['cov_pct']):>6} {str(r['width_um']):>7}")

    better = a_mae <= 11.6 + 0.5 and a_cov >= 88 and a_w < 92 - 1
    print(f"\nVERDICT (few-shot task): " + ("ADOPT" if better else "REJECT — offline does not beat current.")
          + f"  | ONLINE one-step MAE={b_mae:.1f} um (for reference / deployment use).")


if __name__ == "__main__":
    main()
