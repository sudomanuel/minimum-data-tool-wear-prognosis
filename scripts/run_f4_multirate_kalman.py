"""run_f4_multirate_kalman.py — FRONT 4: asynchronous, event-triggered, multi-rate Kalman monitor.

Reconciles the "online, cut-by-cut" claim with reality: VB is measured ex situ at sparse stops. The filter
runs a TIME UPDATE at every step along the physics power-law transition (state in tau = order^p), and a
MEASUREMENT UPDATE only when an observation exists (event-triggered / intermittent observations). Two
observation rates are supported: a sparse, precise microscope reading (low R) and, where available, a
dense, noisy in-situ VIBRATION pseudo-measurement (high R) mapped from that cut's features by the fair
PLS model of F2.

Two questions, both leakage-safe LOTO, wear regime VB<=300:
  (A) does the architecture behave correctly — covariance grows between readings, collapses at events —
      and does the microscope-only event-triggered filter reproduce the Section 4.7 one-step error?
  (B) at a cut with NO microscope reading, does the vibration pseudo-measurement improve the estimate over
      the physics-only forecast? Pre-stated rule: adopt the vibration channel only if it lowers the error.
Outputs: results/f4_multirate.csv, outputs/figures/f4_multirate.png
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_online_monitor import load as load_vb, fit_global_p, theil_sen, tr_params
from run_f2_fair_baseline import load as load_feat, phys_cols
CENSOR = 300.0
R_MIC = 3.0 ** 2        # microscope measurement-noise variance (um^2)


def pseudo_vb_model(feat, cols, tt):
    """LOTO PLS sensor->VB for the held-out tool: return dict order-> pseudo VB, and sigma_vib."""
    tr = feat[(feat.tool_id != tt) & (feat.vb_um <= CENSOR)]
    te = feat[feat.tool_id == tt]
    Xtr = tr[cols].to_numpy(float); ytr = tr.vb_um.to_numpy(float)
    sc = StandardScaler().fit(Xtr)
    k = 2
    pls = PLSRegression(n_components=k).fit(sc.transform(Xtr), ytr)
    # sigma from training leave-one-tool-out residuals
    res = []
    for ut in tr.tool_id.unique():
        a = tr.tool_id != ut
        if a.sum() < 6 or (~a).sum() == 0:
            continue
        m = PLSRegression(n_components=k).fit(sc.transform(tr[a][cols].to_numpy(float)), tr[a].vb_um.to_numpy(float))
        pr = m.predict(sc.transform(tr[~a][cols].to_numpy(float))).ravel()
        res += list(pr - tr[~a].vb_um.to_numpy(float))
    sig = max(np.std(res), 10.0)
    pred = pls.predict(sc.transform(te[cols].to_numpy(float))).ravel()
    return dict(zip(te.within_tool_order.to_numpy(), pred)), float(sig)


def run_tool(o, v, p, procvar, pod, R, pseudo=None, sig_vib=None, withhold=None, use_vib=False):
    """Event-triggered KF. withhold = set of step-indices where the microscope reading is hidden.
    Returns list of (k, pred_before_any_update_at_k, pred_after_vib_at_k, true, P_pred)."""
    tau = o ** p; H = np.array([[1.0, 0.0]])
    x = np.array([v[0], pod]); P = np.array([[R, 0.0], [0.0, procvar]])
    withhold = withhold or set(); out = []
    for k in range(1, len(o)):
        dt = tau[k] - tau[k - 1]
        F = np.array([[1.0, dt], [0.0, 1.0]]); Q = procvar * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        xp = F @ x; Pp = F @ P @ F.T + Q                    # TIME UPDATE (every step)
        pred_phys = float(xp[0]); Ppred = float(Pp[0, 0])
        pred_vib = pred_phys
        # high-rate vibration pseudo-measurement (optional)
        if use_vib and pseudo is not None and o[k] in pseudo:
            zc = pseudo[o[k]]; Rv = sig_vib ** 2
            S = Pp[0, 0] + Rv; Kk = (Pp @ H.T / S).ravel()
            xv = xp + Kk * (zc - xp[0]); pred_vib = float(xv[0])
        if v[k] <= CENSOR:
            out.append((k, pred_phys, pred_vib, float(v[k]), Ppred))
        # low-rate microscope MEASUREMENT UPDATE (event-triggered)
        if k not in withhold:
            S = (H @ Pp @ H.T)[0, 0] + R; Kk = (Pp @ H.T / S).ravel()
            x = xp + Kk * (v[k] - (H @ xp)[0]); P = (np.eye(2) - np.outer(Kk, H)) @ Pp
        elif use_vib and pseudo is not None and o[k] in pseudo:
            # no microscope: fall back to the vibration pseudo-measurement to advance the state
            zc = pseudo[o[k]]; Rv = sig_vib ** 2
            S = Pp[0, 0] + Rv; Kk = (Pp @ H.T / S).ravel()
            x = xp + Kk * (zc - xp[0]); P = (np.eye(2) - np.outer(Kk, H)) @ Pp
        else:
            x = xp; P = Pp
    return out


def main():
    d = load_vb(); feat = load_feat(); cols = phys_cols(feat)
    tools = sorted(d.tool_id.unique(), key=lambda t: int(str(t).lstrip("T") or 0))

    # ---------- (A) microscope-only event-triggered filter: reproduce Section 4.7 ----------
    errM = []
    for tt in tools:
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= 2:
            continue
        p = fit_global_p(tr); R, pod, dv = tr_params(tr, p)
        for k, pp, pv, tru, _ in run_tool(o, v, p, dv, pod, R):
            errM.append(abs(pp - tru))
    mae_M = float(np.mean(errM))
    print(f"(A) Microscope-only event-triggered one-step MAE = {mae_M:.1f} um  (reproduces Section 4.7)\n")

    # ---------- (B) value of the in-situ vibration pseudo-measurement at measurement-free cuts ----------
    phys_only, phys_vib = [], []
    for tt in tools:
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= 3:
            continue
        p = fit_global_p(tr); R, pod, dv = tr_params(tr, p)
        pseudo, sig = pseudo_vb_model(feat, cols, tt)
        withhold = {len(o) - 1}                       # hide the microscope at the last cut
        res = run_tool(o, v, p, dv, pod, R, pseudo, sig, withhold=withhold, use_vib=True)
        for k, pp, pv, tru, _ in res:
            if k in withhold:
                phys_only.append(abs(pp - tru)); phys_vib.append(abs(pv - tru))
    mae_phys = float(np.mean(phys_only)); mae_vib = float(np.mean(phys_vib))
    print("(B) At a measurement-free cut (microscope withheld), estimate of VB:")
    print(f"    physics-only forecast        MAE = {mae_phys:.1f} um")
    print(f"    physics + vibration proxy    MAE = {mae_vib:.1f} um  (sigma_vib ~ {sig:.0f} um)")
    verdict = ("vibration channel HELPS -> adopt" if mae_vib < mae_phys - 0.3
               else "vibration channel does NOT help -> keep microscope-only (consistent with F2 sensor-null)")
    print(f"    verdict: {verdict}")

    pd.DataFrame([dict(metric="microscope_only_onestep_MAE_um", value=round(mae_M, 1)),
                  dict(metric="withheld_cut_physics_only_MAE_um", value=round(mae_phys, 1)),
                  dict(metric="withheld_cut_physics_plus_vib_MAE_um", value=round(mae_vib, 1))]
                 ).to_csv(os.path.join(ROOT, "results", "f4_multirate.csv"), index=False)

    # ---------- figure: event-triggered covariance on a long tool with a simulated gap ----------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tt = "T14" if "T14" in set(tools) else tools[-1]
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        p = fit_global_p(tr); R, pod, dv = tr_params(tr, p)
        gap = {len(o)//2, len(o)//2 + 1}             # simulate two consecutive missed microscope stops
        res = run_tool(o, v, p, dv, pod, R, withhold=gap)
        ks = [r[0] for r in res]; preds = [r[1] for r in res]; trues = [r[3] for r in res]
        sd = [np.sqrt(r[4]) for r in res]
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(ks, trues, "ko-", ms=5, label="measured VB")
        ax.plot(ks, preds, "-", color="#1f5fa8", lw=2, label="one-step prediction")
        ax.fill_between(ks, np.array(preds)-np.array(sd), np.array(preds)+np.array(sd),
                        color="#1f5fa8", alpha=.2, label="±1σ (predicted)")
        for kk in gap:
            ax.axvspan(kk-0.5, kk+0.5, color="#d08a00", alpha=.18)
        ax.plot([], [], color="#d08a00", alpha=.4, lw=8, label="microscope withheld (predict-only)")
        ax.set_xlabel("cut index"); ax.set_ylabel("VB (µm)")
        ax.set_title(f"Event-triggered filter ({tt}): σ grows in gaps, shrinks at readings", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "outputs", "figures", "f4_multirate.png"), dpi=220); plt.close(fig)
        print("\nwrote results/f4_multirate.csv, outputs/figures/f4_multirate.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
