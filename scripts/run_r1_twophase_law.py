"""run_r1_twophase_law.py — R1: two-phase wear law with state-triggered acceleration + envelope RUL.

Law: VB(t) = b + a·t^p                                  for t <= t_T (where VB(t_T) = VB_T)
     VB(t) = VB_T + r_T·(t−t_T) + 0.5·γ·(t−t_T)²        for t  > t_T
     r_T = a·p·t_T^(p−1)  (C¹ continuity),  γ = κ·r_T   (acceleration ∝ rate at transition)

Per-tool (few-shot): b, a — unchanged. Fleet/structural: p (LOTO-fitted), VB_T, κ (declared prior,
sensitivity grid — NOT identifiable in-record because every tool is right-censored below tertiary).

Validations (double):
 1) Within-record LOTO MAE at m=3/4 for κ ∈ {0, 0.005, 0.01, 0.02} × VB_T ∈ {150, 175}:
    does any κ>0 improve or at least not damage in-record accuracy? (κ=0 = current law.)
 2) RUL envelope at VB_fail=200: per-tool spread of predicted crossing across the κ-grid
    (envelope width = structural sensitivity; decision RUL = envelope minimum = safe side).
Outputs: results/r1_twophase_mae.csv, results/r1_rul_envelope.csv, outputs/figures/r1_envelope.png
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from run_mcurve import load, fit_global_p, theil_sen, tools_of
CENSOR = 300.0; VB_FAIL = 200.0
KAPPAS = [0.0, 0.005, 0.01, 0.02]
VBTS = [150.0, 175.0]


def two_phase(b, a, p, vbt, kappa, t):
    """Vectorized two-phase VB(t). Falls back to pure power law if the curve never reaches vbt.
    t_T is floored at 1 (the first cut): the power law's slope is singular as t->0, and the tertiary
    acceleration must scale with the tool's EMPIRICAL rate, not the analytic slope at the origin —
    otherwise born-worn tools (b just below VB_T) get a divergent r_T."""
    t = np.asarray(t, float)
    vb_pow = b + a * np.power(np.maximum(t, 1e-9), p)
    if kappa <= 0 or a <= 0 or vbt <= b:
        return vb_pow
    t_T = max(((vbt - b) / a) ** (1.0 / p), 1.0)
    vbT_eff = b + a * t_T ** p                       # actual level at the (floored) transition time
    r_T = a * p * t_T ** (p - 1.0)                   # now bounded by a·p
    gam = kappa * r_T
    dt = t - t_T
    vb_acc = vbT_eff + r_T * dt + 0.5 * gam * dt * dt
    return np.where(t <= t_T, vb_pow, vb_acc)


def crossing(b, a, p, vbt, kappa, vb_c, t_max=200.0):
    """First t with VB(t) = vb_c (dense grid; monotone law so unique)."""
    ts = np.linspace(0.5, t_max, 4000)
    vb = two_phase(b, a, p, vbt, kappa, ts)
    idx = np.argmax(vb >= vb_c)
    if vb[idx] < vb_c:
        return np.nan
    return float(ts[idx])


def main():
    d = load()
    print("R1 two-phase wear law — state-triggered acceleration, envelope RUL.\n")

    # ---------- (1) within-record LOTO MAE across the (kappa, VB_T) grid ----------
    rows = []
    for m in (3, 4):
        for vbt in VBTS:
            maes = {}
            for kap in KAPPAS:
                per = []
                for tt in tools_of(d):
                    tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
                    o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
                    if len(o) <= m:
                        continue
                    fut = np.arange(m, len(o)); fut = fut[v[fut] <= CENSOR]
                    if len(fut) == 0:
                        continue
                    p = fit_global_p(tr)
                    a, b = theil_sen(o[:m] ** p, v[:m])
                    pred = two_phase(b, a, p, vbt, kap, o[fut])
                    per.append(np.abs(pred - v[fut]).mean())
                maes[kap] = float(np.mean(per))
            rows.append(dict(m=m, VB_T=vbt, **{f"MAE_k{k}": round(maes[k], 2) for k in KAPPAS}))
            print(f"  m={m} VB_T={vbt:.0f}: " + "  ".join(f"κ={k}: {maes[k]:.2f}" for k in KAPPAS))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results", "r1_twophase_mae.csv"), index=False)
    base3 = df[(df.m == 3) & (df.VB_T == 150)].iloc[0]["MAE_k0.0"]
    print(f"\n  In-record verdict: κ=0 MAE (m=3) = {base3:.2f}; deviations under κ>0 are the structural "
          f"sensitivity — acceleration is NOT in-record identifiable (as declared).")

    # ---------- (2) RUL envelope at VB_fail=200 (m=3 few-shot) ----------
    m = 3; vbt = 150.0
    env = []
    for tt in tools_of(d):
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        if len(o) <= m or v[0] >= VB_FAIL:
            continue
        p = fit_global_p(tr); a, b = theil_sen(o[:m] ** p, v[:m])
        t_ref = o[m - 1]
        tcs = {kap: crossing(b, a, p, vbt, kap, VB_FAIL) for kap in KAPPAS}
        if any(np.isnan(list(tcs.values()))):
            continue
        ruls = {kap: max(tcs[kap] - t_ref, 0.0) for kap in KAPPAS}
        env.append(dict(tool=tt, t_ref=round(t_ref, 1),
                        **{f"RUL_k{k}": round(ruls[k], 1) for k in KAPPAS},
                        envelope_shrink_pct=round(100 * (1 - ruls[max(KAPPAS)] / max(ruls[0.0], 1e-9)), 1)))
    ev = pd.DataFrame(env); ev.to_csv(os.path.join(ROOT, "results", "r1_rul_envelope.csv"), index=False)
    print(f"\n  RUL envelope (VB_T=150, m=3): {len(ev)} tools with a predicted 200 µm crossing.")
    print(f"  Mean RUL: κ=0 {ev['RUL_k0.0'].mean():.1f} cuts  ->  κ=0.02 {ev['RUL_k0.02'].mean():.1f} cuts "
          f"(envelope minimum; mean shrink {ev.envelope_shrink_pct.mean():.0f}%)")
    print("  Decision rule: RUL_decision = envelope MINIMUM -> conservative by construction; κ=0 column "
          "is the published optimistic bound (current law).")

    # ---------- figure ----------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tt = ev.sort_values("RUL_k0.0", ascending=False).iloc[0]["tool"]
        tr = d[d.tool_id != tt]; g = d[d.tool_id == tt].sort_values("order")
        o, v = g.order.to_numpy(float), g.vb.to_numpy(float)
        p = fit_global_p(tr); a, b = theil_sen(o[:3] ** p, v[:3])
        ts = np.linspace(0.5, max(o.max() * 2.0, 30), 400)
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        cols = ["#1f5fa8", "#2ca02c", "#d08a00", "#b03030"]
        for kap, c in zip(KAPPAS, cols):
            ax.plot(ts, two_phase(b, a, p, 150.0, kap, ts), color=c, lw=2,
                    label=f"κ={kap}" + ("  (current law)" if kap == 0 else ""))
        ax.plot(o, v, "ko", ms=6, label="measured VB")
        ax.plot(o[:3], v[:3], "o", color="#1f5fa8", ms=9, mfc="none", mew=2, label="few-shot points (m=3)")
        ax.axhline(VB_FAIL, ls="--", color="k", lw=1)
        ax.text(ts[0] + 0.4, VB_FAIL + 6, "VB_fail = 200 µm", ha="left", fontsize=8)
        ax.axhline(150, ls=":", color="#888", lw=1)
        ax.text(ts[-1] * 0.98, 143, "VB_T = 150 µm (transition)", ha="right", va="top",
                fontsize=8, color="#666")
        ax.set_xlabel("cut order t"); ax.set_ylabel("VB (µm)")
        ax.set_ylim(max(min(v) - 40, 0), 320)
        ax.set_title(f"Two-phase law ({tt}): κ-envelope of trajectories and RUL", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "outputs", "figures", "r1_envelope.png"), dpi=220); plt.close(fig)
        print("\nwrote results/r1_twophase_mae.csv, results/r1_rul_envelope.csv, "
              "outputs/figures/r1_envelope.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
