#!/usr/bin/env python3
"""
run_pinn_softplus_rate.py — P8.4 PINN softplus-rate ablation on official VB.

Compares positivity-preserving wear-rate against the legacy (destructive) rate term, on the
reliability-aware minimal feature set. Reuses src/phm/pinn_softplus.py (new module; legacy
pinn.py untouched). NO synthetic data, gate closed, no paper changes.

Variants:
  P0 data-only      P1 +monotonicity      P2 +OLD rate (a+softplus(b)E, historical)
  P3 +SOFTPLUS rate      P4 +monotonicity +SOFTPLUS rate

Success criterion: NOT to beat Linear(t). Show the softplus-rate PINN has a physically more
coherent wear rate (positive by construction), fewer monotonicity/rate violations, less
artificial flattening, and better behaviour around exp77.

Outputs: results/pinn_softplus_rate_ablation.csv; reports/pinn_softplus_rate_report.md;
outputs/figures/pinn_softplus_vb_curves.png; outputs/figures/pinn_softplus_rate_curves.png.

Uso:  python run.py pinn-softplus   [--repo-root D:/KSF/PHM/phm_tool_wear] [--epochs 1200]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.pinn_softplus import SoftplusRatePINN  # noqa: E402

RESULTS = WT / "results"
FIGS = WT / "outputs" / "figures"
FEAT = WT / "data" / "features" / "p8_2_features_experiment_full_contact.csv"
# Reliability-aware minimal input set: robust amplitude/shape/spectral; NO raw energy as a strong
# input. Energy enters ONLY as a rate DRIVER (per-contact mean R_energy_mean, not energy_total).
X_FEATURES = ["A_rms_mean", "R_rms_mean", "A_waveform_length_mean", "R_waveform_length_mean",
              "R_dominant_freq_mean", "R_wavelet_entropy_mean"]
DRIVERS = ["R_energy_mean", "R_rms_mean"]   # [E_drv, RMS_drv] for the softplus rate prior
T_COL = "physical_experiment_order"
REF = {"Linear(t)_official": 3.10, "Poly2(t)_official": 3.83, "PINN_mono_P8.1": 6.65}

# lambda_rate=0.1 matches the legacy scale (1.0 lets the rate term dominate destructively).
VARIANTS = {
    "P0_data_only":      dict(lambda_mono=0.0, lambda_rate=0.0, rate_form="none"),
    "P1_mono":           dict(lambda_mono=1.0, lambda_rate=0.0, rate_form="none"),
    "P2_old_rate":       dict(lambda_mono=0.0, lambda_rate=0.1, rate_form="old"),
    "P3_softplus_rate":  dict(lambda_mono=0.0, lambda_rate=0.1, rate_form="softplus"),
    "P4_mono_softplus":  dict(lambda_mono=1.0, lambda_rate=0.1, rate_form="softplus"),
}


def metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else np.nan
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)
    return mae, rmse, r2, mape


def loeo(df, cfg, epochs, seed):
    X = df[X_FEATURES].to_numpy(float)
    t = df[T_COL].to_numpy(float)
    y = df["VB_um"].to_numpy(float)
    drv = df[DRIVERS].to_numpy(float)
    yp = np.zeros(len(df))
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        m = SoftplusRatePINN(epochs=epochs, random_state=seed, **cfg)
        m.fit(X[tr], t[tr], y[tr], drv[tr])
        yp[i] = m.predict(X[i:i + 1], t[i:i + 1])[0]
    return y, yp


def coherence(df, yp):
    order = df[T_COL].to_numpy(float)
    idx = np.argsort(order)
    seq = np.asarray(yp)[idx]
    diffs = np.diff(seq)
    e77 = int(np.where(df.experiment_id == 77)[0][0])
    true = df["VB_um"].to_numpy(float)
    # flattening: predicted last increment vs real last increment (recorded order)
    real_last = true[idx][-1] - true[idx][-2]
    pred_last = seq[-1] - seq[-2]
    return dict(
        mono_violations=int(np.sum(diffs < -1e-6)),
        negative_rate_fraction=round(float(np.mean(diffs < 0)), 3),
        exp77_residual=round(float(yp[e77] - true[e77]), 2),
        final_point_residual=round(float(seq[-1] - true[idx][-1]), 2),
        end_slope_ratio=round(float(pred_last / (real_last + 1e-9)), 3),  # <1 = flattening
        rul_extrapolated=bool(np.max(yp) < 300.0),
    )


def insample_rate_coherence(df, cfg, epochs, seed=42):
    """Full-data fit (diagnostic) -> is the LEARNED wear rate physically positive?
    Returns fraction of a dense order-grid where dVB/dt < 0, and the minimum rate."""
    X = df[X_FEATURES].to_numpy(float); t = df[T_COL].to_numpy(float)
    y = df["VB_um"].to_numpy(float); drv = df[DRIVERS].to_numpy(float)
    m = SoftplusRatePINN(epochs=epochs, random_state=seed, **cfg).fit(X, t, y, drv)
    grid = np.linspace(t.min(), t.max(), 60)
    Xg = np.vstack([np.interp(grid, t, X[:, j]) for j in range(X.shape[1])]).T
    rate = m.wear_rate_physical(Xg, grid)
    return round(float(np.mean(rate < 0)), 3), round(float(rate.min()), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("D:/KSF/PHM/phm_tool_wear"))
    ap.add_argument("--epochs", type=int, default=1500)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(FEAT).sort_values(T_COL).reset_index(drop=True)
    print(f"features={X_FEATURES}\nrate drivers={DRIVERS}  t={T_COL}\nVB official "
          f"{df.VB_um.min():.0f}-{df.VB_um.max():.0f} (n={len(df)})\n")

    rows, oof = [], {}
    for name, cfg in VARIANTS.items():
        y, yp = loeo(df, cfg, args.epochs, seed=42)
        mae, rmse, r2, mape = metrics(y, yp)
        coh = coherence(df, yp)
        ins_neg, ins_min = insample_rate_coherence(df, cfg, args.epochs)
        rows.append(dict(variant=name, rate_form=cfg["rate_form"],
                         lambda_mono=cfg["lambda_mono"], lambda_rate=cfg["lambda_rate"],
                         MAE=round(mae, 3), RMSE=round(rmse, 3), R2=round(r2, 3),
                         MAPE=round(mape, 2), **coh,
                         insample_rate_neg_fraction=ins_neg, insample_rate_min=ins_min))
        oof[name] = yp
        print(f"  {name:18s} MAE={mae:6.2f} R2={r2:6.2f} OOFmono={coh['mono_violations']} "
              f"exp77={coh['exp77_residual']:+.1f} flatten={coh['end_slope_ratio']:.2f} "
              f"| in-sample rate<0: {ins_neg:.0%} (min {ins_min:+.1f})", flush=True)

    # training stability across seeds for the softplus variants
    print("\nstability across seeds [42,123,7] (P3, P4):")
    for name in ("P3_softplus_rate", "P4_mono_softplus"):
        maes, monos = [], []
        for s in (42, 123, 7):
            y, yp = loeo(df, VARIANTS[name], args.epochs, seed=s)
            maes.append(metrics(y, yp)[0])
            monos.append(coherence(df, yp)["mono_violations"])
        r = next(x for x in rows if x["variant"] == name)
        r["MAE_seed_mean"] = round(float(np.mean(maes)), 3)
        r["MAE_seed_std"] = round(float(np.std(maes)), 3)
        r["mono_seed_std"] = round(float(np.std(monos)), 2)
        print(f"  {name:18s} MAE {np.mean(maes):.2f}±{np.std(maes):.2f} mono_std={np.std(monos):.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "pinn_softplus_rate_ablation.csv", index=False)

    _fig_vb(df, oof)
    _fig_rates(df, args.epochs)
    print(f"\nDONE in {time.time()-t0:.0f}s -> results/pinn_softplus_rate_ablation.csv + 2 figures")
    print(f"refs: Linear(t)={REF['Linear(t)_official']} PINN_mono(P8.1)={REF['PINN_mono_P8.1']}")
    return res


def _fig_vb(df, oof):
    order = df[T_COL].to_numpy(float)
    idx = np.argsort(order)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=140)
    ax.plot(order[idx], df["VB_um"].to_numpy(float)[idx], "ko-", lw=2, label="measured VB (official)")
    colors = {"P0_data_only": "#888", "P1_mono": "#1F4E79", "P2_old_rate": "#8C2D2D",
              "P3_softplus_rate": "#2E6F62", "P4_mono_softplus": "#B3541E"}
    for name, yp in oof.items():
        ax.plot(order[idx], np.asarray(yp)[idx], "--", color=colors[name], label=name)
    ax.axvspan(5.5, 7.5, color="#cccccc", alpha=0.3)
    ax.text(6.5, ax.get_ylim()[0] + 5, "71-72\n(no signal)", fontsize=7, ha="center", color="#666")
    ax.set_xlabel("physical experiment order"); ax.set_ylabel("VB (µm)")
    ax.set_title("P8.4 — LOEO out-of-fold VB curves (softplus-rate ablation, official VB)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "pinn_softplus_vb_curves.png", bbox_inches="tight",
                                    facecolor="white"); plt.close(fig)


def _fig_rates(df, epochs):
    """Full-data fit per variant (diagnostic) -> learned wear-rate on a dense order grid."""
    X = df[X_FEATURES].to_numpy(float); t = df[T_COL].to_numpy(float)
    y = df["VB_um"].to_numpy(float); drv = df[DRIVERS].to_numpy(float)
    grid = np.linspace(t.min(), t.max(), 60)
    # hold features at their trajectory by nearest-order interpolation
    Xg = np.vstack([np.interp(grid, t, X[:, j]) for j in range(X.shape[1])]).T
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=140)
    colors = {"P1_mono": "#1F4E79", "P2_old_rate": "#8C2D2D",
              "P3_softplus_rate": "#2E6F62", "P4_mono_softplus": "#B3541E"}
    for name in ("P1_mono", "P2_old_rate", "P3_softplus_rate", "P4_mono_softplus"):
        m = SoftplusRatePINN(epochs=epochs, random_state=42, **VARIANTS[name]).fit(X, t, y, drv)
        rate = m.wear_rate_physical(Xg, grid)
        ax.plot(grid, rate, color=colors[name], label=name)
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_ylim(bottom=min(-0.5, ax.get_ylim()[0]))
    ax.fill_between(grid, ax.get_ylim()[0], 0, color="#8C2D2D", alpha=0.06)
    ax.text(grid[2], ax.get_ylim()[0] * 0.5, "negative wear rate\n(unphysical)", fontsize=8, color="#8C2D2D")
    ax.set_xlabel("physical experiment order"); ax.set_ylabel("dVB/d(order)  (µm/step)")
    ax.set_title("P8.4 — learned wear rate: softplus is smooth & stable-positive; "
                 "old rate is volatile\n(reliability-aware energy driver keeps all forms positive)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "pinn_softplus_rate_curves.png", bbox_inches="tight",
                                    facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
