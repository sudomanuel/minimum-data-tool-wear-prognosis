#!/usr/bin/env python3
"""
run_hi_rul_threshold_sweep.py — #2 Health Index + RUL threshold sweep.

Consumes predicted VB curves from Linear(t), Poly2(t), best classical (reliability-aware
ElasticNet), PINN_mono, PINN_softplus_rate, PINN_mono+softplus, and derives HI/DI/t_failure/RUL
by threshold crossing for VB_failure in {220, 250, 300, 600} µm (default 300). Marks
rul_extrapolated=true honestly (official VB_max=212 < every threshold -> ALL extrapolated).

Does NOT improve models; converts VB predictions into an auditable PHM output. No synthetic data,
gate closed, paper untouched, legacy P4 (derive_rul.py / physics.yaml) untouched.

Outputs: results/hi_rul_threshold_summary.csv, results/hi_rul_curves.csv,
reports/hi_rul_threshold_report.md, outputs/figures/hi_curves_by_model.png,
outputs/figures/rul_threshold_crossing.png.

Uso:  python run.py hi-rul-sweep   [--repo-root <repo-root>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "src"))
from phm.rul import health_index, degradation_index, derive_rul  # noqa: E402
from phm.pinn_softplus import SoftplusRatePINN  # noqa: E402
from sklearn.linear_model import ElasticNet  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RESULTS = WT / "results"
FIGS = WT / "outputs" / "figures"
FEAT = WT / "data" / "features" / "p8_2_features_experiment_full_contact.csv"
X_FEATURES = ["A_rms_mean", "R_rms_mean", "A_waveform_length_mean", "R_waveform_length_mean",
              "R_dominant_freq_mean", "R_wavelet_entropy_mean"]
DRIVERS = ["R_energy_mean", "R_rms_mean"]
T_COL = "physical_experiment_order"
THRESHOLDS = [220, 250, 300, 600]
DEFAULT_THR = 300
VB_0 = 103.0          # official initial wear (config/rul_thresholds.yaml: vb_initial_um)
T_MAX = 24.0
PINN_EPOCHS = 1500


def feature_grid(df, grid):
    t = df[T_COL].to_numpy(float)
    cols = []
    for c in X_FEATURES:
        v = df[c].to_numpy(float)
        # interpolate within observed range; hold last value beyond it (declared policy)
        cols.append(np.interp(grid, t, v, left=v[0], right=v[-1]))
    return np.vstack(cols).T


def driver_grid(df, grid):
    t = df[T_COL].to_numpy(float)
    return np.vstack([np.interp(grid, t, df[c].to_numpy(float), right=df[c].to_numpy(float)[-1])
                      for c in DRIVERS]).T


def build_curves(df, grid):
    """VB_hat(grid) for each model. Returns dict name -> curve array."""
    t = df[T_COL].to_numpy(float)
    y = df["VB_um"].to_numpy(float)
    curves = {}
    curves["Linear(t)"] = np.polyval(np.polyfit(t, y, 1), grid)
    curves["Poly2(t)"] = np.polyval(np.polyfit(t, y, 2), grid)

    # best classical (x-only, reliability-aware ElasticNet) — full-data fit; hold-last features
    Xtr = df[X_FEATURES].to_numpy(float)
    sc = StandardScaler().fit(Xtr)
    en = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=50000).fit(sc.transform(Xtr), y)
    Xg = feature_grid(df, grid)
    curves["Classical_best(EN)"] = en.predict(sc.transform(Xg))

    # PINN variants (full-data fits, diagnostic curves)
    drv = df[DRIVERS].to_numpy(float)
    dg = driver_grid(df, grid)
    pinn_cfgs = {
        "PINN_mono": dict(lambda_mono=1.0, lambda_rate=0.0, rate_form="none"),
        "PINN_softplus_rate": dict(lambda_mono=0.0, lambda_rate=0.1, rate_form="softplus"),
        "PINN_mono+softplus": dict(lambda_mono=1.0, lambda_rate=0.1, rate_form="softplus"),
    }
    Xg_feat = feature_grid(df, grid)
    for name, cfg in pinn_cfgs.items():
        m = SoftplusRatePINN(epochs=PINN_EPOCHS, random_state=42, **cfg).fit(Xtr, t, y, drv)
        # SoftplusRatePINN.predict ignores drivers; pass features + grid t
        curves[name] = m.predict(Xg_feat, grid)
    return curves


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEAT).sort_values(T_COL).reset_index(drop=True)
    t_obs = df[T_COL].to_numpy(float)
    vb_obs = df["VB_um"].to_numpy(float)
    t_last, vb_max = float(t_obs.max()), float(vb_obs.max())
    grid = np.arange(float(t_obs.min()), T_MAX + 0.05, 0.05)
    print(f"official VB {vb_obs.min():.0f}-{vb_max:.0f} (max < every threshold -> all RUL extrapolated)")
    print(f"VB_0={VB_0}  thresholds={THRESHOLDS} default={DEFAULT_THR}\n")

    curves = build_curves(df, grid)

    # ---- threshold sweep ----
    rows = []
    for name, curve in curves.items():
        cmax = float(np.max(curve[grid <= t_last]))  # max within observed range
        for thr in THRESHOLDS:
            r = derive_rul(grid, curve, thr, VB_0, t_last, vb_max)
            r.update(model=name, model_vb_max_in_range=round(cmax, 1))
            rows.append(r)
    summ = pd.DataFrame(rows)[["model", "vb_failure_um", "vb_0_um", "vb_max_observed",
                               "model_vb_max_in_range", "t_failure", "rul_at_last_obs",
                               "crossing_within_horizon", "rul_extrapolated"]]
    summ.to_csv(RESULTS / "hi_rul_threshold_summary.csv", index=False)

    # ---- HI/DI curves at default threshold ----
    crow = []
    for name, curve in curves.items():
        hi = health_index(curve, DEFAULT_THR, VB_0)
        di = degradation_index(curve, DEFAULT_THR, VB_0)
        for g, vb, h, d in zip(grid, curve, hi, di):
            crow.append(dict(model=name, t=round(float(g), 3), VB_hat=round(float(vb), 3),
                             HI=round(float(h), 4), DI=round(float(d), 4),
                             threshold=DEFAULT_THR, observed=bool(g <= t_last)))
    pd.DataFrame(crow).to_csv(RESULTS / "hi_rul_curves.csv", index=False)

    _figs(df, grid, curves, t_last, vb_max)

    print("=== RUL summary (default threshold 300 µm) ===")
    d300 = summ[summ.vb_failure_um == 300]
    print(d300[["model", "t_failure", "rul_at_last_obs", "crossing_within_horizon",
                "rul_extrapolated"]].to_string(index=False))
    print("\nALL thresholds extrapolated (official VB_max 212 < 220):",
          bool(summ.rul_extrapolated.all()))
    print(f"\n-> results/hi_rul_threshold_summary.csv ({len(summ)} rows), hi_rul_curves.csv, 2 figs")


def _figs(df, grid, curves, t_last, vb_max):
    colors = {"Linear(t)": "#1F4E79", "Poly2(t)": "#4A6628", "Classical_best(EN)": "#888",
              "PINN_mono": "#B3541E", "PINN_softplus_rate": "#2E6F62", "PINN_mono+softplus": "#9C6B1E"}
    t_obs = df[T_COL].to_numpy(float); vb_obs = df["VB_um"].to_numpy(float)

    # HI curves (default threshold 300)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for name, curve in curves.items():
        hi = health_index(curve, DEFAULT_THR, VB_0)
        ax.plot(grid, hi, color=colors.get(name, "#333"), label=name,
                lw=2 if name == "PINN_mono" else 1.4)
    ax.axvspan(grid[0], t_last, color="#2E6F62", alpha=0.07)
    ax.text(t_last - 0.3, 0.05, "observed range", fontsize=8, ha="right", color="#2E6F62")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("physical experiment order"); ax.set_ylabel("Health Index HI(t)")
    ax.set_title(f"#2 Health Index by model (threshold {DEFAULT_THR} µm; HI=1 healthy, 0=failure)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "hi_curves_by_model.png", bbox_inches="tight",
                                    facecolor="white"); plt.close(fig)

    # RUL threshold crossing (VB curves + threshold lines)
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=140)
    ax.plot(t_obs, vb_obs, "ko-", lw=2, label="measured VB (official)", zorder=5)
    for name, curve in curves.items():
        ax.plot(grid, curve, "--", color=colors.get(name, "#333"), label=name, lw=1.4)
    for thr in THRESHOLDS:
        ax.axhline(thr, color="#8C2D2D", ls=":", lw=1, alpha=0.6)
        ax.text(grid[-1], thr + 3, f"{thr} µm", fontsize=7, ha="right", color="#8C2D2D")
    ax.axvspan(grid[0], t_last, color="#2E6F62", alpha=0.07)
    ax.axhline(vb_max, color="#2E6F62", lw=1, ls="-.")
    ax.text(grid[0] + 0.2, vb_max + 4, f"VB_max observed = {vb_max:.0f} µm (< all thresholds)",
            fontsize=8, color="#2E6F62")
    ax.set_ylim(80, 640); ax.set_xlabel("physical experiment order"); ax.set_ylabel("VB (µm)")
    ax.set_title("#2 RUL threshold crossing — every crossing is EXTRAPOLATED (VB_max 212 < 220)")
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "rul_threshold_crossing.png", bbox_inches="tight",
                                    facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
