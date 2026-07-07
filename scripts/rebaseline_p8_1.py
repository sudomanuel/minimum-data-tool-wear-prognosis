#!/usr/bin/env python3
"""
rebaseline_p8_1.py — P8.1 bridge re-baseline of classical ML + PINN on the OFFICIAL VB target.

Uses the CURRENT features (no segmentation/feature refactor yet) but replaces the legacy VB with
the official VB (microscope_vb.csv) for the 10 recorded experiments. 71-72 are target-only
(excluded from sensor models). exp77 kept with reliability flags. Reuses the existing PINN
infrastructure (PINNRegressor, PINN_VARIANTS). Does NOT modify frozen P1/P3 code.

Outputs (results/):
  p8_1_official_vb_classical_results.csv
  p8_1_official_vb_pinn_ablation_results.csv
  p8_1_official_vb_model_comparison_summary.csv
  p8_1_fold_predictions.csv
Figures: outputs/figures/p8_1/

Uso:  python run.py rebaseline-p8-1   [--repo-root <repo-root>] [--pinn-epochs 2000]
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

WT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT_ROOT / "src"))

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = WT_ROOT / "results"
FIGS = WT_ROOT / "outputs" / "figures" / "p8_1"
META_COLS = {"experiment_id", "tool_id", "experiment_order", "F", "end_of_life", "VB_um"}
ENERGY_TOTAL_COLS = ["A_energy_total_6_contacts", "R_energy_total_6_contacts",
                     "total_energy_6_contacts"]
RECORDED = [66, 67, 68, 69, 70, 73, 74, 75, 76, 77]
# Official time-aware baselines from P8.0 (recorded-10, gapped t) for the comparison table.
OFFICIAL_TA = {"Linear(t)": 3.10, "Poly2(t)": 3.83}
LEGACY_HIST = {"P1_best_legacy": ("ElasticNet SOLO_A", 19.07),
               "P3_best_legacy": ("PINN_mono", 18.57)}


def _metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else float("nan")
    return mae, rmse, r2


def _mono_violations(pred_df):
    d = pred_df.sort_values("experiment_order")["VB_pred"].to_numpy(float)
    diffs = np.diff(d)
    return int(np.sum(diffs < -1e-6)), float(np.mean(diffs < 0))


def _models():
    return {
        "Dummy(mean)": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: Ridge(alpha=10.0, random_state=0),
        "Lasso": lambda: Lasso(alpha=1.0, max_iter=50000),
        "ElasticNet": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=50000),
        "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=1.0),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=0),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=0),
        "MLP(no physics)": lambda: MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=4000,
                                                random_state=0, early_stopping=False),
    }


def classical_loeo(df, feat_cols, variant):
    """LOEO classical battery, train-only median impute + StandardScaler."""
    out_rows, preds_all = [], []
    ids = df["experiment_id"].to_numpy()
    order = df["experiment_order"].to_numpy()
    X = df[feat_cols].to_numpy(float)
    y = df["VB_um"].to_numpy(float)
    for name, builder in _models().items():
        yp = np.zeros(len(df))
        for i in range(len(df)):
            tr = np.arange(len(df)) != i
            med = np.nanmedian(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[i:i + 1]), med, X[i:i + 1])
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = builder().fit(Xtr, y[tr])
            yp[i] = m.predict(Xte)[0]
        mae, rmse, r2 = _metrics(y, yp)
        pdf = pd.DataFrame({"experiment_id": ids, "experiment_order": order,
                            "VB_true": y, "VB_pred": yp, "model": name,
                            "variant": variant, "group": "classical"})
        mono, negr = _mono_violations(pdf)
        e77 = pdf[pdf.experiment_id == 77]
        out_rows.append({"group": "classical", "variant": variant, "model": name,
                         "n_features": len(feat_cols), "MAE": round(mae, 3),
                         "RMSE": round(rmse, 3), "R2": round(r2, 3),
                         "monotonicity_violations": mono,
                         "exp77_residual": round(float(e77.VB_pred.iloc[0] - e77.VB_true.iloc[0]), 2)
                         if len(e77) else np.nan})
        preds_all.append(pdf)
        print(f"  [{variant}] {name:18s} MAE={mae:6.2f} RMSE={rmse:6.2f} R2={r2:6.2f} "
              f"monoViol={mono} exp77res={out_rows[-1]['exp77_residual']:+.1f}", flush=True)
    return out_rows, pd.concat(preds_all, ignore_index=True)


def pinn_loeo(df, epochs):
    from phm.pinn import (PINNRegressor, PINN_VARIANTS, select_minimal_physical_features,
                          resolve_driver_col)
    x_min = select_minimal_physical_features(df.columns)
    driver = resolve_driver_col(df.columns)
    print(f"[PINN] x_min={x_min} driver={driver}", flush=True)
    variants = ["PINN_no_physics", "PINN_mono", "PINN_rate", "PINN_full"]
    out_rows, preds_all = [], []
    for vname in variants:
        lambdas = PINN_VARIANTS[vname]
        yp = np.zeros(len(df))
        idx = df.reset_index(drop=True)
        for i in range(len(idx)):
            tr = idx.index != i
            trd, ted = idx[tr], idx[idx.index == i]
            pinn = PINNRegressor(hidden=(32, 32), epochs=epochs, random_state=42, **lambdas)
            e_tr = trd[driver].values if driver is not None else None
            pinn.fit(trd[x_min].values, trd["experiment_order"].values,
                     trd["VB_um"].values, e_rot=e_tr)
            yp[i] = pinn.predict(ted[x_min].values, ted["experiment_order"].values)[0]
        y = idx["VB_um"].to_numpy(float)
        mae, rmse, r2 = _metrics(y, yp)
        pdf = pd.DataFrame({"experiment_id": idx.experiment_id, "experiment_order": idx.experiment_order,
                            "VB_true": y, "VB_pred": yp, "model": vname,
                            "variant": "pinn_xmin", "group": "pinn"})
        mono, negr = _mono_violations(pdf)
        # physical residual proxy: smoothness of OOF pooled trajectory
        seq = pdf.sort_values("experiment_order")["VB_pred"].to_numpy(float)
        smooth = float(np.mean(np.diff(seq, n=2) ** 2)) if len(seq) > 2 else np.nan
        e77 = pdf[pdf.experiment_id == 77]
        out_rows.append({"group": "pinn", "variant": "pinn_xmin", "model": vname,
                         "uses_physics": any(v > 0 for v in lambdas.values()),
                         "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3),
                         "monotonicity_violations": mono,
                         "negative_rate_fraction": round(negr, 3),
                         "physical_residual_smoothness": round(smooth, 1),
                         "exp77_residual": round(float(e77.VB_pred.iloc[0] - e77.VB_true.iloc[0]), 2)})
        preds_all.append(pdf)
        print(f"  {vname:18s} MAE={mae:6.2f} R2={r2:6.2f} monoViol={mono} "
              f"smooth={smooth:8.1f} exp77res={out_rows[-1]['exp77_residual']:+.1f}", flush=True)
    return out_rows, pd.concat(preds_all, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--pinn-epochs", type=int, default=2000)
    ap.add_argument("--skip-pinn", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from phm.targets import load_official_vb
    feats = pd.read_csv(args.repo_root / "data" / "processed" / "experiment_features.csv")
    feats = feats[feats.experiment_id.isin(RECORDED)].copy()
    official = load_official_vb(recorded_only=True,
                               data_root=args.repo_root / "data" / "targets")
    vb_map = dict(zip(official.experiment_id, official.VB_um))
    feats["VB_um"] = feats.experiment_id.map(vb_map)   # REPLACE legacy with official
    feats = feats.sort_values("experiment_order").reset_index(drop=True)
    print(f"official VB joined: {dict(zip(feats.experiment_id, feats.VB_um))}")
    print(f"(legacy was 85-280; official 103-212)\n")

    all_feat = [c for c in feats.columns if c not in META_COLS
                and pd.api.types.is_numeric_dtype(feats[c])]
    feat_B = [c for c in all_feat if c not in ENERGY_TOTAL_COLS]

    print("=== CLASSICAL — Variant A (current features) ===")
    rowsA, predA = classical_loeo(feats, all_feat, "A_current")
    print("\n=== CLASSICAL — Variant B (reliability-aware: drop energy_total) ===")
    rowsB, predB = classical_loeo(feats, feat_B, "B_reliability_aware")
    cls = pd.DataFrame(rowsA + rowsB)
    cls.to_csv(RESULTS / "p8_1_official_vb_classical_results.csv", index=False)

    pinn_rows, pinn_pred = [], pd.DataFrame()
    if not args.skip_pinn:
        print(f"\n=== PINN ablation (epochs={args.pinn_epochs}) ===")
        pinn_rows, pinn_pred = pinn_loeo(feats, args.pinn_epochs)
        pd.DataFrame(pinn_rows).to_csv(
            RESULTS / "p8_1_official_vb_pinn_ablation_results.csv", index=False)

    # fold predictions (for plots)
    allpred = pd.concat([predA, predB] + ([pinn_pred] if len(pinn_pred) else []),
                        ignore_index=True)
    allpred.to_csv(RESULTS / "p8_1_fold_predictions.csv", index=False)

    # comparison summary
    best_cls = cls.loc[cls.MAE.idxmin()]
    summary = [
        {"model": "Linear(t) official", "target_source": "microscope_vb.csv", "n_points": 10,
         "uses_signal": False, "uses_time": True, "MAE": OFFICIAL_TA["Linear(t)"],
         "notes": "P8.0 time-aware (recorded-10, gapped t)"},
        {"model": "Poly2(t) official", "target_source": "microscope_vb.csv", "n_points": 10,
         "uses_signal": False, "uses_time": True, "MAE": OFFICIAL_TA["Poly2(t)"],
         "notes": "P8.0 time-aware"},
        {"model": f"best classical ML official ({best_cls.model}/{best_cls.variant})",
         "target_source": "microscope_vb.csv", "n_points": 10, "uses_signal": True,
         "uses_time": False, "MAE": float(best_cls.MAE), "RMSE": float(best_cls.RMSE),
         "R2": float(best_cls.R2), "notes": "sensor features, LOEO"},
    ]
    if pinn_rows:
        pr = pd.DataFrame(pinn_rows)
        best_pinn = pr.loc[pr.MAE.idxmin()]
        best_phys = pr[pr.uses_physics]
        best_phys = best_phys.loc[best_phys.MAE.idxmin()] if len(best_phys) else best_pinn
        summary += [
            {"model": f"best PINN official ({best_pinn.model})", "target_source": "microscope_vb.csv",
             "n_points": 10, "uses_signal": True, "uses_time": True, "MAE": float(best_pinn.MAE),
             "RMSE": float(best_pinn.RMSE), "R2": float(best_pinn.R2),
             "monotonicity_violations": int(best_pinn.monotonicity_violations),
             "notes": "x_min + t"},
            {"model": f"best PINN physical-consistency ({best_phys.model})",
             "target_source": "microscope_vb.csv", "n_points": 10, "uses_signal": True,
             "uses_time": True, "MAE": float(best_phys.MAE),
             "monotonicity_violations": int(best_phys.monotonicity_violations),
             "physical_residual": float(best_phys.physical_residual_smoothness),
             "notes": "best physics-constrained variant"},
        ]
    summary += [
        {"model": "historical P1 best (legacy)", "target_source": "vb_targets.csv", "n_points": 10,
         "uses_signal": True, "uses_time": False, "MAE": LEGACY_HIST["P1_best_legacy"][1],
         "notes": LEGACY_HIST["P1_best_legacy"][0] + " — legacy target 85-280"},
        {"model": "historical P3 best (legacy)", "target_source": "vb_targets.csv", "n_points": 10,
         "uses_signal": True, "uses_time": True, "MAE": LEGACY_HIST["P3_best_legacy"][1],
         "notes": LEGACY_HIST["P3_best_legacy"][0] + " — legacy target 85-280"},
    ]
    pd.DataFrame(summary).to_csv(
        RESULTS / "p8_1_official_vb_model_comparison_summary.csv", index=False)

    _plots(allpred, cls, pinn_rows)
    print(f"\nDONE in {time.time()-t0:.0f}s. Results in {RESULTS}/  figures in {FIGS}/")
    print("\nComparison summary:")
    for s in summary:
        print(f"  {s['model']:48s} MAE={s.get('MAE'):>6}  {s['notes']}")
    return 0


def _plots(allpred, cls, pinn_rows):
    # real vs predicted (best classical + best pinn) + residuals by experiment
    best_cls = cls.loc[cls.MAE.idxmin()]
    bc = allpred[(allpred.model == best_cls.model) & (allpred.variant == best_cls.variant)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    d = bc.sort_values("experiment_order")
    ax[0].plot(d.experiment_order, d.VB_true, "ko-", label="measured VB (official)")
    ax[0].plot(d.experiment_order, d.VB_pred, "s--", color="#1F4E79",
               label=f"{best_cls.model} (MAE {best_cls.MAE})")
    if pinn_rows:
        pr = pd.DataFrame(pinn_rows)
        bp = pr.loc[pr.MAE.idxmin()]
        bpp = allpred[allpred.model == bp.model].sort_values("experiment_order")
        ax[0].plot(bpp.experiment_order, bpp.VB_pred, "^--", color="#B3541E",
                   label=f"{bp.model} (MAE {bp.MAE})")
    ax[0].set_xlabel("experiment_order"); ax[0].set_ylabel("VB (µm)")
    ax[0].set_title("Official VB — measured vs predicted (LOEO)"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)
    d["residual"] = d.VB_pred - d.VB_true
    ax[1].bar(d.experiment_order, d.residual, color="#1F4E79", alpha=0.8)
    ax[1].axhline(0, color="k", lw=0.8)
    e77 = d[d.experiment_id == 77]
    if len(e77):
        ax[1].bar(e77.experiment_order, e77.residual, color="#8C2D2D", label="exp77 (4 contacts)")
        ax[1].legend(fontsize=8)
    ax[1].set_xlabel("experiment_order"); ax[1].set_ylabel("residual (µm)")
    ax[1].set_title(f"Residuals by experiment — {best_cls.model}"); ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "p8_1_official_vb_pred_residuals.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {FIGS / 'p8_1_official_vb_pred_residuals.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
