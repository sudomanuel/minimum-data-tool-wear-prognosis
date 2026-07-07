#!/usr/bin/env python3
"""
calibrate_c1.py — STAGE C1: calibracion del termino de monotonicidad.

Enciende SOLO `lambda_mono` y barre su valor. Mide, bajo LOEO (la misma
maquinaria de folds del pipeline), las metricas del gate C1:

  - MAE_oof            : MAE de las predicciones out-of-fold (debe NO empeorar
                         > 5 µm vs lambda=0 = BNN).
  - viol_oof           : nº de pasos donde la TRAYECTORIA predicha OOF (10
                         experimentos ordenados por experiment_order) decrece
                         -> GATE: debe llegar a 0.
  - viol_insample      : idem pero con el modelo entrenado en los 10 (full-fit);
                         diagnostico de cuanto "agarra" la fisica.

Regla anti-cherry-pick: elegimos el MENOR lambda que da viol_oof=0 con
|ΔMAE| <= 5 µm. NO se elige por MAE de test.

Uso:
    python scripts/calibrate_c1.py [BRANCH] [SUBSET]
    (default: SOLO_A_N_ST SOLO_A)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.base import clone
from phm.config import (
    PROCESSED_DATASET, TARGET_COLUMN, EXPERIMENT_ID_COL, EXP_ORDER_COL, METRICS_DIR,
)
from phm.dataset_builder import get_feature_columns
from phm.layered_pipeline import (
    get_features_for_subset, _physics_fit_kwargs,
)
from phm.modeling import build_pinn
from phm.splitting import loeo_iter
from phm.evaluation import compute_metrics

LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1000.0]
MAE_TOL = 5.0  # µm


def _count_violations(order, pred):
    """nº de pasos con prediccion decreciente al ordenar por `order`."""
    o = np.asarray(order, float)
    p = np.asarray(pred, float)
    idx = np.argsort(o)
    seq = p[idx]
    return int(np.sum(np.diff(seq) < -1e-6))


def sweep(branch_id: str, subset: str):
    df = pd.read_csv(PROCESSED_DATASET)
    feat = get_features_for_subset(get_feature_columns(df), subset)
    order_by_eid = dict(zip(df[EXPERIMENT_ID_COL].astype(int),
                            df[EXP_ORDER_COL].astype(float)))
    print(f"[INFO] branch={branch_id} subset={subset} n_feats={len(feat)}")
    print(f"[INFO] barriendo lambda_mono={LAMBDAS}\n")

    rows = []
    for lam in LAMBDAS:
        # ---- LOEO: predicciones out-of-fold ----
        oof_eid, oof_true, oof_pred = [], [], []
        for tr_df, te_df in loeo_iter(df, group_col=EXPERIMENT_ID_COL):
            est = build_pinn(lambda_mono=lam)
            fitkw = _physics_fit_kwargs(est, tr_df, feat)
            X_tr = tr_df[feat].values.astype(float)
            y_tr = tr_df[TARGET_COLUMN].values.astype(float)
            X_te = te_df[feat].values.astype(float)
            est.fit(X_tr, y_tr, **fitkw)
            p = est.predict(X_te)
            oof_eid.extend(te_df[EXPERIMENT_ID_COL].astype(int).tolist())
            oof_true.extend(te_df[TARGET_COLUMN].astype(float).tolist())
            oof_pred.extend(np.asarray(p, float).tolist())
        mets = compute_metrics(np.array(oof_true), np.array(oof_pred))
        viol_oof = _count_violations([order_by_eid[e] for e in oof_eid], oof_pred)

        # ---- full-fit: trayectoria in-sample ----
        est_full = build_pinn(lambda_mono=lam)
        fitkw = _physics_fit_kwargs(est_full, df, feat)
        est_full.fit(df[feat].values.astype(float),
                     df[TARGET_COLUMN].values.astype(float), **fitkw)
        p_full = est_full.predict(df[feat].values.astype(float))
        viol_in = _count_violations(df[EXP_ORDER_COL].values, p_full)

        rows.append({
            "lambda_mono": lam, "MAE_oof": round(mets["MAE"], 2),
            "RMSE_oof": round(mets["RMSE"], 2), "R2_oof": round(mets["R2"], 3),
            "viol_oof": viol_oof, "viol_insample": viol_in,
        })
        print(f"  lambda={lam:7.1f}  MAE={mets['MAE']:6.2f}  R2={mets['R2']:6.3f}  "
              f"viol_oof={viol_oof}  viol_insample={viol_in}")

    res = pd.DataFrame(rows)
    base_mae = float(res.loc[res.lambda_mono == 0, "MAE_oof"].iloc[0])
    res["delta_MAE_vs_BNN"] = (res["MAE_oof"] - base_mae).round(2)

    # ---- recomendacion del gate ----
    ok = res[(res.viol_oof == 0) & (res.delta_MAE_vs_BNN.abs() <= MAE_TOL)
             & (res.lambda_mono > 0)]
    rec = float(ok.sort_values("lambda_mono")["lambda_mono"].iloc[0]) if not ok.empty else None

    print("\n" + "=" * 64)
    print("STAGE C1 — barrido lambda_mono  (gate: viol_oof=0, |ΔMAE|<=5)")
    print("=" * 64)
    print(res.to_string(index=False))
    print("=" * 64)
    print(f"baseline (BNN, lambda=0): MAE_oof={base_mae:.2f}, "
          f"viol_oof={int(res.loc[res.lambda_mono==0,'viol_oof'].iloc[0])}")
    if rec is not None:
        print(f"GATE C1: PASS ✓  -> lambda_mono recomendado = {rec:g}")
    else:
        print("GATE C1: sin lambda que logre viol_oof=0 con |ΔMAE|<=5; "
              "ampliar grid o revisar escala del termino.")
    print("=" * 64)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"c1_lambda_mono_sweep_{branch_id}.csv"
    res.to_csv(out, index=False)
    print(f"CSV: {out.relative_to(PROJECT_ROOT)}")
    return res, rec


if __name__ == "__main__":
    branch = sys.argv[1] if len(sys.argv) > 1 else "SOLO_A_N_ST"
    subset = sys.argv[2] if len(sys.argv) > 2 else "SOLO_A"
    sweep(branch, subset)
