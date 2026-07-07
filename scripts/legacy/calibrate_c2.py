#!/usr/bin/env python3
"""
calibrate_c2.py — STAGE C2: calibracion de la ley de tasa de desgaste.

Enciende SOLO `lambda_rate` (mono=conv=0). La ley embebida es
    dVB/dt = g(E_rot),  g(E)=a+softplus(b)*E,  b>=0
con E_rot = energia de vibracion ROTACIONAL (driver validado, rho=0.76 vs
tasa real). Debe calibrarse en una rama que CONTENGA el driver -> FUSION
(SOLO_A no tiene features R_ => driver_idx=None => termino inactivo).

Mide, bajo LOEO y full-fit:
  - MAE_oof                : no debe empeorar > 5 µm vs lambda=0 (BNN).
  - cons_insample / cons_oof: Spearman(tasa PREDICHA, E_rot) — ¿la tasa que
                              predice el modelo empieza a trazar la energia
                              rotacional? Comparar contra la consistencia
                              OBSERVADA (tasa real vs E_rot ~ 0.76).

Gate C2: cons_insample sube hacia/encima de la observada SIN regresion de
MAE > 5 µm. (Eleccion de lambda por criterio fisico, no por MAE de test.)

Uso:
    python scripts/calibrate_c2.py [BRANCH] [SUBSET]
    (default: FUSION_N_ST FUSION)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phm.config import (
    PROCESSED_DATASET, TARGET_COLUMN, EXPERIMENT_ID_COL, EXP_ORDER_COL, METRICS_DIR,
)
from phm.dataset_builder import get_feature_columns
from phm.layered_pipeline import (
    get_features_for_subset, _physics_fit_kwargs, PHYSICS_DRIVER_COL,
)
from phm.modeling import build_pinn
from phm.splitting import loeo_iter
from phm.evaluation import compute_metrics

LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1000.0]
MAE_TOL = 5.0  # µm


def _rate_energy_consistency(order, pred, e_rot):
    """Spearman(tasa predicha, E_rot del experimento anterior). Convencion
    de la validacion fisica: rate_k = pred(k+1)-pred(k) alineado con E_rot[k]."""
    o = np.asarray(order, float); idx = np.argsort(o)
    p = np.asarray(pred, float)[idx]
    e = np.asarray(e_rot, float)[idx]
    rate = np.diff(p)              # 9 tasas
    e_prev = e[:-1]                # E_rot del experimento anterior
    if np.std(rate) < 1e-9 or np.std(e_prev) < 1e-9:
        return float("nan")
    rho, _ = spearmanr(rate, e_prev)
    return float(rho)


def _violations(order, pred):
    o = np.asarray(order, float); p = np.asarray(pred, float)
    return int(np.sum(np.diff(p[np.argsort(o)]) < -1e-6))


def sweep(branch_id: str, subset: str):
    df = pd.read_csv(PROCESSED_DATASET)
    feat = get_features_for_subset(get_feature_columns(df), subset)
    has_driver = PHYSICS_DRIVER_COL in feat
    print(f"[INFO] branch={branch_id} subset={subset} n_feats={len(feat)}")
    print(f"[INFO] driver={PHYSICS_DRIVER_COL} en features: {has_driver}")
    if not has_driver:
        print("ERROR: la rama no contiene el driver rotacional -> usar FUSION.")
        sys.exit(2)

    order_by_eid = dict(zip(df[EXPERIMENT_ID_COL].astype(int), df[EXP_ORDER_COL].astype(float)))
    erot_by_eid = dict(zip(df[EXPERIMENT_ID_COL].astype(int), df[PHYSICS_DRIVER_COL].astype(float)))

    # consistencia OBSERVADA: tasa real vs E_rot
    obs_cons = _rate_energy_consistency(
        df[EXP_ORDER_COL].values, df[TARGET_COLUMN].values, df[PHYSICS_DRIVER_COL].values)
    print(f"[INFO] consistencia OBSERVADA (tasa real vs E_rot) = {obs_cons:.3f}")
    print(f"[INFO] barriendo lambda_rate={LAMBDAS}\n")

    rows = []
    for lam in LAMBDAS:
        oof_eid, oof_true, oof_pred = [], [], []
        for tr_df, te_df in loeo_iter(df, group_col=EXPERIMENT_ID_COL):
            est = build_pinn(lambda_rate=lam)
            fitkw = _physics_fit_kwargs(est, tr_df, feat)
            est.fit(tr_df[feat].values.astype(float),
                    tr_df[TARGET_COLUMN].values.astype(float), **fitkw)
            p = est.predict(te_df[feat].values.astype(float))
            oof_eid.extend(te_df[EXPERIMENT_ID_COL].astype(int).tolist())
            oof_true.extend(te_df[TARGET_COLUMN].astype(float).tolist())
            oof_pred.extend(np.asarray(p, float).tolist())
        mets = compute_metrics(np.array(oof_true), np.array(oof_pred))
        o_oof = [order_by_eid[e] for e in oof_eid]
        e_oof = [erot_by_eid[e] for e in oof_eid]
        cons_oof = _rate_energy_consistency(o_oof, oof_pred, e_oof)
        viol_oof = _violations(o_oof, oof_pred)

        est_full = build_pinn(lambda_rate=lam)
        fitkw = _physics_fit_kwargs(est_full, df, feat)
        est_full.fit(df[feat].values.astype(float),
                     df[TARGET_COLUMN].values.astype(float), **fitkw)
        p_full = est_full.predict(df[feat].values.astype(float))
        cons_in = _rate_energy_consistency(df[EXP_ORDER_COL].values, p_full,
                                           df[PHYSICS_DRIVER_COL].values)

        rows.append({
            "lambda_rate": lam, "MAE_oof": round(mets["MAE"], 2),
            "R2_oof": round(mets["R2"], 3), "cons_insample": round(cons_in, 3),
            "cons_oof": round(cons_oof, 3), "viol_oof": viol_oof,
        })
        print(f"  lambda={lam:7.1f}  MAE={mets['MAE']:6.2f}  R2={mets['R2']:6.3f}  "
              f"cons_in={cons_in:+.3f}  cons_oof={cons_oof:+.3f}")

    res = pd.DataFrame(rows)
    base_mae = float(res.loc[res.lambda_rate == 0, "MAE_oof"].iloc[0])
    base_cons = float(res.loc[res.lambda_rate == 0, "cons_insample"].iloc[0])
    res["delta_MAE_vs_BNN"] = (res["MAE_oof"] - base_mae).round(2)

    print("\n" + "=" * 70)
    print("STAGE C2 — barrido lambda_rate  (gate: cons_insample sube, |ΔMAE|<=5)")
    print("=" * 70)
    print(res.to_string(index=False))
    print("=" * 70)
    print(f"consistencia OBSERVADA (tasa real vs E_rot) = {obs_cons:.3f}")
    print(f"baseline BNN (lambda=0): MAE={base_mae:.2f}, cons_insample={base_cons:+.3f}")

    # gate: cons_insample mejora respecto a BNN y MAE no empeora >5
    ok = res[(res.lambda_rate > 0)
             & (res.cons_insample > base_cons + 0.05)
             & (res.delta_MAE_vs_BNN.abs() <= MAE_TOL)]
    rec = float(ok.sort_values("lambda_rate")["lambda_rate"].iloc[0]) if not ok.empty else None
    if rec is not None:
        print(f"GATE C2: PASS ✓ -> lambda_rate recomendado = {rec:g} "
              f"(la tasa predicha empieza a trazar E_rot sin danar MAE)")
    else:
        print("GATE C2: revisar — ningun lambda sube cons_insample con |ΔMAE|<=5.")
    print("=" * 70)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"c2_lambda_rate_sweep_{branch_id}.csv"
    res.to_csv(out, index=False)
    print(f"CSV: {out.relative_to(PROJECT_ROOT)}")
    return res, rec


if __name__ == "__main__":
    branch = sys.argv[1] if len(sys.argv) > 1 else "FUSION_N_ST"
    subset = sys.argv[2] if len(sys.argv) > 2 else "FUSION"
    sweep(branch, subset)
