#!/usr/bin/env python3
"""
validate_c0.py — GATE C0 del B-PINN.

Verifica las dos condiciones que el equipo firma antes de encender la fisica:

  G0.1  PINN_off (PhysicsBNNRegressor con lambda=0) reproduce al BNN bit a
        bit bajo LOEO (|Δ MAE| <= 0.1, idealmente 0.0). Esto valida que el
        ruteo de fit-params (model__order/groups/driver_idx) NO altera el
        resultado y que la fisica esta correctamente "apagada" en lambda=0.

  G0.2  El check de leakage `physics_aux_train_only` da PASS (experiment_order
        y tool_id NO son features de X).

Uso:
    python scripts/validate_c0.py
Salida: PASS/FAIL por gate + tabla comparativa. Exit code != 0 si falla.
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

from phm.config import PROCESSED_DATASET
from phm.dataset_builder import get_feature_columns
from phm.layered_pipeline import run_branch, get_features_for_subset
from phm.leakage_audit import check_physics_aux_train_only

MODELS = ["BNN", "PINN_off"]
BRANCHES = [("FUSION_N_ST", "FUSION"), ("SOLO_A_N_ST", "SOLO_A")]
TOL = 0.1  # µm


def main():
    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}.")
        sys.exit(2)

    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    print(f"[INFO] dataset shape={df.shape}  features={len(feat_cols)}")
    print(f"[INFO] GATE C0: comparando {MODELS} bajo LOEO\n")

    rows = []
    for bid, subset in BRANCHES:
        feat = get_features_for_subset(feat_cols, subset)
        res = run_branch(branch_id=bid, feature_subset=subset, data_branch="N",
                         tuning_method="none", aug_strategy="none",
                         full_df=df, feat_cols=feat, models_filter=MODELS)
        by_model = {r["model"]: r for r in res["metrics_rows"]}
        for m in MODELS:
            r = by_model[m]
            rows.append({"branch_id": bid, "model": m,
                         "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]})

    res_df = pd.DataFrame(rows)
    print("=" * 68)
    print("Resultados LOEO")
    print("=" * 68)
    print(res_df.round(4).to_string(index=False))
    print("=" * 68)

    # ---- G0.1: reproduccion ----
    ok_repro = True
    print("\n[G0.1] PINN_off reproduce BNN:")
    for bid, _ in BRANCHES:
        bnn = res_df[(res_df.branch_id == bid) & (res_df.model == "BNN")].iloc[0]
        pof = res_df[(res_df.branch_id == bid) & (res_df.model == "PINN_off")].iloc[0]
        d_mae = abs(float(bnn.MAE) - float(pof.MAE))
        d_rmse = abs(float(bnn.RMSE) - float(pof.RMSE))
        passed = (d_mae <= TOL) and (d_rmse <= TOL)
        ok_repro &= passed
        print(f"   {bid:14s}  Δ|MAE|={d_mae:.4g}  Δ|RMSE|={d_rmse:.4g}  "
              + ("PASS ✓" if passed else "FAIL ✗"))

    # ---- G0.2: leakage ----
    chk = check_physics_aux_train_only(df)
    ok_leak = chk["status"] == "PASS"
    print(f"\n[G0.2] leakage physics_aux_train_only: {chk['status']}  "
          + ("✓" if ok_leak else "✗"))
    print(f"        {chk['details']}")

    print("\n" + "=" * 68)
    if ok_repro and ok_leak:
        print("GATE C0: PASS ✓  — el ruteo no altera resultados y la fisica")
        print("apagada (lambda=0) reproduce el BNN. Listo para C1 (monotonicidad).")
        print("=" * 68)
        sys.exit(0)
    else:
        print("GATE C0: FAIL ✗  — revisar antes de encender la fisica.")
        print("=" * 68)
        sys.exit(1)


if __name__ == "__main__":
    main()
