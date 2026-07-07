#!/usr/bin/env python3
"""
validate_bnn_entrant.py — chequeo rapido de que el BNN entra al harness.

Corre `run_branch` (la MISMA maquinaria LOEO del pipeline por capas) en dos
ramas clave — FUSION_N_ST y SOLO_A_N_ST — comparando BNN vs ElasticNet vs
DummyRegressor. No ejecuta las 36 ramas ni el cleanup/SHAP: solo valida
que el nuevo entrant BNN funciona bajo LOEO y muestra donde aterriza.

Uso:
    python scripts/validate_bnn_entrant.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phm.config import PROCESSED_DATASET
from phm.dataset_builder import get_feature_columns
from phm.layered_pipeline import run_branch, get_features_for_subset

MODELS = ["DummyRegressor", "ElasticNet", "BNN"]
BRANCHES = [
    ("FUSION_N_ST", "FUSION"),
    ("SOLO_A_N_ST", "SOLO_A"),
]


def main():
    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}. Corre scripts/build_dataset.py primero.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    print(f"[INFO] dataset shape={df.shape}  features_ML={len(feat_cols)}")
    print(f"[INFO] modelos a comparar: {MODELS}\n")

    all_rows = []
    for bid, subset in BRANCHES:
        feat_subset = get_features_for_subset(feat_cols, subset)
        t0 = time.time()
        res = run_branch(
            branch_id=bid,
            feature_subset=subset,
            data_branch="N",
            tuning_method="none",
            aug_strategy="none",
            full_df=df,
            feat_cols=feat_subset,
            models_filter=MODELS,
        )
        dt = time.time() - t0
        for r in res["metrics_rows"]:
            all_rows.append({
                "branch_id": bid,
                "subset": subset,
                "n_features": len(feat_subset),
                "model": r["model"],
                "MAE": round(r["MAE"], 2),
                "RMSE": round(r["RMSE"], 2),
                "R2": round(r["R2"], 3),
                "MAPE_%": round(r["MAPE_%"], 2),
            })
        print(f"[OK] rama {bid} en {dt:.1f}s\n")

    out = pd.DataFrame(all_rows).sort_values(["branch_id", "MAE"]).reset_index(drop=True)
    print("=" * 72)
    print("RESULTADOS LOEO (honesto, n=10) — BNN integrado al harness")
    print("=" * 72)
    print(out.to_string(index=False))
    print("=" * 72)
    print("\nReferencia (ranking actual del proyecto):")
    print("  ElasticNet SOLO_A (tuned): MAE=18.79, R2=0.82  <- mejor global vigente")
    print("  Esperado: el BNN regulariza fuerte con n=10; no deberia ganar a los")
    print("  lineales aqui. Su valor real llega con +datos (LOTO) y como backbone")
    print("  del B-PINN (fisica embebida).")


if __name__ == "__main__":
    main()
