#!/usr/bin/env python3
"""
run_pinn.py — train and evaluate the wear-curve PINN under LOEO.

Searches a small grid of PINN configurations (hidden size, physics weights),
evaluates each under Leave-One-Experiment-Out, and reports the best one next
to the project's linear baseline. This is the environment in which the
"optimal PINN" is found once the multi-cutter data is in; on T01 alone the
formulation is degenerate (t predicts VB almost perfectly), which the output
flags explicitly.

Usage:
    python run.py pinn          # via the dispatcher
    python scripts/run_pinn.py  # direct
"""
from __future__ import annotations
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phm.config import PROCESSED_DATASET, METRICS_DIR, EXP_ORDER_COL
from phm.dataset_builder import get_feature_columns
from phm.pinn import loeo_evaluate_pinn, TORCH_AVAILABLE

# Small search grid (the "optimal PINN" lives here; widen once data lands).
GRID = {
    "hidden":      [(32,), (64, 64)],
    "lambda_mono": [0.0, 1.0],
    "lambda_rate": [0.0, 0.1],
    "lr":          [1e-2],
    "epochs":      [2000],
}


def main():
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch no disponible. Instala torch>=2.5.")
        sys.exit(2)
    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}. Corre `python run.py dataset` primero.")
        sys.exit(2)

    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    print(f"[INFO] dataset {df.shape}  features={len(feat_cols)}  t={EXP_ORDER_COL}")

    keys = list(GRID)
    combos = list(product(*(GRID[k] for k in keys)))
    print(f"[INFO] evaluando {len(combos)} configuraciones de PINN bajo LOEO\n")

    rows = []
    for values in combos:
        cfg = dict(zip(keys, values))
        t0 = time.time()
        mets, _ = loeo_evaluate_pinn(df, feat_cols, **cfg)
        rows.append({
            "hidden": str(cfg["hidden"]), "lambda_mono": cfg["lambda_mono"],
            "lambda_rate": cfg["lambda_rate"], "lr": cfg["lr"], "epochs": cfg["epochs"],
            "MAE": round(mets["MAE"], 2), "RMSE": round(mets["RMSE"], 2),
            "R2": round(mets["R2"], 3), "MAPE_%": round(mets["MAPE_%"], 2),
            "mono_violations": mets["mono_violations"],
        })
        print(f"  {cfg}  ->  MAE={mets['MAE']:.2f}  R2={mets['R2']:.3f}  "
              f"mono_viol={mets['mono_violations']}  ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    print("\n" + "=" * 70)
    print("PINN — LOEO ranking (best MAE on top)")
    print("=" * 70)
    print(res.to_string(index=False))
    print("=" * 70)
    print("Baseline de referencia (ElasticNet, axial): MAE=18.79, R2=0.82")
    print("AVISO: en T01 (1 cuchilla) la PINN puede apoyarse en t e ignorar la")
    print("vibracion; un buen MAE aqui NO prueba uso de sensores. La validacion")
    print("real es Leave-One-Tool-Out con varias cuchillas.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "pinn_loeo_search.csv"
    res.to_csv(out, index=False)
    print(f"\nCSV: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
