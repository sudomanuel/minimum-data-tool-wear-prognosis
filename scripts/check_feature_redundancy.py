#!/usr/bin/env python3
"""
check_feature_redundancy.py — redundancia entre las 16 features base.

Reune las features de TODAS las senales crudas (una fila por archivo) y mide
cuanto se solapan entre si:
  - matriz de correlacion (|r|),
  - VIF (variance inflation factor) = diagonal de inv(matriz de correlacion);
    VIF alto => la feature es casi reconstruible desde las otras (redundante),
  - pares muy correlacionados (|r| >= 0.9),
  - columnas casi constantes (varianza ~ 0).

No mide aporte al TARGET (eso es la validacion fisica); mide solapamiento
entre features para decidir si conviene podar alguna.

Uso: python scripts/check_feature_redundancy.py
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

from phm.config import SEGMENTS_DIR, METRICS_DIR
from phm.feature_extraction import all_feature_names
from phm.ingestion import extract_file_features


def main():
    base = all_feature_names(True)
    files = sorted(Path(SEGMENTS_DIR).glob("*.txt"))
    if not files:
        print(f"ERROR: no hay .txt en {SEGMENTS_DIR}")
        sys.exit(2)
    print(f"[INFO] senales: {len(files)}   features base: {len(base)}")

    rows = [extract_file_features(f, enable_frequency=True, use_cache=True) for f in files]
    df = pd.DataFrame(rows).reindex(columns=base)

    # 1) columnas casi constantes (varianza ~ 0)
    std = df.std(numeric_only=True)
    near_const = std[std < 1e-9].index.tolist()
    if near_const:
        print(f"[CASI CONSTANTES] (sin info): {near_const}")
    usable = [c for c in base if c not in near_const]

    # imputar NaN con mediana para el calculo
    X = df[usable].copy()
    X = X.fillna(X.median(numeric_only=True))

    # 2) correlacion
    corr = X.corr()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(METRICS_DIR / "feature_correlation.csv")

    # 3) VIF = diagonal de la inversa (pseudo) de la matriz de correlacion
    C = corr.values
    vif = np.diag(np.linalg.pinv(C))
    vif_s = pd.Series(vif, index=usable).sort_values(ascending=False)

    # 4) score de redundancia simple: |corr| medio con las demas
    abscorr = corr.abs()
    np.fill_diagonal(abscorr.values, np.nan)
    mean_abscorr = abscorr.mean().reindex(usable)

    summary = pd.DataFrame({
        "VIF": vif_s,
        "mean_abs_corr": mean_abscorr.reindex(vif_s.index).round(3),
    })
    summary["VIF"] = summary["VIF"].round(1)
    summary["veredicto"] = np.where(
        summary["VIF"] >= 10, "redundante (VIF>=10)",
        np.where(summary["VIF"] >= 5, "solapada (VIF 5-10)", "aporta (VIF<5)"))
    summary.to_csv(METRICS_DIR / "feature_redundancy_vif.csv")

    print("\n" + "=" * 64)
    print("REDUNDANCIA POR FEATURE  (VIF alto = redundante)")
    print("=" * 64)
    print(summary.to_string())

    # 5) pares muy correlacionados
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.9:
                pairs.append((cols[i], cols[j], round(float(r), 3)))
    pairs.sort(key=lambda p: -abs(p[2]))
    print("\n" + "=" * 64)
    print(f"PARES casi duplicados  (|r| >= 0.9):  {len(pairs)}")
    print("=" * 64)
    for a, b, r in pairs:
        print(f"  {a:18s} <-> {b:18s}  r={r:+.3f}")

    n_red = int((summary["VIF"] >= 10).sum())
    n_keep = int((summary["VIF"] < 5).sum())
    print("\n" + "=" * 64)
    print(f"RESUMEN: {n_keep} features aportan (VIF<5), {n_red} redundantes (VIF>=10), "
          f"{len(near_const)} casi constantes.")
    print("CSV: outputs/metrics/feature_redundancy_vif.csv + feature_correlation.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
