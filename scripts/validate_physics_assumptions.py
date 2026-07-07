#!/usr/bin/env python3
"""
validate_physics_assumptions.py — VALIDACION EMPIRICA de la fisica antes
de construir el PINN.

Un PINN solo es honesto si las leyes que embebe se cumplen (al menos
aproximadamente) en los datos. Este script verifica, sobre los 10
experimentos de T01:

  H1. Monotonicidad:  VB es no-decreciente con experiment_order.
  H2. Aceleracion:    la tasa de desgaste dVB/dorder crece en el tercer
                      regimen (curva de desgaste clasica).
  H3. Ley de tasa:    algun proxy de energia de vibracion correla con VB
                      y/o con la tasa dVB. Si si -> justifica
                      dVB/dt = g(energia). Si no -> hay que repensar el
                      driver fisico.

Salidas: tabla por consola + CSV en outputs/metrics/physics_validation.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phm.config import (
    PROCESSED_DATASET, TARGET_COLUMN, EXP_ORDER_COL, TOOL_ID_COL, METRICS_DIR,
)


def main():
    df = pd.read_csv(PROCESSED_DATASET)
    df = df.sort_values([TOOL_ID_COL, EXP_ORDER_COL]).reset_index(drop=True)
    vb = df[TARGET_COLUMN].values.astype(float)
    order = df[EXP_ORDER_COL].values.astype(float)

    print("=" * 74)
    print("VALIDACION EMPIRICA DE LA FISICA  (T01, n=%d)" % len(df))
    print("=" * 74)

    # ---- H1: monotonicidad ----
    dvb = np.diff(vb)
    n_neg = int((dvb < 0).sum())
    print("\n[H1] Monotonicidad de VB con experiment_order")
    print(f"   VB = {vb.tolist()}")
    print(f"   dVB por paso = {np.round(dvb, 1).tolist()}")
    print(f"   pasos con dVB<0: {n_neg}  ->  "
          + ("MONOTONA ESTRICTA ✓" if n_neg == 0 else "NO monotona ✗"))
    rho_vb_order, _ = spearmanr(order, vb)
    print(f"   Spearman(order, VB) = {rho_vb_order:.3f} (1.0 = monotona perfecta)")

    # ---- H2: aceleracion (tercer regimen) ----
    print("\n[H2] Aceleracion de la tasa de desgaste")
    half = len(dvb) // 2
    early = float(np.mean(dvb[:half]))
    late = float(np.mean(dvb[half:]))
    print(f"   tasa media primera mitad = {early:.1f} µm/paso")
    print(f"   tasa media segunda mitad = {late:.1f} µm/paso")
    print(f"   aceleracion = {late - early:+.1f} µm/paso  -> "
          + ("ACELERA ✓ (justifica convexidad)" if late > early else "no acelera"))
    rho_rate_order, _ = spearmanr(order[1:], dvb)
    print(f"   Spearman(order, tasa) = {rho_rate_order:.3f}")

    # ---- H3: ley de tasa guiada por energia de vibracion ----
    print("\n[H3] ¿Algun proxy de energia/vibracion traza el desgaste?")
    # Candidatos: agregados + medias por-contacto si existen.
    candidates = [c for c in [
        "A_energy_total_6_contacts", "R_energy_total_6_contacts",
        "total_energy_6_contacts", "A_rms_mean_6_contacts",
        "R_rms_mean_6_contacts", "A_to_R_rms_ratio", "A_to_R_energy_ratio",
    ] if c in df.columns]
    # Tambien medias derivadas de columnas por contacto (rms, energy).
    for ch in ("A", "R"):
        for stat in ("rms", "energy", "std", "peak_to_peak"):
            cols = [f"{ch}_p{p}_{stat}" for p in range(1, 7)
                    if f"{ch}_p{p}_{stat}" in df.columns]
            if cols:
                newc = f"{ch}_{stat}_mean_derived"
                df[newc] = df[cols].mean(axis=1)
                candidates.append(newc)

    rows = []
    for c in candidates:
        d = df[c].values.astype(float)
        if np.allclose(np.nanstd(d), 0) or np.isnan(d).all():
            continue
        rho_vb, _ = spearmanr(d, vb)
        # tasa: driver en paso k  vs  dVB de k a k+1 (Usui/Archard: rate
        # depende del estado actual).
        rho_rate, _ = spearmanr(d[:-1], dvb)
        # monotonia del propio driver
        rho_mono, _ = spearmanr(order, d)
        rows.append({
            "driver": c,
            "spearman_vs_VB": round(float(rho_vb), 3),
            "spearman_vs_rate": round(float(rho_rate), 3),
            "spearman_monotonic": round(float(rho_mono), 3),
        })

    res = pd.DataFrame(rows).sort_values("spearman_vs_VB",
                                         key=lambda s: s.abs(),
                                         ascending=False).reset_index(drop=True)
    print("\n   Correlaciones (|Spearman|), ordenadas por |corr con VB|:")
    print(res.to_string(index=False))

    # Mejor driver
    if not res.empty:
        best = res.iloc[0]
        print(f"\n   Mejor proxy de desgaste: {best['driver']}  "
              f"(|rho_VB|={abs(best['spearman_vs_VB']):.2f})")
        strong = res[res["spearman_vs_VB"].abs() >= 0.7]
        print(f"   Drivers con |rho_VB|>=0.7: {len(strong)}  -> "
              + ("ley de tasa por energia JUSTIFICADA ✓"
                 if len(strong) > 0 else
                 "señal debil; revisar driver fisico ✗"))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "physics_validation.csv"
    res.to_csv(out_path, index=False)
    print(f"\n   CSV: {out_path.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    print("CONCLUSION: estas correlaciones definen el driver de la ley de")
    print("tasa del PINN. Solo embebemos fisica que los datos respaldan.")
    print("=" * 74)


if __name__ == "__main__":
    main()
