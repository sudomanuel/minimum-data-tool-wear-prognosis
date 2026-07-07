#!/usr/bin/env python3
"""
diag_pinn_vs_line.py — DIAGNOSTIC (standalone, NOT part of the official pipeline).

Answers the supervisor's question: "if VB is basically a straight line in experiment order,
why does the PINN_mono lose to the time model?" It decomposes the gap into two "taxes" under
LOEO on the official VB:

    Linear(t)         pure linear regression on order            (the control, ~3.1)
    PINN (t-only)     same PINN but with NO sensor features      (isolates the "neural-net tax")
    PINN_mono (x,t)   PINN with sensor features + order          (adds the "noisy-sensor tax", ~6.6)

  gap1 = PINN(t-only) - Linear(t)      -> cost of using a neural net to fit a line at n=10
  gap2 = PINN(x,t)    - PINN(t-only)   -> cost of ingesting noisy sensor features it cannot ignore

Writes results/diag_pinn_vs_line.csv, outputs/figures/diag_pinn_vs_line.png,
reports/diag_pinn_vs_line.md. Does NOT touch any official results/metrics or the `pipeline` task.

Uso:  python run.py diag-pinn-vs-line   [--repo-root <repo-root>] [--epochs 1500]
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
RESULTS = WT / "results"
FIGS = WT / "outputs" / "figures"
RECORDED = [66, 67, 68, 69, 70, 73, 74, 75, 76, 77]


def _mae(yt, yp):
    return float(np.mean(np.abs(np.asarray(yt, float) - np.asarray(yp, float))))


def load(repo_root):
    from phm.targets import load_official_vb
    f = pd.read_csv(repo_root / "data" / "processed" / "experiment_features.csv")
    f = f[f.experiment_id.isin(RECORDED)].copy()
    o = load_official_vb(recorded_only=True, data_root=repo_root / "data" / "targets")
    f["VB_um"] = f.experiment_id.map(dict(zip(o.experiment_id, o.VB_um)))
    return f.sort_values("experiment_order").reset_index(drop=True)


def linear_loeo(df, deg):
    t = df["experiment_order"].to_numpy(float); y = df["VB_um"].to_numpy(float)
    yp = np.zeros(len(df))
    for i in range(len(df)):
        m = np.arange(len(df)) != i
        yp[i] = np.polyval(np.polyfit(t[m], y[m], deg), t[i])
    return _mae(y, yp)


def pinn_loeo(df, x_cols, epochs):
    """LOEO PINN_mono. x_cols=None -> t-only (constant feature, no signal)."""
    from phm.pinn import PINNRegressor, PINN_VARIANTS, resolve_driver_col
    lam = PINN_VARIANTS["PINN_mono"]
    drv = resolve_driver_col(df.columns)
    t = df["experiment_order"].to_numpy(float); y = df["VB_um"].to_numpy(float)
    X = (df[x_cols].to_numpy(float) if x_cols else np.ones((len(df), 1)))
    e = df[drv].to_numpy(float) if drv is not None else None
    yp = np.zeros(len(df))
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        m = PINNRegressor(hidden=(32, 32), epochs=epochs, random_state=42, **lam)
        m.fit(X[tr], t[tr], y[tr], e_rot=(e[tr] if e is not None else None))
        yp[i] = m.predict(X[i:i + 1], t[i:i + 1])[0]
    return _mae(y, yp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--epochs", type=int, default=1500)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from phm.pinn import select_minimal_physical_features
    df = load(args.repo_root)
    x_min = select_minimal_physical_features(df.columns)

    lin = linear_loeo(df, 1)
    pol = linear_loeo(df, 2)
    pinn_t = pinn_loeo(df, None, args.epochs)
    pinn_xt = pinn_loeo(df, x_min, args.epochs)
    gap1 = pinn_t - lin
    gap2 = pinn_xt - pinn_t

    rows = [
        dict(model="Linear(t)", inputs="orden (t)", LOEO_MAE=round(lin, 2), note="control / regresion lineal"),
        dict(model="Poly2(t)", inputs="orden (t)", LOEO_MAE=round(pol, 2), note="control / polinomio grado 2"),
        dict(model="PINN (solo t)", inputs="orden (t), sin sensores", LOEO_MAE=round(pinn_t, 2),
             note="red ajustando la recta"),
        dict(model="PINN_mono (x,t)", inputs="orden (t) + vibracion", LOEO_MAE=round(pinn_xt, 2),
             note="titular 6.65"),
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "diag_pinn_vs_line.csv", index=False)

    # ---- figure: bars + the two taxes ----
    labels = ["Linear(t)\n(regresion)", "Poly2(t)", "PINN\n(solo t)", "PINN_mono\n(x, t)"]
    vals = [lin, pol, pinn_t, pinn_xt]
    cols = ["#1F4E79", "#4A6628", "#7A8CA3", "#B3541E"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.bar(labels, vals, color=cols, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    # decomposition annotations (sign-aware)
    g1_col = "#2E6F62" if gap1 < 0 else "#8C2D2D"
    g1_txt = (f"la red sola (t)\nMEJORA la recta\n{gap1:.1f} µm" if gap1 < 0
              else f"peaje de ser red\n+{gap1:.1f} µm")
    ax.annotate("", xy=(2, pinn_t), xytext=(0, lin),
                arrowprops=dict(arrowstyle="<->", color=g1_col, lw=1.4))
    ax.text(-0.30, max(vals) * 0.96, g1_txt, ha="left", va="top", fontsize=8.5, color=g1_col)
    ax.annotate("", xy=(3, pinn_xt), xytext=(2, pinn_t),
                arrowprops=dict(arrowstyle="<->", color="#8C2D2D", lw=1.4))
    ax.text(2.5, (pinn_t + pinn_xt) / 2 + 0.6, f"COSTO REAL:\nvibracion ruidosa\n+{gap2:.1f} µm",
            ha="center", fontsize=8.5, color="#8C2D2D")
    ax.set_ylabel("LOEO MAE (µm)  — menor es mejor")
    ax.set_title("¿Por qué la PINN pierde contra el tiempo si VB es casi una recta?\n"
                 "Descomposicion del gap (diagnostico, VB oficial)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "diag_pinn_vs_line.png", dpi=200,
                                    bbox_inches="tight", facecolor="white"); plt.close(fig)

    rep = f"""# Diagnostico — PINN vs recta de tiempo (descomposicion del gap)

Standalone (NO toca el pipeline oficial). Fuente: setup P8.1 (features + VB oficial + experiment_order).
epochs={args.epochs}. LOEO.

| Modelo | Entradas | LOEO MAE (µm) |
|---|---|---:|
| Linear(t) | orden | {lin:.2f} |
| Poly2(t) | orden | {pol:.2f} |
| PINN (solo t) | orden, sin sensores | {pinn_t:.2f} |
| PINN_mono (x,t) | orden + vibracion | {pinn_xt:.2f} |

## Descomposicion del gap (respuesta a la pregunta del supervisor)
- **Red vs recta** = PINN(solo t) − Linear(t) = **{gap1:+.1f} µm**.
  { "La PINN sola con el orden es MEJOR que la recta: la monotonia + flexibilidad capturan la leve "
    "curvatura de la trayectoria. La red NO es el problema." if gap1 < 0 else
    "La red ajusta la recta algo peor que una regresion lineal a n=10." }
- **Costo de la vibracion ruidosa** = PINN(x,t) − PINN(solo t) = **{gap2:+.1f} µm**: este es el costo
  real. Al sumar features de vibracion que no puede ignorar con ~9 puntos, el error sube.

## Conclusion (corregida por el experimento)
El problema NO es que "la PINN no sepa modelar la recta": sola con el orden es **excelente**
({pinn_t:.1f} µm, mejor que la recta {lin:.1f}). Pierde **solo al ingerir la vibracion ruidosa**
(+{gap2:.1f} µm). Es decir: en UNA sola herramienta los sensores son practicamente ruido frente al
orden (degeneracion temporal), y meterlos perjudica. El valor de los sensores solo aparecera con
varias herramientas (donde el orden deja de ser un proxy valido). Por eso la prueba real es **LOTO**.
(Nota: aqui Linear(t)={lin:.1f} usa la MISMA codificacion de orden que la PINN, para una comparacion
justa; el titular Linear(t)=3.10 usa la codificacion 'gapped' de P8.0 — distinta escala de t.)
"""
    (WT / "reports" / "diag_pinn_vs_line.md").write_text(rep, encoding="utf-8")
    print(f"Linear(t)={lin:.2f}  Poly2(t)={pol:.2f}  PINN(t-only)={pinn_t:.2f}  PINN(x,t)={pinn_xt:.2f}")
    print(f"gap1 (NN tax)={gap1:+.2f}  gap2 (noisy-sensor tax)={gap2:+.2f}")
    print(f"DONE in {time.time()-t0:.0f}s -> results/diag_pinn_vs_line.csv + fig + report")


if __name__ == "__main__":
    main()
